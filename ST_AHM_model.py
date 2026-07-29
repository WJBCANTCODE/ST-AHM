from __future__ import annotations
import datetime
import math
import numpy as np
import torch
from entmax import entmax_bisect
from torch import nn
import torch.nn.functional as F


def sparsemax(scores, dim=-1):
    if scores.numel() == 0:
        return scores

    shifted = scores - scores.max(dim=dim, keepdim=True).values
    sorted_scores = torch.sort(shifted, dim=dim, descending=True).values
    cumulative = sorted_scores.cumsum(dim)
    support_range = torch.arange(
        1,
        scores.size(dim) + 1,
        device=scores.device,
        dtype=scores.dtype,
    )
    shape = [1] * scores.dim()
    shape[dim] = -1
    support_range = support_range.view(shape)

    support = 1 + support_range * sorted_scores > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (
        cumulative.gather(dim, support_size - 1) - 1
    ) / support_size.to(scores.dtype)
    return torch.clamp(shifted - tau, min=0.0)


class HypergraphConvolution(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, incidence, node_features, node_mask=None, eps=1e-8):
        # incidence: [B, N, M]
        hyperedge_degree = incidence.sum(dim=1)  # [B, M]
        node_degree = incidence.sum(dim=2)       # [B, N]

        de_inv = torch.where(
            hyperedge_degree > 0,
            1.0 / hyperedge_degree.clamp_min(eps),
            torch.zeros_like(hyperedge_degree),
        )
        dv_inv_sqrt = torch.where(
            node_degree > 0,
            torch.rsqrt(node_degree.clamp_min(eps)),
            torch.zeros_like(node_degree),
        )

        weighted_h = incidence * de_inv.unsqueeze(1)
        propagation = torch.matmul(weighted_h, incidence.transpose(1, 2))
        propagation = (
            dv_inv_sqrt.unsqueeze(2)
            * propagation
            * dv_inv_sqrt.unsqueeze(1)
        )

        output = torch.matmul(propagation, node_features)
        output = F.relu(self.linear(output))
        if node_mask is not None:
            output = output * node_mask.unsqueeze(-1).to(output.dtype)
        return output


class SessionHSPA(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, node_mask=None):
        batch_size, n_nodes, hidden_size = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, n_nodes, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        head_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        consensus_scores = head_scores.mean(dim=1)  # [B, N, N]

        if node_mask is not None:
            valid_keys = node_mask.bool().unsqueeze(1)
            consensus_scores = consensus_scores.masked_fill(
                ~valid_keys, torch.finfo(consensus_scores.dtype).min
            )

        shared_weights = sparsemax(consensus_scores, dim=-1)
        if node_mask is not None:
            valid_queries = node_mask.to(shared_weights.dtype).unsqueeze(-1)
            shared_weights = shared_weights * valid_queries
        head_output = torch.matmul(shared_weights.unsqueeze(1), v)
        output = head_output.transpose(1, 2).contiguous().reshape(
            batch_size, n_nodes, hidden_size
        )
        output = self.proj(output)
        if node_mask is not None:
            output = output * node_mask.unsqueeze(-1).to(output.dtype)
        return output


class RelationGAT(nn.Module):
    """Target-aware aggregation over individual related sessions."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.target_projection = nn.Linear(2 * hidden_size, hidden_size)
        self.alpha_projection = nn.Linear(hidden_size, 1)
        self.attn_vector = nn.Parameter(torch.empty(1, hidden_size))
        self.neighbor_projection = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.query_projection = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.attn_bias = nn.Parameter(torch.zeros(hidden_size))

    @staticmethod
    def _avoid_exact_one(value):
        return value.masked_fill(value == 1, 1.00001)

    def get_alpha(self, target):
        return self._avoid_exact_one(torch.sigmoid(self.alpha_projection(target)) + 1)

    def forward(
        self,
        item_embedding,
        items,
        adjacency,
        degree_inverse,
        target_embedding,
        node_mask,
    ):
        item_vectors = item_embedding[items]  # [B, N, d]
        item_vectors = item_vectors * node_mask.unsqueeze(-1).to(item_vectors.dtype)
        session_vectors = item_vectors.sum(dim=1)  # [B, d]

        normalized_adjacency = torch.matmul(degree_inverse, adjacency)
        batch_size = session_vectors.size(0)
        neighbors = session_vectors.unsqueeze(0).expand(batch_size, -1, -1)
        propagated_neighbors = neighbors * normalized_adjacency.unsqueeze(-1)

        target = self.target_projection(target_embedding)  # [B, 1, d]
        scores = torch.matmul(
            F.relu(
                torch.matmul(propagated_neighbors, self.neighbor_projection)
                + torch.matmul(target, self.query_projection)
                + self.attn_bias
            ),
            self.attn_vector.t(),
        )  # [B, B, 1]

        neighbor_mask = adjacency.gt(0).unsqueeze(-1)
        scores = scores.masked_fill(~neighbor_mask, torch.finfo(scores.dtype).min)
        alpha = entmax_bisect(scores, self.get_alpha(target), dim=1)
        context = torch.matmul(alpha.transpose(1, 2), propagated_neighbors)
        context = F.selu(context).squeeze(1)
        return F.normalize(context, p=2, dim=-1, eps=1e-12)


class SessionGraph(nn.Module):
    def __init__(self, opt, n_node):
        super().__init__()
        self.dataset = opt.dataset
        self.hidden_size = opt.hiddenSize
        self.dim = 2 * self.hidden_size
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.beta = opt.beta
        self.w_hspa = opt.w_hspa
        self.num_attention_heads = opt.num_attention_heads
        self.attention_head_size = self.dim // self.num_attention_heads

        self.embedding = nn.Embedding(
            n_node, self.hidden_size, padding_idx=0, max_norm=1.5
        )
        self.pos_embedding = nn.Embedding(
            301, self.hidden_size, padding_idx=0, max_norm=1.5
        )
        self.embedding_norm = nn.LayerNorm(self.dim)
        self.hypergraph_conv = HypergraphConvolution(self.dim)
        self.hspa = SessionHSPA(self.dim, self.num_attention_heads)
        self.hspa_norm = nn.LayerNorm(self.dim)

        self.dropout = nn.Dropout(0.2)
        self.attention_mlp = nn.Linear(self.dim, self.dim)
        self.multi_alpha_w = nn.Linear(self.attention_head_size, 1)
        self.self_atten_w1 = nn.Linear(self.dim, self.dim)
        self.self_atten_w2 = nn.Linear(self.dim, self.dim)
        self.self_attention_norm = nn.LayerNorm(self.dim)

        self.alpha_w = nn.Linear(self.dim, 1)
        self.atten_w0 = nn.Parameter(torch.empty(1, self.dim))
        self.atten_w1 = nn.Parameter(torch.empty(self.dim, self.dim))
        self.atten_w2 = nn.Parameter(torch.empty(self.dim, self.dim))
        self.atten_bias = nn.Parameter(torch.zeros(self.dim))

        self.relation_graph = RelationGAT(self.hidden_size)
        self.decoder_projection = nn.Linear(4 * self.hidden_size, self.hidden_size)
        self.linear_one = nn.Linear(self.dim, self.dim)
        self.linear_two = nn.Linear(self.dim, self.dim)

        self.score_scale = 20.0
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=opt.lr, weight_decay=opt.l2
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=opt.lr_dc_step,
            gamma=opt.lr_dc,
        )
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                lower_name = name.lower()
                if lower_name.endswith("bias"):
                    parameter.zero_()
                elif "norm" in lower_name and lower_name.endswith("weight"):
                    parameter.fill_(1.0)
                else:
                    parameter.normal_(mean=0.0, std=0.1)
            self.embedding.weight[0].zero_()
            self.pos_embedding.weight[0].zero_()

    def add_position_embedding(self, items, node_mask):
        batch_size, n_nodes = items.shape

        positions = torch.arange(
            1, n_nodes + 1, dtype=torch.long, device=items.device
        ).unsqueeze(0).expand(batch_size, -1)
        positions = positions.masked_fill(~node_mask.bool(), 0)
        item_embeddings = self.embedding(items)
        position_embeddings = self.pos_embedding(positions)
        sequence_embeddings = torch.cat(
            [item_embeddings, position_embeddings], dim=-1
        )
        sequence_embeddings = self.embedding_norm(sequence_embeddings)
        return sequence_embeddings * node_mask.unsqueeze(-1).to(
            sequence_embeddings.dtype
        )

    @staticmethod
    def _avoid_exact_one(value):
        return value.masked_fill(value == 1, 1.00001)

    def get_global_alpha(self, target):
        return self._avoid_exact_one(torch.sigmoid(self.alpha_w(target)) + 1)

    def get_self_attention_alpha(self, blank_query, sequence_length):
        alpha = torch.sigmoid(self.multi_alpha_w(blank_query)) + 1
        alpha = self._avoid_exact_one(alpha).unsqueeze(2)
        return alpha.expand(-1, -1, sequence_length, -1)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        return x.view(*new_shape).permute(0, 2, 1, 3)

    def multi_self_attention(self, q, k, v, valid_mask):
        q_projected = self.dropout(F.relu(self.attention_mlp(q)))
        query_layer = self.transpose_for_scores(q_projected)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores / math.sqrt(self.attention_head_size)
        key_mask = valid_mask.bool().unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        alpha = self.get_self_attention_alpha(
            query_layer[:, :, -1, :], q.size(1)
        )
        attention = entmax_bisect(scores, alpha, dim=-1)
        context = torch.matmul(attention, value_layer)
        context = context.permute(0, 2, 1, 3).contiguous().view(
            q.size(0), q.size(1), self.dim
        )
        context = (
            self.dropout(self.self_atten_w2(F.relu(self.self_atten_w1(context))))
            + context
        )
        context = self.self_attention_norm(context)
        context = context * valid_mask.unsqueeze(-1).to(context.dtype)
        return context[:, -1:, :], context[:, :-1, :]

    def global_attention(self, target, keys, values, mask):
        scores = torch.matmul(
            F.relu(
                torch.matmul(keys, self.atten_w1)
                + torch.matmul(target, self.atten_w2)
                + self.atten_bias
            ),
            self.atten_w0.t(),
        )
        scores = scores.masked_fill(
            ~mask.bool().unsqueeze(-1), torch.finfo(scores.dtype).min
        )
        alpha = entmax_bisect(scores, self.get_global_alpha(target), dim=1)
        return torch.matmul(alpha.transpose(1, 2), values)

    def decoder(self, local_context, target):
        fused = torch.cat([local_context, target], dim=2)
        decoded = self.dropout(F.selu(self.decoder_projection(fused))).squeeze(1)
        return F.normalize(decoded, p=2, dim=-1, eps=1e-12)

    def compute_scores(self, hidden, mask, target_emb, attended_hidden, relation_emb):
        batch_index = torch.arange(mask.size(0), device=hidden.device)
        last_index = mask.sum(dim=1).long() - 1
        last_hidden = hidden[batch_index, last_index]
        q1 = self.linear_one(last_hidden).unsqueeze(1)
        q2 = self.linear_two(hidden)
        local_values = torch.sigmoid(q1 + q2)

        local_context = self.global_attention(
            target_emb, attended_hidden, local_values, mask
        )
        local_session = self.decoder(local_context, target_emb)
        fused_session = F.normalize(
            local_session + relation_emb, p=2, dim=-1, eps=1e-12
        )

        candidate_items = F.normalize(
            self.embedding.weight[1:], p=2, dim=-1, eps=1e-12
        )
        return self.score_scale * torch.matmul(
            fused_session, candidate_items.transpose(0, 1)
        )

    @staticmethod
    def contrastive_loss(hidden, target_emb, sequence_mask):
        weights = sequence_mask.unsqueeze(-1).to(hidden.dtype)
        local_context = (hidden * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)
        target = target_emb.squeeze(1)

        positive_score = (local_context * target).sum(dim=-1)
        batch_size = local_context.size(0)
        if batch_size < 2:
            return hidden.new_zeros(())
        shift = int(torch.randint(1, batch_size, (1,), device=hidden.device))
        corrupted_context = torch.roll(local_context, shifts=shift, dims=0)
        negative_score = (corrupted_context * target).sum(dim=-1)

        positive_loss = F.logsigmoid(positive_score)
        negative_loss = F.logsigmoid(-negative_score)
        return -(positive_loss + negative_loss).mean()

    def forward(
        self,
        items,
        incidence,
        alias_inputs,
        relation_adjacency,
        relation_degree_inverse,
        node_mask,
        sequence_mask,
        compute_contrastive=True,
    ):
        node_embeddings = self.add_position_embedding(items, node_mask)
        hidden_hg = self.hypergraph_conv(
            incidence, node_embeddings, node_mask=node_mask
        )
        hidden_hspa = self.hspa(hidden_hg, node_mask=node_mask)
        hidden_fused = self.hspa_norm(
            hidden_hg + self.w_hspa * hidden_hspa
        )
        hidden_fused = hidden_fused * node_mask.unsqueeze(-1).to(
            hidden_fused.dtype
        )

        batch_index = torch.arange(items.size(0), device=items.device).unsqueeze(1)
        sequence_hidden = hidden_fused[batch_index, alias_inputs]
        sequence_hidden = sequence_hidden * sequence_mask.unsqueeze(-1).to(
            sequence_hidden.dtype
        )

        blank = sequence_hidden.new_zeros(sequence_hidden.size(0), 1, self.dim)
        augmented = torch.cat([sequence_hidden, blank], dim=1)
        augmented_mask = torch.cat(
            [
                sequence_mask,
                torch.ones(
                    sequence_mask.size(0),
                    1,
                    dtype=sequence_mask.dtype,
                    device=sequence_mask.device,
                ),
            ],
            dim=1,
        )
        target_emb, attended_hidden = self.multi_self_attention(
            augmented, augmented, augmented, augmented_mask
        )

        relation_emb = self.relation_graph(
            self.embedding.weight,
            items,
            relation_adjacency,
            relation_degree_inverse,
            target_emb,
            node_mask,
        )

        if compute_contrastive:
            contrastive = self.contrastive_loss(
                sequence_hidden, target_emb, sequence_mask
            )
        else:
            contrastive = sequence_hidden.new_zeros(())

        return (
            sequence_hidden,
            target_emb,
            attended_hidden,
            relation_emb,
            self.beta * contrastive,
        )


def trans_to_cuda(variable):
    return variable.cuda() if torch.cuda.is_available() else variable


def trans_to_cpu(variable):
    return variable.cpu() if variable.is_cuda else variable


def forward(model, indices, data, compute_contrastive=True):
    (
        alias_inputs,
        incidence,
        items,
        node_mask,
        sequence_mask,
        targets,
    ) = data.get_slice(indices)
    adjacency, degree_inverse = data.get_overlap(items)

    alias_inputs = trans_to_cuda(torch.as_tensor(alias_inputs, dtype=torch.long))
    incidence = trans_to_cuda(torch.as_tensor(incidence, dtype=torch.float32))
    items = trans_to_cuda(torch.as_tensor(items, dtype=torch.long))
    node_mask = trans_to_cuda(torch.as_tensor(node_mask, dtype=torch.long))
    sequence_mask = trans_to_cuda(
        torch.as_tensor(sequence_mask, dtype=torch.long)
    )
    adjacency = trans_to_cuda(torch.as_tensor(adjacency, dtype=torch.float32))
    degree_inverse = trans_to_cuda(
        torch.as_tensor(degree_inverse, dtype=torch.float32)
    )

    hidden, target_emb, attended_hidden, relation_emb, contrastive = model(
        items,
        incidence,
        alias_inputs,
        adjacency,
        degree_inverse,
        node_mask,
        sequence_mask,
        compute_contrastive=compute_contrastive,
    )
    scores = model.compute_scores(
        hidden, sequence_mask, target_emb, attended_hidden, relation_emb
    )
    return targets, scores, contrastive


def train_one_epoch(model, train_data):
    print("start training:", datetime.datetime.now())
    model.train()
    total_loss = 0.0
    batches = train_data.generate_batch(model.batch_size)

    for batch_number, indices in enumerate(batches):
        model.optimizer.zero_grad()
        targets, scores, contrastive = forward(
            model, indices, train_data, compute_contrastive=True
        )
        targets_tensor = trans_to_cuda(torch.as_tensor(targets, dtype=torch.long))
        loss = model.loss_function(scores, targets_tensor - 1) + contrastive
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()

        report_every = max(1, len(batches) // 5)
        if batch_number % report_every == 0:
            print(f"[{batch_number}/{len(batches)}] Loss: {loss.item():.4f}")

    model.scheduler.step()
    return total_loss / max(1, len(batches))


def evaluate(model, data):
    print("start evaluation:", datetime.datetime.now())
    model.eval()
    hit_20, mrr_20, hit_10, mrr_10 = [], [], [], []
    total_loss = 0.0
    batches = data.generate_batch(model.batch_size)

    with torch.no_grad():
        for indices in batches:
            targets, scores, _ = forward(
                model, indices, data, compute_contrastive=False
            )
            target_tensor = trans_to_cuda(
                torch.as_tensor(targets, dtype=torch.long)
            )
            total_loss += model.loss_function(scores, target_tensor - 1).item()

            top_k = min(20, scores.size(1))
            top_20 = trans_to_cpu(scores.topk(top_k, dim=1).indices).numpy()
            top_10 = top_20[:, : min(10, top_k)]

            for predictions_20, predictions_10, target in zip(
                top_20, top_10, targets
            ):
                target_index = target - 1
                hit20 = np.isin(target_index, predictions_20)
                hit10 = np.isin(target_index, predictions_10)
                hit_20.append(float(hit20))
                hit_10.append(float(hit10))

                position_20 = np.where(predictions_20 == target_index)[0]
                position_10 = np.where(predictions_10 == target_index)[0]
                mrr_20.append(
                    0.0 if len(position_20) == 0 else 1.0 / (position_20[0] + 1)
                )
                mrr_10.append(
                    0.0 if len(position_10) == 0 else 1.0 / (position_10[0] + 1)
                )

    metrics = {
        "P@20": 100.0 * float(np.mean(hit_20)),
        "MRR@20": 100.0 * float(np.mean(mrr_20)),
        "P@10": 100.0 * float(np.mean(hit_10)),
        "MRR@10": 100.0 * float(np.mean(mrr_10)),
        "loss": total_loss / max(1, len(batches)),
    }
    return metrics


def train_test(model, train_data, test_data):
    train_one_epoch(model, train_data)
    metrics = evaluate(model, test_data)
    return (
        metrics["P@20"],
        metrics["MRR@20"],
        metrics["P@10"],
        metrics["MRR@10"],
        metrics["P@10"],
    )
