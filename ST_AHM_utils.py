#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Data utilities for the manuscript-aligned ST-AHM implementation."""

from __future__ import annotations

import numpy as np


def data_masks(all_usr_pois, item_tail):
    """Pad input prefixes and return their binary validity masks."""
    if not all_usr_pois:
        raise ValueError("The dataset contains no sessions.")
    us_lens = [len(upois) for upois in all_usr_pois]
    if min(us_lens) < 1:
        raise ValueError("Every session prefix must contain at least one item.")
    len_max = max(us_lens)
    us_pois = [
        list(upois) + item_tail * (len_max - length)
        for upois, length in zip(all_usr_pois, us_lens)
    ]
    us_masks = [
        [1] * length + [0] * (len_max - length)
        for length in us_lens
    ]
    return us_pois, us_masks, len_max


def split_validation(train_set, valid_portion, seed=2025):
    """Randomly reserve ``valid_portion`` of the training samples for validation."""
    if not 0.0 < valid_portion < 1.0:
        raise ValueError("valid_portion must be between 0 and 1.")

    train_set_x, train_set_y = train_set
    if len(train_set_x) != len(train_set_y):
        raise ValueError("Training inputs and labels have different lengths.")

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(train_set_x))
    n_train = int(np.round(len(train_set_x) * (1.0 - valid_portion)))
    train_indices = shuffled_indices[:n_train]
    valid_indices = shuffled_indices[n_train:]

    train_x = [train_set_x[i] for i in train_indices]
    train_y = [train_set_y[i] for i in train_indices]
    valid_x = [train_set_x[i] for i in valid_indices]
    valid_y = [train_set_y[i] for i in valid_indices]
    return (train_x, train_y), (valid_x, valid_y)


def session_hyperedges(sequence):
    """Create the original model's non-overlapping within-session hyperedges."""
    valid_sequence = np.asarray(sequence, dtype=np.int64)
    valid_sequence = valid_sequence[valid_sequence != 0]
    if valid_sequence.size == 0:
        raise ValueError("Cannot build a hypergraph for an empty session.")

    hyperedge_size = max(2, int(np.ceil(valid_sequence.size / 2.0)))
    hyperedges = []
    pending_singletons = []
    for start in range(0, valid_sequence.size, hyperedge_size):
        hyperedge = np.unique(valid_sequence[start : start + hyperedge_size])
        if hyperedge.size > 1:
            if pending_singletons:
                hyperedge = np.unique(
                    np.concatenate(
                        [np.asarray(pending_singletons, dtype=np.int64), hyperedge]
                    )
                )
                pending_singletons = []
            hyperedges.append(hyperedge)
        else:
            pending_singletons.extend(hyperedge.tolist())
    if pending_singletons and hyperedges:
        hyperedges[-1] = np.unique(
            np.concatenate(
                [hyperedges[-1], np.asarray(pending_singletons, dtype=np.int64)]
            )
        )
    if not hyperedges:
        hyperedges.append(np.unique(valid_sequence))
    return hyperedges


def build_incidence(sequence, session_nodes, max_n_node, max_n_edge):
    """Build an aligned node-hyperedge incidence matrix for one session.

    The original code's within-session hyperedge segmentation is retained so
    the overall model design is not replaced. Only the node/padding alignment
    error is corrected. Hyperedges use positive unit weights; HGNN degree
    normalization is performed in the model.
    """
    session_nodes = np.asarray(session_nodes, dtype=np.int64)
    hyperedges = session_hyperedges(sequence)

    incidence = np.zeros((max_n_node, max_n_edge), dtype=np.float32)
    node_to_row = {int(item): row for row, item in enumerate(session_nodes)}
    for edge_index, hyperedge in enumerate(hyperedges):
        for item in hyperedge:
            incidence[node_to_row[int(item)], edge_index] = 1.0
    return incidence


class Data():
    """数据处理类"""
    def __init__(self, data, shuffle=False, hypergraph=None):
        """
        初始化数据对象
        Args:
            data: 输入数据
            shuffle: 是否打乱数据
            hypergraph: 超图结构数据
        """
        inputs = data[0]  # 获取输入序列
        inputs, mask, len_max = data_masks(inputs, [0])  # 创建掩码
        self.inputs = np.asarray(inputs)  # 转换为numpy数组
        self.mask = np.asarray(mask)  # 转换掩码为numpy数组
        self.len_max = len_max  # 保存最大序列长度
        self.targets = np.asarray(data[1])  # 转换目标值为numpy数组
        self.length = len(inputs)  # 数据长度
        self.shuffle = shuffle  # 是否打乱
        self.hypergraph = hypergraph  # 超图结构

    @staticmethod
    def get_overlap(sessions):
        """Return Jaccard session similarity and row-degree normalization."""
        n_sessions = len(sessions)
        matrix = np.zeros((n_sessions, n_sessions), dtype=np.float32)
        session_sets = []
        for session in sessions:
            item_set = set(np.asarray(session).tolist())
            item_set.discard(0)
            session_sets.append(item_set)

        for i in range(n_sessions):
            for j in range(i + 1, n_sessions):
                union = session_sets[i] | session_sets[j]
                similarity = (
                    float(len(session_sets[i] & session_sets[j])) / float(len(union))
                    if union else 0.0
                )
                matrix[i, j] = similarity
                matrix[j, i] = similarity

        matrix += np.eye(n_sessions, dtype=np.float32)
        degree = matrix.sum(axis=1)
        degree_inv = np.diag(1.0 / np.maximum(degree, 1e-12)).astype(np.float32)
        return matrix, degree_inv

    def generate_batch(self, batch_size):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.shuffle:
            shuffled = np.random.permutation(self.length)
            self.inputs = self.inputs[shuffled]
            self.mask = self.mask[shuffled]
            self.targets = self.targets[shuffled]

        n_batch = int(np.ceil(self.length / batch_size))
        return [
            np.arange(start, min(start + batch_size, self.length))
            for start in range(0, self.length, batch_size)
        ] if n_batch else []

    def get_slice(self, indices):
        """Return one batch with aligned node, incidence, alias, and masks.

        Returns
        -------
        alias_inputs : ndarray [B, L]
            Maps every sequence position to its unique-node row. Padding maps
            to the first dedicated zero row after the real nodes.
        H_batch : ndarray [B, N, M]
            Within-session node-hyperedge incidence matrices.
        items : ndarray [B, N]
            Unique real item ids followed by zero padding.
        node_mask : ndarray [B, N]
            Validity mask for unique-node rows.
        sequence_mask : ndarray [B, L]
            Validity mask for original sequence positions.
        targets : ndarray [B]
            Ground-truth next-item ids; never passed into the model encoder.
        """
        inputs = self.inputs[indices]
        sequence_mask = self.mask[indices]
        targets = self.targets[indices]

        node_lists = [np.unique(seq[seq != 0]) for seq in inputs]
        hyperedge_lists = [session_hyperedges(seq) for seq in inputs]
        if any(nodes.size == 0 for nodes in node_lists):
            raise ValueError("Encountered an empty session after padding removal.")

        # Reserve at least one node row for padding so H/items/alias stay aligned.
        max_n_node = max(nodes.size + 1 for nodes in node_lists)
        max_n_edge = max(len(edges) for edges in hyperedge_lists)
        items, node_masks, incidences, aliases = [], [], [], []

        for sequence, nodes in zip(inputs, node_lists):
            n_real = nodes.size
            padding_count = max_n_node - n_real
            item_rows = np.pad(nodes, (0, padding_count), constant_values=0)
            node_mask = np.concatenate(
                [np.ones(n_real, dtype=np.int64), np.zeros(padding_count, dtype=np.int64)]
            )
            incidence = build_incidence(
                sequence, nodes, max_n_node, max_n_edge
            )

            node_to_row = {int(item): row for row, item in enumerate(nodes)}
            padding_row = n_real
            alias = [
                node_to_row[int(item)] if item != 0 else padding_row
                for item in sequence
            ]

            items.append(item_rows)
            node_masks.append(node_mask)
            incidences.append(incidence)
            aliases.append(alias)

        return (
            np.asarray(aliases, dtype=np.int64),
            np.asarray(incidences, dtype=np.float32),
            np.asarray(items, dtype=np.int64),
            np.asarray(node_masks, dtype=np.int64),
            sequence_mask,
            targets,
        )
