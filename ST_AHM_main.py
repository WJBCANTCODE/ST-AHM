#!/usr/bin/env python36
# -*- coding: utf-8 -*-



import argparse
import pickle
import time
from ST_AHM_utils import Data, split_validation
from ST_AHM_model import *
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: /Tmall/Nowplaying/diginetica/yoochoose1_4/yoochoose1_64')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size') #64,100,256,512,TMALL数据集可以增大(128,256)，30music为128(128,128)
parser.add_argument('--hiddenSize', type=int, default=256, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=5, help='the number of steps after which the learning rate decay 3')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=3, help='the number of epoch to wait before early stop ')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--num_attention_heads', type=int, default=8, help='Multi-Att heads')
parser.add_argument('--w_hspa', type=float, default=0.01, help='HSPA与GNN的融合权重')
parser.add_argument('--beta', type=float, default=0.001, help='对比学习权重系数')#
parser.add_argument('--w_score', type=int, default=20, help='Weight for the final score computation')
opt = parser.parse_args()
print(opt)
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#torch.cuda.set_device(0)

def main():
    train_data = pickle.load(open('/home/Data8TDisk/experiment/WJB/lunwen/datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('/home/Data8TDisk/experiment/WJB/lunwen/datasets/' + opt.dataset + '/test.txt', 'rb'))
    # del all_train_seq, g
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    elif opt.dataset == 'Nowplaying':
        n_node = 60417
    elif opt.dataset == 'Tmall':
        n_node = 40728
    elif opt.dataset == 'RetailRocket':
        n_node = 36969
    elif opt.dataset == '30music':
        n_node = 36980
    else:
        n_node = 310

    start = time.time()
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    model = trans_to_cuda(SessionGraph(opt, n_node))

    best_result = [0, 0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit_20, mrr_20, hit_10, mrr_10, p_10 = train_test(model, train_data, test_data)
        flag = 0
        if hit_20 >= best_result[0]:
            best_result[0] = hit_20
            best_epoch[0] = epoch
            flag = 1
        if mrr_20 >= best_result[1]:
            best_result[1] = mrr_20
            best_epoch[1] = epoch
            flag = 1
        if hit_10 >= best_result[2]:
            best_result[2] = hit_10
            best_epoch[2] = epoch
            flag = 1
        if mrr_10 >= best_result[3]:
            best_result[3] = mrr_10
            best_epoch[3] = epoch
            flag = 1
        if p_10 >= best_result[4]:
            best_result[4] = p_10
            best_epoch[4] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d' % (best_result[0], best_result[1], best_epoch[0]))
        print('\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tP@10:\t%.4f\tEpoch:\t%d,\t%d,\t%d' % (
            best_result[2], best_result[3], best_result[4], best_epoch[2], best_epoch[3], best_epoch[4]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()