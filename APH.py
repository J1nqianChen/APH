"""
Federated Assemble Projection Heads
Official Implementation
"""
import sys

# Add the directory path to sys.path
new_path = '../'
sys.path.append(new_path)
sys.path.append('../PFL/system')

import torch.nn.functional as F
from PFL.system.flcore.trainmodel.models import BaseHeadSplit
import argparse
import copy
import os
import pickle
import random

import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from PFL.dataset.Read_Data import select_read
from Uncertainty.uncertainty_utils import evaluate_, evaluate_from_head, aggregate_out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rearrange_model_all(model):
    head = copy.deepcopy(model.fc)
    model.fc = nn.Identity()
    if hasattr(model, 'fc1'):
        head1 = copy.deepcopy(model.fc1)
        model.fc1 = nn.Identity()
        head_all = nn.Sequential(head1, head)
    else:
        head_all = nn.Sequential(head)

    model = BaseHeadSplit(model, head_all)
    return model


def fine_tune_freeze(model):
    model.base.eval()
    model.head.train()
    for param in model.base.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True


def random_initialize(model_temp, lamda=1):
    for param in model_temp.parameters():
        param_mean = torch.mean(param.data).item()
        magnitude = order_of_magnitude(param_mean)
        weight = torch.randn_like(param.data) * pow(10, magnitude + lamda)
        param.data = param.data + weight


def get_multi_heads(dataset_name, model, num_clients, epoch_num=10,
                    sampling_num=10, upper_lr=1, lower_lr=0.0001, lamda=1):
    loss_func = nn.CrossEntropyLoss()

    list_sample_num = []
    total_heads_list = []
    for i in range(num_clients):
        head_list = []
        list_client = [i]
        train_dataset, test_dataset, sample_number = select_read(dataset_name, list_client)
        list_sample_num.append(sample_number)
        train_loader = DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=True)

        upper_limit = upper_lr
        lower_limit = lower_lr

        # Sample from the logarithmic space
        log_upper = math.log10(upper_limit)
        log_lower = math.log10(lower_limit)
        log_samples = [random.uniform(log_lower, log_upper) for _ in range(sampling_num)]

        # Convert the samples back to the original space
        samples_lr = [10 ** log_sample for log_sample in log_samples]

        # samples_lr = [0.01 for _ in range(sampling_num)]

        for j in range(sampling_num):
            lr_ = samples_lr[j]
            model_temp = copy.deepcopy(model)
            random_initialize(model_temp, lamda)

            optimizer = torch.optim.SGD(lr=lr_, weight_decay=1e-5, params=model_temp.parameters())

            for epoch in range(epoch_num):
                fine_tune_freeze(model_temp)
                for x, y in tqdm(train_loader):
                    x = x.cuda()
                    y = y.cuda()

                    optimizer.zero_grad()
                    output = model_temp(x)
                    loss = loss_func(output, y)

                    loss.backward()
                    optimizer.step()
            head_list.append(copy.deepcopy(model_temp.head))
        total_heads_list.append(head_list)
    return total_heads_list


def get_multi_heads_pfl(dataset_name, model_list, num_clients, epoch_num=10,
                    sampling_num=10, upper_lr=1, lower_lr=0.0001, lamda=1):
    loss_func = nn.CrossEntropyLoss()

    list_sample_num = []
    total_heads_list = []
    for i in range(num_clients):
        head_list = []
        list_client = [i]
        train_dataset, test_dataset, sample_number = select_read(dataset_name, list_client)
        list_sample_num.append(sample_number)
        train_loader = DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=True)

        upper_limit = upper_lr
        lower_limit = lower_lr

        # Sample from the logarithmic space
        log_upper = math.log10(upper_limit)
        log_lower = math.log10(lower_limit)
        log_samples = [random.uniform(log_lower, log_upper) for _ in range(sampling_num)]

        # Convert the samples back to the original space
        samples_lr = [10 ** log_sample for log_sample in log_samples]

        # samples_lr = [0.01 for _ in range(sampling_num)]

        for j in range(sampling_num):
            lr_ = samples_lr[j]
            model_temp = copy.deepcopy(model_list[i])
            random_initialize(model_temp, lamda)

            optimizer = torch.optim.SGD(lr=lr_, weight_decay=1e-5, params=model_temp.parameters())

            for epoch in range(epoch_num):
                fine_tune_freeze(model_temp)
                for x, y in tqdm(train_loader):
                    x = x.cuda()
                    y = y.cuda()

                    optimizer.zero_grad()
                    output = model_temp(x)
                    loss = loss_func(output, y)

                    loss.backward()
                    optimizer.step()
            head_list.append(copy.deepcopy(model_temp.head))
        total_heads_list.append(head_list)
    return total_heads_list


def order_of_magnitude(parameter):
    scientific_notation = "{:e}".format(parameter)
    order_of_magnitude = int(scientific_notation.split('e')[1])

    return order_of_magnitude


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', "--seed", required=True, default=1234, type=int)
    parser.add_argument('-rtp', "--root_path",
                        default='/home/chenjinqian/code/MINE_FL/PFL/system/models', type=str)
    parser.add_argument('-dp', "--dir_path", required=True, default='CNN_FedAvg_NonIID_NonBalance_jr_0.2_dir_0.1',
                        type=str)
    parser.add_argument('-data', "--dataset_name", required=True, default=None, type=str)
    parser.add_argument('-nc', "--num_clients", required=True, default=None, type=int)
    parser.add_argument('-nb', "--num_classes", required=True, default=None, type=int)
    parser.add_argument('-ae', "--aph_epoch", required=True, default=10, type=int)
    parser.add_argument('-sn', "--sampling_num", required=True, default=10, type=int)
    parser.add_argument('-go', "--goal", default=None, type=str)
    parser.add_argument('-mn', "--model_name", default=None, required=True, type=str)
    parser.add_argument('-did', "--device_id", default=0, type=str)
    parser.add_argument('-ulr', "--upper_lr", default=1, type=float)
    parser.add_argument('-llr', "--lower_lr", default=0.0001, type=float)
    parser.add_argument('-algo', "--algorithm", default='FedAvg', type=str)
    parser.add_argument('-lamda', "--lamda", default=1, type=float)
    parser.add_argument('-pr', "--participation_ratio", default=1.0, type=float)
    parser.add_argument('-pfl', "--pfl", action="store_true")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    return args



if __name__ == '__main__':
    args = parser_args()
    print(args)
    set_seed(args.seed)
    lamda = args.lamda
    algo = args.algorithm

    root_path = args.root_path

    dir_path = args.dir_path
    dataset_name = args.dataset_name
    num_clients = args.num_clients
    global_epoch = args.aph_epoch
    num_classes = args.num_clients
    sampling_num = args.sampling_num

    keyword = args.goal

    save_file_path = f'APH_{algo}_{dataset_name}_{dir_path}_Epoch_{global_epoch}_HeadsNum_{sampling_num}_ulr_{args.upper_lr}_llr_{args.lower_lr}_seed_{args.seed}_lamda_{lamda}.pkl'

    model_dir_path = os.path.join(root_path, dataset_name, dir_path)
    model_pt_name = args.model_name


    if args.pfl:
        c_model_list = []
        for i in range(num_clients):
            c_model_name = f'{model_pt_name}_client_{i}.pt'
            total_path = os.path.join(model_dir_path, c_model_name)
            model = torch.load(total_path)
            model = rearrange_model_all(model)
            c_model_list.append(copy.deepcopy(model))

        c_list_heads = get_multi_heads_pfl(dataset_name, c_model_list, epoch_num=global_epoch,
                                       num_clients=num_clients, sampling_num=sampling_num, upper_lr=args.upper_lr,
                                       lower_lr=args.lower_lr, lamda=lamda)
    else:
        total_path = os.path.join(model_dir_path, model_pt_name)
        model = torch.load(total_path)
        model = rearrange_model_all(model)

        c_list_heads = get_multi_heads(dataset_name, model, epoch_num=global_epoch,
                                       num_clients=num_clients, sampling_num=sampling_num, upper_lr=args.upper_lr,
                                       lower_lr=args.lower_lr, lamda=lamda)

    with open(save_file_path, "wb") as fp1:  # Pickling
        pickle.dump(c_list_heads, fp1)
