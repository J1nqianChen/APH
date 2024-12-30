import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader
from utils import *
from torch import nn

algo = 'FedAvg'
num_classes = 10
num_clients = 10
dataset_name = 'Cifar10_NonIID_dir_0.1_Client_10'
root_path = './ckpt/'
model_pt_name = 'FedAvg_gr_100_ls_10_jr_1.0_CNN_FedAvg_NonIID_NonBalance_jr_1.0_dir_0.1_best.pt'
save_file_path = ''

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


def evaluate_from_multi_heads(model, heads_list, c_testloader_list, weight_arr, nc=num_classes):
    num_clients = len(c_testloader_list)
    list_out = []
    for i in range(num_clients):
        heads = heads_list[i]
        out = evaluate_aph(model, c_testloader_list[i], heads, num_classes=nc)
        list_out.append(out)
    result = aggregate_out(weight_arr, list_out)
    return result

def evaluate_from_multi_heads_pfl(model_list, heads_list, c_testloader_list, weight_arr, nc=num_classes):
    num_clients = len(c_testloader_list)
    list_out = []
    for i in range(num_clients):
        model = model_list[i]
        heads = heads_list[i]
        out = evaluate_aph(model, c_testloader_list[i], heads, num_classes=nc)
        list_out.append(out)
    result = aggregate_out(weight_arr, list_out)
    return result




if __name__ == '__main__':

    print(save_file_path)
    with open(save_file_path, "rb") as fp:  # Pickling
        heads_list = pickle.load(fp)

    c_testloader_list = []
    c_trainloader_list = []
    c_test_sample_list = []
    for i in range(num_clients):
        c_train_dataset, c_test_dataset, c_test_sample = select_read(dataset_name, [i])
        c_test_loader = DataLoader(c_test_dataset, batch_size=1024, shuffle=False)
        c_train_loader = DataLoader(c_train_dataset, batch_size=128, shuffle=True)
        c_test_sample_list.append(c_test_sample)
        c_testloader_list.append(c_test_loader)
        c_trainloader_list.append(c_train_loader)
    c_weight = np.array(c_test_sample_list) / np.array(c_test_sample_list).sum()


    PFL_flag = True
    # PFL_flag = False
    model_dir_path = os.path.join(root_path, dataset_name)
    if not PFL_flag:
        total_path = os.path.join(model_dir_path, model_pt_name)
        model = torch.load(total_path)
        model = rearrange_model_all(model)
        time1 = time.time()
        result = evaluate_from_multi_heads(model, heads_list, c_testloader_list, c_weight, nc=num_classes)
    else:
        c_model_list = []
        for i in range(num_clients):
            c_model_name = f'{model_pt_name}_client_{i}.pt'
            total_path = os.path.join(model_dir_path, c_model_name)
            model = torch.load(total_path)
            model = rearrange_model_all(model)
            c_model_list.append(model)
        time1 = time.time()
        result = evaluate_from_multi_heads_pfl(c_model_list, heads_list, c_testloader_list, c_weight, nc=num_classes)
    time2 = time.time()
    print(result['acc'])
    print(result['p_ece'])
    print(result['nll'])
    print(result['F_KDE_ECE'])
    print(result['F_L2_CE'])

    # print(time2 - time1)