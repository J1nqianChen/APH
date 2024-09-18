import os
import time

from Uncertainty.APH_ood_eval import check_pfl

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader

from Read_Data import select_read
from Uncertainty.Fine_Tune_Head import rearrange_model_all
from Uncertainty.uncertainty_utils import evaluate_from_head, aggregate_out, evaluate_aph

# algo = 'FedAvg'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'CNN_FedAvg_NonIID_NonBalance_jr_1.0_dir_0.1'
# model_pt_name = f'FedAvg_gr_100_ls_10_jr_1.0_CNN_FedAvg_NonIID_NonBalance_jr_1.0_dir_0.1_best.pt'
# num_clients = 10
# global_epoch = 10
# num_classes = 10
# dataset_name = f'Cifar10_NonIID_dir_0.1_Client_{num_clients}'

# algo = 'FedAvg'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'CNN_FedAvg_NonIID_NonBalance_jr_1.0_dir_0.1'
# model_pt_name = f'FedAvg_gr_100_ls_10_jr_1.0_CNN_FedAvg_NonIID_NonBalance_jr_1.0_dir_0.1_best.pt'
# num_clients = 10
# global_epoch = 10
# num_classes = 10
# dataset_name = f'Cifar10_NonIID_dir_0.1_Client_{num_clients}'
# save_file_path = ''


# Cifar100 ResNet
# algo = 'FedAvg'
# dataset_name = f'Cifar100_NonIID_dir_0.1_Client_10'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'ResNet50_FedAvg_NonIID_NonBalance_jr1.0'
# model_pt_name = f'FedAvg_gr_100_ls_10_jr_1.0_ResNet50_FedAvg_NonIID_NonBalance_jr1.0_best.pt'
# num_classes = 100
# num_clients = 10
# global_epoch = 10
# save_file_path = 'APH_FedAvg_Cifar100_NonIID_dir_0.1_Client_10_PR_1.0_Epoch_10_HeadsNum_10_ulr_1000.0_llr_0.001_seed_2345_lamda_0.2.pkl'
# save_file_path = 'APH_ResNet50_FedAvg_NonIID_NonBalance_jr1.0_Epoch_10_HeadsNum_50_ulr_1000.0_llr_0.001_seed_2345_lamda_0.2.pkl'
# save_file_path = 'APH_ResNet50_FedAvg_NonIID_NonBalance_jr1.0_Epoch_10_HeadsNum_100_ulr_1000.0_llr_0.001_seed_2345_lamda_0.2.pkl'


# # Tiny-ImageNet ResNet
# algo = 'FedAvg'
# dataset_name = f'Tiny-imagenet'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'ResNet_FedAvg_NonIID_NonBalance_jr1.0'
# model_pt_name = f'FedAvg_gr_20_ls_10_jr_1.0_ResNet_FedAvg_NonIID_NonBalance_jr1.0_best.pt'
# num_classes = 200
# num_clients = 10
# global_epoch = 10
# save_file_path = 'APH_FedAvg_Tiny-imagenet_PR_1.0_Epoch_10_HeadsNum_10_ulr_0.1_llr_0.001_seed_2345_lamda_0.1.pkl'
#

###### FedProx

# FedProx Cifar10
# algo = 'FedProx'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'CNN_FedProx_NonIID_NonBalance_jr1.0_mu_0.01'
# model_pt_name = f'FedProx_gr_100_ls_10_jr_1.0_CNN_FedProx_NonIID_NonBalance_jr1.0_mu_0.01.pt'
# num_clients = 10
# global_epoch = 10
# num_classes = 10
# dataset_name = f'Cifar10_NonIID_dir_0.1_Client_{num_clients}'
# save_file_path = 'APH_FedProx_Cifar10_NonIID_dir_0.1_Client_10_PR_1.0_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2.pkl'


# Cifar100 ResNet
# algo = 'FedProx'
# dataset_name = f'Cifar100_NonIID_dir_0.1_Client_10'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'ResNet50_FedProx_NonIID_NonBalance_jr1.0_mu_0.01'
# model_pt_name = f'FedProx_gr_100_ls_10_jr_1.0_ResNet50_FedProx_NonIID_NonBalance_jr1.0_mu_0.01.pt'
# num_classes = 100
# num_clients = 10
# global_epoch = 10
# save_file_path = 'APH_FedProx_Cifar100_NonIID_dir_0.1_Client_10_PR_1.0_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.0.pkl'
#

# Tiny-ImageNet FedProx
# algo = 'FedProx'
# dataset_name = f'Tiny-imagenet'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'ResNet_FedProx_NonIID_NonBalance_jr1.0_mu_0.01'
# model_pt_name = f'FedProx_gr_20_ls_10_jr_1.0_ResNet_FedProx_NonIID_NonBalance_jr1.0_mu_0.01.pt'
# num_classes = 200
# num_clients = 10
# global_epoch = 10
# save_file_path = 'APH_FedProx_Tiny-imagenet_PR_1.0_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.0.pkl'



# algo = 'FedDyn'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'CNN_FedDyn_NonIID_NonBalance_jr_1.0_dir_0.1_alpha_0.001'
# model_pt_name = f'FedDyn_gr_100_ls_10_jr_1.0_CNN_FedDyn_NonIID_NonBalance_jr_1.0_dir_0.1_alpha_0.001.pt'
# num_clients = 10
# global_epoch = 10
# num_classes = 10
# dataset_name = f'Cifar10_NonIID_dir_0.1_Client_{num_clients}'
# save_file_path = 'APH_Dyn_Cifar10_NonIID_dir_0.1_Client_10_PR_1.0_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2.pkl'


# algo = 'FedDyn'
# dataset_name = f'Cifar100_NonIID_dir_0.1_Client_10'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'ResNet50_FedDyn_NonIID_NonBalance_jr_1.0_dir_0.1_alpha_0.01_ev'
# model_pt_name = f'FedDyn_gr_100_ls_10_jr_1.0_ResNet50_FedDyn_NonIID_NonBalance_jr_1.0_dir_0.1_alpha_0.01_ev.pt'
# num_classes = 100
# num_clients = 10
# global_epoch = 10
# save_file_path = 'APH_FedDyn_Cifar100_NonIID_dir_0.1_Client_10_PR_1.0_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.01_seed_2345_lamda_-0.5.pkl'


# algo = 'FedDyn'
# dataset_name = f'Tiny-imagenet'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'ResNet50_FedDyn_NonIID_NonBalance_jr1.0'
# model_pt_name = f'FedDyn_gr_20_ls_10_jr_1.0_ResNet50_FedDyn_NonIID_NonBalance_jr1.0.pt'
# num_classes = 200
# num_clients = 10
# global_epoch = 10
# save_file_path = 'APH_FedDyn_Tiny-imagenet_PR_1.0_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2.pkl'


# algo = 'FedNTD'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'CNN_FedNTD_NonIID_NonBalance_jr1.0'
# model_pt_name = f'FedNTD_gr_100_ls_10_jr_1.0_CNN_FedNTD_NonIID_NonBalance_jr1.0_best.pt'
# num_clients = 10
# global_epoch = 10
# num_classes = 10
# dataset_name = f'Cifar10_NonIID_dir_0.1_Client_{num_clients}'
# save_file_path = 'APH_FedNTD_Cifar10_NonIID_dir_0.1_Client_10_PR_1.0_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2.pkl'


# algo = 'FedNTD'
# dataset_name = f'Cifar100_NonIID_dir_0.1_Client_10'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'ResNet_FedNTD_NonIID_NonBalance_jr1.0'
# model_pt_name = f'FedNTD_gr_100_ls_10_jr_1.0_ResNet_FedNTD_NonIID_NonBalance_jr1.0_best.pt'
# num_classes = 100
# num_clients = 10
# global_epoch = 10
# save_file_path = 'APH_FedNTD_Cifar100_NonIID_dir_0.1_Client_10_PR_1.0_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.0.pkl'


# algo = 'FedNTD'
# dataset_name = f'Tiny-imagenet'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'ResNet_FedNTD_NonIID_NonBalance_jr1.0'
# model_pt_name = f'FedNTD_gr_20_ls_10_jr_1.0_ResNet_FedNTD_NonIID_NonBalance_jr1.0.pt'
# num_classes = 200
# num_clients = 10
# global_epoch = 10
# save_file_path = 'APH_FedNTD_Tiny-imagenet_PR_1.0_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2.pkl'
#




######## Robustness Exp

# FedAvg PR Exp 0.2
# algo = 'FedAvg'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'CNN_FedAvg_NonIID_NonBalance_jr_0.2_dir_0.1'
# model_pt_name = f'FedAvg_gr_100_ls_10_jr_0.2_CNN_FedAvg_NonIID_NonBalance_jr_0.2_dir_0.1_best.pt'
# num_clients = 10
# global_epoch = 10
# num_classes = 10
# dataset_name = f'Cifar10_NonIID_dir_0.1_Client_{num_clients}'
# save_file_path = 'APH_CNN_FedAvg_NonIID_NonBalance_jr_0.2_dir_0.1_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2.pkl'

# FedAvg PR Exp 0.6
# algo = 'FedAvg'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'CNN_FedAvg_NonIID_NonBalance_jr_0.6_dir_0.1'
# model_pt_name = f'FedAvg_gr_100_ls_10_jr_0.6_CNN_FedAvg_NonIID_NonBalance_jr_0.6_dir_0.1_best.pt'
# num_clients = 10
# global_epoch = 10
# num_classes = 10
# dataset_name = f'Cifar10_NonIID_dir_0.1_Client_{num_clients}'
# save_file_path = 'APH_CNN_FedAvg_NonIID_NonBalance_jr_0.6_dir_0.1_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2.pkl'

# FedAvg PR Exp PR 1.0
# algo = 'FedAvg'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'CNN_FedAvg_NonIID_NonBalance_jr1.0'
# model_pt_name = f'FedAvg_gr_100_ls_10_jr_1.0_CNN_FedAvg_NonIID_NonBalance_jr1.0_best.pt'
# num_clients = 10
# global_epoch = 10
# num_classes = 10
# dataset_name = f'Cifar10_NonIID_dir_0.1_Client_{num_clients}'
# save_file_path = 'APH_FedAvg_Cifar10_NonIID_dir_0.1_Client_10_CNN_FedAvg_NonIID_NonBalance_jr1.0_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_-0.5.pkl'
#

# FedAvg NonIID Exp dir 0.05
# algo = 'FedAvg'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'CNN_FedAvg_NonIID_NonBalance_jr1.0_dir_0.05'
# model_pt_name = f'FedAvg_gr_100_ls_10_jr_1.0_CNN_FedAvg_NonIID_NonBalance_jr1.0_dir_0.05_best.pt'
# num_clients = 10
# global_epoch = 10
# num_classes = 10
# dataset_name = f'Cifar10_NonIID_dir_0.05_Client_10'
# save_file_path = 'APH_CNN_FedAvg_NonIID_NonBalance_jr1.0_dir_0.05_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2.pkl'
#



# FedAvg NonIID Exp dir 0.5
# algo = 'FedAvg'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'CNN_FedAvg_NonIID_NonBalance_jr1.0_dir_0.5'
# model_pt_name = f'FedAvg_gr_100_ls_10_jr_1.0_CNN_FedAvg_NonIID_NonBalance_jr1.0_dir_0.5_best.pt'
# num_clients = 10
# global_epoch = 10
# num_classes = 10
# dataset_name = f'Cifar10_NonIID_dir_0.5_Client_10'
# save_file_path = 'APH_CNN_FedAvg_NonIID_NonBalance_jr1.0_dir_0.5_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2.pkl'
# save_file_path = 'APH_CNN_FedAvg_NonIID_NonBalance_jr1.0_dir_0.5_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.5.pkl'
# save_file_path = 'APH_CNN_FedAvg_NonIID_NonBalance_jr1.0_dir_0.5_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.0.pkl'
# save_file_path = 'APH_CNN_FedAvg_NonIID_NonBalance_jr1.0_dir_0.5_Epoch_10_HeadsNum_10_ulr_10.0_llr_0.001_seed_2345_lamda_0.0.pkl'


# # FedAvg LS Exp LS 20
# algo = 'FedAvg'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'CNN_FedAvg_NonIID_NonBalance_jr1.0_ls_20'
# model_pt_name = f'FedAvg_gr_100_ls_20_jr_1.0_CNN_FedAvg_NonIID_NonBalance_jr1.0_ls_20_best.pt'
# num_clients = 10
# global_epoch = 10
# num_classes = 10
# dataset_name = f'Cifar10_NonIID_dir_0.1_Client_10'
# save_file_path = 'APH_CNN_FedAvg_NonIID_NonBalance_jr1.0_ls_20_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2.pkl'


# # # FedAvg LS Exp LS 40
# algo = 'FedAvg'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'CNN_FedAvg_NonIID_NonBalance_jr1.0_ls_40'
# model_pt_name = f'FedAvg_gr_100_ls_40_jr_1.0_CNN_FedAvg_NonIID_NonBalance_jr1.0_ls_40_best.pt'
# num_clients = 10
# global_epoch = 10
# num_classes = 10
# dataset_name = f'Cifar10_NonIID_dir_0.1_Client_10'
# save_file_path = 'APH_CNN_FedAvg_NonIID_NonBalance_jr1.0_ls_40_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2.pkl'

#
#
# # # # FedAvg Client Exp 50
# algo = 'FedAvg'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'CNN_FedAvg_NonIID_NonBalance_jr1.0_dir_0.1_nc_50'
# model_pt_name = f'FedAvg_gr_100_ls_10_jr_1.0_CNN_FedAvg_NonIID_NonBalance_jr1.0_dir_0.1_nc_50_best.pt'
# num_clients = 50
# global_epoch = 10
# num_classes = 10
# dataset_name = f'Cifar10_NonIID_dir_0.1_Client_50'
# save_file_path = 'APH_CNN_FedAvg_NonIID_NonBalance_jr1.0_dir_0.1_nc_50_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2.pkl'


# # FedAvg Client Exp 100
# algo = 'FedAvg'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'CNN_FedAvg_NonIID_NonBalance_jr1.0_dir_0.1_nc_100'
# model_pt_name = f'FedAvg_gr_100_ls_10_jr_1.0_CNN_FedAvg_NonIID_NonBalance_jr1.0_dir_0.1_nc_100_best.pt'
# num_clients = 100
# global_epoch = 10
# num_classes = 10
# dataset_name = f'Cifar10_NonIID_dir_0.1_Client_100'
# save_file_path = 'APH_CNN_FedAvg_NonIID_NonBalance_jr1.0_dir_0.1_nc_100_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2.pkl'
#

# algo = 'FedALA'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'CNN_FedALA_NonIID_NonBalance_jr_1.0_dir_0.1'
# model_pt_name = f'FedALA_gr_99_ls_10_jr_1.0_CNN_FedALA_NonIID_NonBalance_jr_1.0_dir_0.1'
# num_clients = 10
# global_epoch = 10
# num_classes = 10
# dataset_name = f'Cifar10_NonIID_dir_0.1_Client_{num_clients}'
# save_file_path = 'APH_CNN_FedALA_NonIID_NonBalance_jr_1.0_dir_0.1_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2.pkl'

#
#
# algo = 'FedALA'
# dataset_name = f'Cifar100_NonIID_dir_0.1_Client_10'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'ResNet50_FedALA_NonIID_NonBalance_jr_1.0_dir_0.1'
# model_pt_name = f'FedALA_gr_99_ls_10_jr_1.0_ResNet50_FedALA_NonIID_NonBalance_jr_1.0_dir_0.1'
# num_classes = 100
# num_clients = 10
# global_epoch = 10
# save_file_path = 'APH_ResNet50_FedALA_NonIID_NonBalance_jr_1.0_dir_0.1_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.0.pkl'
# save_file_path = 'APH_FedALA_Cifar100_NonIID_dir_0.1_Client_10_ResNet50_FedALA_NonIID_NonBalance_jr_1.0_dir_0.1_Epoch_10_HeadsNum_10_ulr_10.0_llr_0.001_seed_2345_lamda_-0.2.pkl'
# save_file_path = 'APH_FedALA_Cifar100_NonIID_dir_0.1_Client_10_ResNet50_FedALA_NonIID_NonBalance_jr_1.0_dir_0.1_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.0.pkl'
# save_file_path = 'APH_FedALA_Cifar100_NonIID_dir_0.1_Client_10_ResNet50_FedALA_NonIID_NonBalance_jr_1.0_dir_0.1_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_-0.2.pkl'
# save_file_path = 'APH_FedALA_Cifar100_NonIID_dir_0.1_Client_10_ResNet50_FedALA_NonIID_NonBalance_jr_1.0_dir_0.1_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2.pkl'

# algo = 'FedALA'
# dataset_name = f'Tiny-imagenet'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'ResNet50_FedALA_NonIID_NonBalance_jr_1.0_dir_0.1'
# model_pt_name = f'FedALA_gr_19_ls_10_jr_1.0_ResNet50_FedALA_NonIID_NonBalance_jr_1.0_dir_0.1'
# num_classes = 200
# num_clients = 10
# global_epoch = 10
# save_file_path = 'APH_ResNet50_FedALA_NonIID_NonBalance_jr_1.0_dir_0.1_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.0.pkl'
# save_file_path = 'APH_ResNet50_FedALA_NonIID_NonBalance_jr_1.0_dir_0.1_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2.pkl'
# save_file_path = 'APH_ResNet50_FedALA_NonIID_NonBalance_jr_1.0_dir_0.1_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_-0.2.pkl'
# save_file_path = 'APH_FedALA_Tiny-imagenet_ResNet50_FedALA_NonIID_NonBalance_jr_1.0_dir_0.1_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_-0.5.pkl'
# save_file_path = 'APH_FedALA_Tiny-imagenet_ResNet50_FedALA_NonIID_NonBalance_jr_1.0_dir_0.1_Epoch_10_HeadsNum_10_ulr_10.0_llr_0.001_seed_2345_lamda_-0.5.pkl'
# save_file_path = 'APH_FedALA_Tiny-imagenet_ResNet50_FedALA_NonIID_NonBalance_jr_1.0_dir_0.1_Epoch_10_HeadsNum_10_ulr_10.0_llr_0.001_seed_2345_lamda_-0.2.pkl'
# save_file_path = 'APH_FedALA_Tiny-imagenet_ResNet50_FedALA_NonIID_NonBalance_jr_1.0_dir_0.1_Epoch_10_HeadsNum_10_ulr_1000.0_llr_0.001_seed_2345_lamda_-0.2.pkl'

# algo = 'FedFOMO'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'CNN_FedFomo_NonIID_NonBalance_jr_1.0_dir_0.1'
# model_pt_name = f'FedFomo_gr_99_ls_10_jr_1.0_CNN_FedFomo_NonIID_NonBalance_jr_1.0_dir_0.1'
# num_clients = 10
# global_epoch = 10
# num_classes = 10
# dataset_name = f'Cifar10_NonIID_dir_0.1_Client_{num_clients}'
# save_file_path = 'APH_FedFOMO_Cifar10_NonIID_dir_0.1_Client_10_CNN_FedFomo_NonIID_NonBalance_jr_1.0_dir_0.1_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.0.pkl'
# save_file_path = 'APH_FedFOMO_Cifar10_NonIID_dir_0.1_Client_10_CNN_FedFomo_NonIID_NonBalance_jr_1.0_dir_0.1_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2.pkl'
# save_file_path = 'APH_FedFOMO_Cifar10_NonIID_dir_0.1_Client_10_CNN_FedFomo_NonIID_NonBalance_jr_1.0_dir_0.1_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_-0.2.pkl'

#
# algo = 'FedFOMO'
# dataset_name = f'Cifar100_NonIID_dir_0.1_Client_10'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'ResNet50_FedFOMO_NonIID_NonBalance_jr_1.0_dir_0.1_M_5'
# model_pt_name = f'FedFomo_gr_99_ls_10_jr_1.0_ResNet50_FedFOMO_NonIID_NonBalance_jr_1.0_dir_0.1_M_5'
# num_classes = 100
# num_clients = 10
# global_epoch = 10
# # save_file_path = 'APH_FedFOMO_Cifar100_NonIID_dir_0.1_Client_10_ResNet50_FedFOMO_NonIID_NonBalance_jr_1.0_dir_0.1_M_5_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.0.pkl'
# # save_file_path = 'APH_FedFOMO_Cifar100_NonIID_dir_0.1_Client_10_ResNet50_FedFOMO_NonIID_NonBalance_jr_1.0_dir_0.1_M_5_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2.pkl'
# # save_file_path = 'APH_FedFOMO_Cifar100_NonIID_dir_0.1_Client_10_ResNet50_FedFOMO_NonIID_NonBalance_jr_1.0_dir_0.1_M_5_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_-0.2.pkl'
# # save_file_path = 'APH_FedFOMO_Cifar100_NonIID_dir_0.1_Client_10_ResNet50_FedFOMO_NonIID_NonBalance_jr_1.0_dir_0.1_M_5_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_-0.5.pkl'
# save_file_path = 'APH_FedFOMO_Cifar100_NonIID_dir_0.1_Client_10_ResNet50_FedFOMO_NonIID_NonBalance_jr_1.0_dir_0.1_M_5_Epoch_10_HeadsNum_10_ulr_1000.0_llr_0.001_seed_2345_lamda_-0.5.pkl'
# save_file_path = 'APH_FedFOMO_Cifar100_NonIID_dir_0.1_Client_10_ResNet50_FedFOMO_NonIID_NonBalance_jr_1.0_dir_0.1_M_5_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_-1.0.pkl'



# algo = 'FedFomo'
# dataset_name = f'Tiny-imagenet'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'ResNet50_FedFomo_NonIID_NonBalance_jr1.0_M_5'
# model_pt_name = f'FedFomo_gr_19_ls_10_jr_1.0_ResNet50_FedFomo_NonIID_NonBalance_jr1.0_M_5'
# num_classes = 200
# num_clients = 10
# global_epoch = 10
# save_file_path = 'APH_FedFOMO_Tiny-imagenet_ResNet50_FedFomo_NonIID_NonBalance_jr1.0_M_5_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.0.pkl'
# save_file_path = 'APH_FedFOMO_Tiny-imagenet_ResNet50_FedFomo_NonIID_NonBalance_jr1.0_M_5_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2.pkl'
# save_file_path = 'APH_FedFOMO_Tiny-imagenet_ResNet50_FedFomo_NonIID_NonBalance_jr1.0_M_5_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_-0.2.pkl'
# save_file_path = 'APH_FedFOMO_Tiny-imagenet_ResNet50_FedFomo_NonIID_NonBalance_jr1.0_M_5_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_-0.5.pkl'
# save_file_path = 'APH_FedFOMO_Tiny-imagenet_ResNet50_FedFomo_NonIID_NonBalance_jr1.0_M_5_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_-1.0.pkl'

# save_file_path = f'AHP_{algo}_{dataset_name}_heads.pkl'
# save_file_path = f'AHP_avg_{algo}_{dataset_name}_heads.pkl'
# keyword = 'order_plus_1_1_0.1_0.001'
# save_file_path = f'AHP_avg_{algo}_{dataset_name}_{global_epoch}_{keyword}_heads.pkl'


# FedAvg Ablation Study
#
# algo = 'FedAvg'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'CNN_FedAvg_NonIID_NonBalance_jr1.0'
# model_pt_name = f'FedAvg_gr_100_ls_10_jr_1.0_CNN_FedAvg_NonIID_NonBalance_jr1.0_best.pt'
# num_clients = 10
# global_epoch = 10
# num_classes = 10
# dataset_name = f'Cifar10_NonIID_dir_0.1_Client_{num_clients}'
# save_file_path = 'APH_FedAvg_Cifar10_NonIID_dir_0.1_Client_10_CNN_FedAvg_NonIID_NonBalance_jr1.0_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_-0.5.pkl'
# # save_file_path = 'APH_FedAvg_Cifar10_NonIID_dir_0.1_Client_10_CNN_FedAvg_NonIID_NonBalance_jr1.0_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2_ablation_key_1.pkl'
# # save_file_path = 'APH_FedAvg_Cifar10_NonIID_dir_0.1_Client_10_CNN_FedAvg_NonIID_NonBalance_jr1.0_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2_ablation_key_2.pkl'
# # save_file_path = 'APH_FedAvg_Cifar10_NonIID_dir_0.1_Client_10_CNN_FedAvg_NonIID_NonBalance_jr1.0_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2_ablation_key_2.pkl'
# # save_file_path = 'APH_FedAvg_Cifar10_NonIID_dir_0.1_Client_10_CNN_FedAvg_NonIID_NonBalance_jr1.0_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2_ablation_key_2.pkl'
# save_file_path = 'APH_FedAvg_Cifar10_NonIID_dir_0.1_Client_10_CNN_FedAvg_NonIID_NonBalance_jr1.0_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.2_ablation_key_2.pkl'
#


# Cifar100 ResNet
# algo = 'FedAvg'
# dataset_name = f'Cifar100_NonIID_dir_0.1_Client_10'
# root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
# dir_path = f'ResNet50_FedAvg_NonIID_NonBalance_jr1.0'
# model_pt_name = f'FedAvg_gr_100_ls_10_jr_1.0_ResNet50_FedAvg_NonIID_NonBalance_jr1.0_best.pt'
# num_classes = 100
# num_clients = 10
# global_epoch = 10
# save_file_path = 'APH_FedAvg_Cifar100_NonIID_dir_0.1_Client_10_PR_1.0_Epoch_10_HeadsNum_10_ulr_1000.0_llr_0.001_seed_2345_lamda_0.2.pkl'
# save_file_path = 'APH_FedAvg_Cifar100_NonIID_dir_0.1_Client_10_ResNet50_FedAvg_NonIID_NonBalance_jr1.0_Epoch_10_HeadsNum_10_ulr_1000.0_llr_0.001_seed_2345_lamda_0.2_ablation_key_1.pkl'
# save_file_path = 'APH_FedAvg_Cifar100_NonIID_dir_0.1_Client_10_ResNet50_FedAvg_NonIID_NonBalance_jr1.0_Epoch_10_HeadsNum_10_ulr_1000.0_llr_0.001_seed_2345_lamda_0.2_ablation_key_2.pkl'


# Tiny-ImageNet ResNet
algo = 'FedAvg'
dataset_name = f'Tiny-imagenet'
root_path = '/home/chenjinqian/code/MINE_FL/PFL/system/models'
dir_path = f'ResNet_FedAvg_NonIID_NonBalance_jr1.0'
model_pt_name = f'FedAvg_gr_20_ls_10_jr_1.0_ResNet_FedAvg_NonIID_NonBalance_jr1.0_best.pt'
num_classes = 200
num_clients = 10
global_epoch = 10
# # save_file_path = 'APH_FedAvg_Tiny-imagenet_PR_1.0_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.1.pkl'
save_file_path = 'APH_FedAvg_Tiny-imagenet_ResNet_FedAvg_NonIID_NonBalance_jr1.0_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_-0.5.pkl'
# # save_file_path = 'APH_FedAvg_Tiny-imagenet_ResNet_FedAvg_NonIID_NonBalance_jr1.0_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.1_ablation_key_1.pkl'
# # save_file_path = 'APH_FedAvg_Tiny-imagenet_ResNet_FedAvg_NonIID_NonBalance_jr1.0_Epoch_10_HeadsNum_10_ulr_1.0_llr_0.001_seed_2345_lamda_0.1_ablation_key_2.pkl'
# #



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


    PFL_flag = check_pfl(model_pt_name)
    model_dir_path = os.path.join(root_path, dataset_name, dir_path)
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