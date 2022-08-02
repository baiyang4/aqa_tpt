from utils.opt import parse_opt
import torch
from dataloader_7 import get_AQA7_dataloader
from dataloader_MLT_52 import get_MLT_dataloader
import numpy as np
import torch
import random
from models_cluster_2layer_pairencode1_clsreg_decoder_1selfatt_self8head_ffn.model import I3D_backbone
import os
import pdb

args = parse_opt()
args.bs_train = 1
args.bs_test = 1
args.use_pretrain = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = os.path.join(args.data_root, args.dataset)
folder_name = 'features_52_ffn512_dmodel256'

if args.dataset == 'AQA_7':
    train_loader, test_loader = get_AQA7_dataloader(args)
else:
    train_loader, test_loader, _ = get_MLT_dataloader(args)
check_point='/root/paddlejob/workspace/env_run/aqa/BY_AQA/results/PairModel_I3D_backbone_5_10_MLT_AQA_1_True_True_52_decoder_1selfatt_self8head_256_512/0.9503.pth'
state_dict = torch.load(check_point, map_location=device)
# # base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
base_ckpt = state_dict['backbone']

backbone = I3D_backbone().to(device)
backbone.load_state_dict(base_ckpt)
#backbone.load_pretrain(args.pretrained_i3d_weight)
backbone.eval()

# backbone = SwinBackbone().to(device)
# backbone.load_pretrain()
# state_dict = torch.load('/root/paddlejob/workspace/env_run/aqa/BY_AQA/results/InterModelV6_SwinBackbone_10_52_MLT_AQA_1_True_True_st/0.9550.pth', map_location=device)
# base_ckpt = state_dict['base_model']
# backbone.load_state_dict(base_ckpt)


with torch.no_grad():
    for data in train_loader:
        video = data['video'].to(device)
        video_id = data['id_str']
        I3d_feature = backbone(video).squeeze()
        pdb.set_trace()
        torch.save(I3d_feature.cpu(), os.path.join(data_root, folder_name, f'{video_id[0]}.pt'))
        # swin_feature = backbone(video)
        # torch.save(swin_feature.cpu(), os.path.join(data_root, folder_name, f'{video_id[0]}.pt'))

    for data in test_loader:
        video = data['video'].to(device)
        video_id = data['id_str']
        I3d_feature = backbone(video).squeeze()
        torch.save(I3d_feature.cpu(), os.path.join(data_root, folder_name, f'{video_id[0]}.pt'))
        # swin_feature = backbone(video)
        # torch.save(swin_feature.cpu(), os.path.join(data_root, folder_name, f'{video_id[0]}.pt'))


