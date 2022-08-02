from utils.opt import parse_opt
import torch
from dataloader_7 import get_AQA7_dataloader
from dataloader_MLT import get_MLT_dataloader
import numpy as np
import torch
import random
from models.model import I3D_backbone
from models.model import SwinBackbone
import os


args = parse_opt()
args.bs_train = 1
args.bs_test = 1
args.use_pretrain = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = os.path.join(args.data_root, args.dataset)
folder_name = 'features_8'

if args.dataset == 'AQA_7':
    train_loader, test_loader = get_AQA7_dataloader(args)
else:
    train_loader, test_loader, _ = get_MLT_dataloader(args)
# check_point='/root/paddlejob/workspace/env_run/aqa/BY_AQA/results/InterModelV6_I3D_backbone_5_10_MLT_AQA_1_True_True_time_norm_multi_1/0.9551.pth'
# state_dict = torch.load(check_point, map_location=device)
# # base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
# base_ckpt = state_dict['base_model']

backbone = I3D_backbone().to(device)
# backbone.load_state_dict(base_ckpt)
backbone.load_pretrain(args.pretrained_i3d_weight)
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

