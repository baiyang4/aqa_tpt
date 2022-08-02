from utils.opt import parse_opt
import torch
from dataloader_7 import get_AQA7_dataloader
from dataloader_MLT import get_MLT_dataloader
import numpy as np
import torch
import random
from models.model import I3D_backbone
import os


args = parse_opt()
args.bs_train = 1
args.bs_test = 1
args.use_pretrain = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = os.path.join(args.data_root, args.dataset)
folder_name = 'features'

if args.dataset == 'AQA_7':
    train_loader, test_loader = get_AQA7_dataloader(args)
else:
    train_loader, test_loader = get_MLT_dataloader(args)
# check_point='/baiy/AQA/Core/MTL-AQA/MTL_CoRe.pth'
# state_dict = torch.load(check_point, map_location=device)
# base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}

backbone = I3D_backbone().to(device)
# backbone.load_state_dict(base_ckpt)
backbone.load_pretrain(args.pretrained_i3d_weight)
backbone.eval()

class_name = ['diving', 'gym_vault', 'ski_big_air', 'snowboard_big_air', 'sync_diving_3m', 'sync_diving_10m']

with torch.no_grad():
    # for data in train_loader:
    #     video = data['video'].to(device)
    #     video_id = data['id']
    #     I3d_feature = backbone(video).squeeze()
    #     action_id, data_id = video_id[0].split('_')
    #     folder = os.path.join(data_root, folder_name, f'{class_name[int(action_id)-1]}-out')
    #     os.makedirs(folder, exist_ok=True)
    #     torch.save(I3d_feature.cpu(), os.path.join(folder, f'{int(data_id):03d}.pt'))

    for data in test_loader:
        video = data['video'].to(device)
        video_id = data['id']
        I3d_feature = backbone(video).squeeze()
        action_id, data_id = video_id[0].split('_')
        folder = os.path.join(data_root, folder_name, f'{class_name[int(action_id) - 1]}-out')
        os.makedirs(folder, exist_ok=True)
        torch.save(I3d_feature.cpu(), os.path.join(folder, f'{int(data_id):03d}.pt'))

