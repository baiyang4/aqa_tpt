# coding: utf-8
import argparse
import time
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_opt():
    # parser
    parser = argparse.ArgumentParser()
    # General settings
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--epoch_num', type=int, default=2500)
    # parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--lr_factor', type=float, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save_per_epoch', type=int, default=8)
    parser.add_argument('--bs_train', type=int, default=3)
    parser.add_argument('--bs_test', type=int, default=3)
    parser.add_argument('--use_pretrain', type=str2bool, default=False)
    parser.add_argument("--use_checkpoint", type=str2bool, default=False)
    parser.add_argument('--multi_gpu_test', type=str2bool, default=True)
    parser.add_argument('--use_dd', type=str2bool, default=True)
    parser.add_argument('--cross_dd', type=str2bool, default=False)
    parser.add_argument('--encode_video', type=str2bool, default=True)
    parser.add_argument('--weighed_cluster', type=str2bool, default=False)
    parser.add_argument('--num_voting', type=int, default=10)
    parser.add_argument('--multi_e', type=str2bool, default=False)
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--d_ffn', type=int, default=512)
    parser.add_argument('--pe', type=str2bool, default=True)
    parser.add_argument('--se', type=str2bool, default=False)
    parser.add_argument('--attention', type=str2bool, default=True)
    parser.add_argument('--n_ep', type=int, default=10)
    parser.add_argument('--tree_target', type=str2bool, default=True)

    # Network settings
    parser.add_argument('--model', type=str, default='CoRe')  # RMN
    parser.add_argument('--exp_name', type=str, default='mean_rela_no_tree_cluster_gather_4')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--frame_hidden_size', type=int, default=1000)
    parser.add_argument('--num_cluster', type=int, default=5)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--margin_factor', type=float, default=20)
    parser.add_argument("--hinge_loss", type=str2bool, default=True)
    parser.add_argument("--multi_hinge", type=str2bool, default=True)
    parser.add_argument("--cluster_norm_dim", type=int, default=1)
    parser.add_argument("--mse_loss", type=str2bool, default=True)
    parser.add_argument("--prob_loss", type=str2bool, default=False)
    parser.add_argument("--class_loss", type=str2bool, default=False)
    parser.add_argument("--query_embed_cross", type=str2bool, default=False)
    # Feature extract settings
    parser.add_argument('--max_frames', type=int, default=26)
    # dataset
    parser.add_argument('--dataset', type=str, default='MLT_AQA', help='choose from AQA_7|MLT_AQA')
    parser.add_argument('--class_idx', type=int, default=1, help='choose from AQA_7|MLT_AQA')
    parser.add_argument('--data_root', type=str, default='./data/data_preprocessed')
    parser.add_argument('--pretrained_i3d_weight', type=str, default='./data/model_rgb.pth')
    parser.add_argument('--frame_length', type=int, default=103)


    args = parser.parse_args()
    return args


