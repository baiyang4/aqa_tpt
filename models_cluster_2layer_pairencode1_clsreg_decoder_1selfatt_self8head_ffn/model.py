import os
import torch.nn as nn
import torch
from .i3d import I3D
import logging
from .layer import *
from utils.util import distributed_concat
import torch.nn.functional as F
from collections import OrderedDict

class I3D_backbone(nn.Module):
    def __init__(self):
        super(I3D_backbone, self).__init__()
        self.backbone = I3D(num_classes=400, modality='rgb', dropout_prob=0.5)
        self.se = False
        if self.se:
            self.avg_pool_se = torch.nn.AvgPool2d(7, 7)
            self.spacial_se = torch.nn.Sequential(
                torch.nn.Conv2d(1024, 1, 3, 1, 1),
                torch.nn.Sigmoid()
            )
        else:
            self.avg_pool = torch.nn.AvgPool3d((1, 7, 7), (1, 1, 1))

    def load_pretrain(self, I3D_ckpt_path):
        self.backbone.load_state_dict(torch.load(I3D_ckpt_path))

    def get_feature_dim(self):
        return self.backbone.get_logits_dim()

    def forward(self, video):
        # start_idx = [0, 10, 20, 30, 40, 50, 60, 70, 80, 86]
        bs, len_video, _, _, _, _ = video.size()
        video_feature = self.backbone(video.view(-1, 3, 8, 224, 224))
        if self.se:
            video_feature = video_feature.mean(2)
            video_feature = self.avg_pool_se(self.spacial_se(video_feature) * video_feature)
            # feature = self.avg_pool(out)
        else:
            video_feature = self.avg_pool(video_feature)
        # video_feature = video_feature.reshape(20, len(video), -1).transpose(0, 1)  # 2N * 10 * 1024
        video_feature = video_feature.view(bs, len_video, -1)
        return video_feature



class PairModel(nn.Module):
    def __init__(self, args):
        super(PairModel, self).__init__()
        self.action_decoder = ActionDecoder(args.num_cluster, args.d_model, args.d_ffn)
        self.pair_encoder = PairEncoder(args.d_model)

        self.cls = nn.Sequential(
            nn.Linear(args.d_model, 128),
            nn.ReLU(True),
            nn.Linear(128, 16)
        )

        self.reg = nn.Sequential(
            nn.Linear(args.d_model, 128),
            nn.ReLU(True),
            nn.Linear(128, 16),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, train=True):

        x1, query_att1 = self.action_decoder(x1)
        x2, query_att2 = self.action_decoder(x2)
        query_att = torch.cat((query_att1, query_att2), 0)

        if train:
            pair_reprs1 = self.pair_encoder(x1, x2).mean(1)
            pair_reprs2 = self.pair_encoder(x2, x1).mean(1)

            feat_all = torch.cat((pair_reprs1, pair_reprs2), 0)
        else:
            pair_reprs2 = self.pair_encoder(x2, x1).mean(1)
            feat_all = pair_reprs2
            query_att = query_att1

        cls = self.cls(feat_all)
        prob = F.log_softmax(cls, dim=-1)
        reg = self.reg(feat_all)

        return [prob], reg, query_att

