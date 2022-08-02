from einops import repeat
import pdb
import os.path
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from utils.util import ResultsHandler, fix_bn
from torch.utils.tensorboard import SummaryWriter
from scipy import stats
import time
from torch import distributed
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from utils.opt import parse_opt
import torch.distributed as dist
from dataloader_pair_7 import get_AQA7Pair_dataloader
from dataloader_pair_MLT import get_MLTPair_dataloader
import random

from models_cluster_2layer_pairencode1_clsreg_decoder_1selfatt_self8head_ffn.model import I3D_backbone, PairModel



class RunPair:
    def __init__(self, args, device, train_loader=None, test_loader=None, grouper=None, multi_gpu=False, evaluate=False, plot=False):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.plot = plot
        self.grouper = grouper
        self.args = args
        self.last_epoch = -1

        self.multi_gpu_test = args.multi_gpu_test
        self.device = device
        self.local_rank = args.local_rank
        self.voter_number = args.num_voting
        self.cross_dd = args.cross_dd
        self.multi_gpu = multi_gpu
        self.use_checkpoint = args.use_checkpoint
        self.checkpoint = None
        self.use_pretrain = args.use_pretrain
        self.lr = 0.0001
        self.lr_factor = args.lr_factor
        self.epoch_num = args.epoch_num
        self.mse_loss = args.mse_loss
        self.prob_loss = args.prob_loss
        self.class_loss = args.class_loss
        self.dataset = args.dataset
        self.use_dd = args.use_dd
        self.num_cluster = args.num_cluster
        self.cluster_loss = False
        self.model = PairModel(args).to(device)
        self.backbone = None

        if not self.use_pretrain:
            self.backbone = I3D_backbone().to(device)
            self.backbone.load_pretrain(args.pretrained_i3d_weight)
        self.base_name = f'{type(self.model).__name__}_{type(self.backbone).__name__}_{args.num_cluster}_{args.margin_factor}_{args.dataset}_{args.cluster_norm_dim}_{args.hinge_loss}_{args.multi_hinge}_{args.exp_name}'
        print(f'num_cluster: {args.num_cluster}, '
              f'margin_factor:{args.margin_factor} '
              f'encode video:{args.encode_video} '
              f'hinge loss:{args.hinge_loss} '
              f'weighted_cluster :{args.weighed_cluster} '
              f'multi hinge:{args.multi_hinge} ' 
              f'norm dim:{args.cluster_norm_dim}')
        # optimizer
        if self.use_pretrain:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.0005)
        else:
            self.optimizer = torch.optim.Adam([
                {'params': self.backbone.parameters(), 'lr': self.lr},
                {'params': self.model.action_decoder.parameters(), 'lr': self.lr},
                {'params': self.model.pair_encoder.parameters(), 'lr': self.lr},
                # {'params': self.model.weighed_norm.parameters()},
                {'params': self.model.cls.parameters(), 'lr': 0.001},
                {'params': self.model.reg.parameters(), 'lr': 0.001}
            ], lr=self.lr)

        checkpoint = None
        if self.use_checkpoint or evaluate:
            checkpoint_path = os.path.join('./results', self.base_name, 'checkpoint.pth')
            if evaluate:
                try:
                    value_list = [i for i in os.listdir(os.path.join('./results', self.base_name)) if
                                  i.split('.')[0].isnumeric()]
                    value_list.sort()
                except:
                    value_list = []
                if len(value_list) > 0:
                    checkpoint_path = os.path.join('./results', self.base_name, value_list[-1])
                else:
                    checkpoint_path = 'none'
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint is not None:
            self.last_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model'])
            if not self.use_pretrain:
                self.backbone.load_state_dict(checkpoint['backbone'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.classname = ['diving', 'gym_vault', 'ski_big_air', 'snowboard_big_air', 'sync_diving_3m', 'sync_diving_10m']
        if self.multi_gpu:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                                   find_unused_parameters=True)
            if not self.use_pretrain:
                self.backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
                self.backbone = torch.nn.parallel.DistributedDataParallel(self.backbone, device_ids=[args.local_rank],
                                                                          find_unused_parameters=True)

        # setup results handler
        self.result_handler = ResultsHandler(self.model, self.backbone, args.dataset,
                                             self.base_name, checkpoint,
                                             local_rank=args.local_rank, multi_gpu=multi_gpu, class_idx=args.class_idx)
        self.result_handler.optimizer = self.optimizer
        if self.local_rank <= 0:
            self.writer = SummaryWriter(comment=self.base_name)

    def train(self):
        mse = nn.MSELoss().to(self.device)
        nll = nn.NLLLoss().to(self.device)
        hinge = nn.MarginRankingLoss(margin=self.args.margin_factor)
        # hinge = nn.KLDivLoss(reduction='batchmean')

        # lr_steps = [4, 14]
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=0.5)
        score_key = 'score' if self.dataset == 'AQA_7' else 'raw_score'

        data_len = len(self.train_loader)
        for epoch in range(self.last_epoch + 1, self.epoch_num):
            if self.local_rank<1:
                print(f'Start Epoch {epoch}, time: {datetime.now()}', flush=True)
            self.model.train()
            score_all = []
            score_pred_all = []
            if not self.use_pretrain:
                self.backbone.train()
                self.backbone.apply(fix_bn)
            loss_cls_all = 0
            loss_reg_all = 0
            loss_att_all = 0
            loss_var_all = 0
            if self.cross_dd:
                loss_dd_all = 0
            """ Train """
            self.result_handler.start_epoch()

            for batch_idx, (data, target) in enumerate(self.train_loader):
                # if self.dataset == 'AQA_7':
                #     assert (data['class'].float() == target['class'].float()).all()
                # else:
                #     assert (data['difficulty'].float() == target['difficulty'].float()).all()
                # loss = 0.0
                video_1 = data['video'].to(self.device)
                score_1 = data[score_key].unsqueeze(-1).to(self.device)

                video_2 = target['video'].to(self.device)
                score_2 = target[score_key].unsqueeze(-1).to(self.device)
                batch_size = video_1.size(0)
                score_all.append(data['score'])
                if not self.use_pretrain:
                    video_1, video_2 = torch.chunk(self.backbone(torch.cat([video_1, video_2], 0)), 2)

                out_prob, delta, logits_all = self.model(video_1, video_2)
                glabel_1, rlabel_1 = self.grouper.produce_label(score_2 - score_1)
                glabel_2, rlabel_2 = self.grouper.produce_label(score_1 - score_2)
                leaf_probs = out_prob[-1].reshape(batch_size * 2, -1)
                leaf_probs_1 = leaf_probs[:leaf_probs.shape[0] // 2]
                leaf_probs_2 = leaf_probs[leaf_probs.shape[0] // 2:]
                delta_1 = delta[:delta.shape[0] // 2]
                delta_2 = delta[delta.shape[0] // 2:]

                loss_cls = nll(leaf_probs_1, glabel_1.argmax(0))
                loss_cls += nll(leaf_probs_2, glabel_2.argmax(0))

                loss_reg = 0.
                for i in range(self.grouper.number_leaf()):
                    mask = rlabel_1[i] >= 0
                    if mask.sum() != 0:
                        loss_reg += mse(delta_1[:, i][mask].reshape(-1, 1).float(),
                                    rlabel_1[i][mask].reshape(-1, 1).float())
                    mask = rlabel_2[i] >= 0
                    if mask.sum() != 0:
                        loss_reg += mse(delta_2[:, i][mask].reshape(-1, 1).float(),
                                    rlabel_2[i][mask].reshape(-1, 1).float())

                if self.num_cluster > 0 and self.args.hinge_loss:
                    len_att = logits_all.shape[1] // self.args.num_layers
                    hinge_loss = 0.
                    var_loss = 0.
                    for i in range(self.args.num_layers):
                        hinge_loss_curr, var_loss_curr = self.get_att_loss(logits_all[:, i * len_att: (i+1) * len_att:, :], hinge)
                        hinge_loss += hinge_loss_curr
                        var_loss += var_loss_curr
                    # hinge_loss = self.get_att_loss(logits_all[:,:len_att, :], hinge)
                    loss_all = loss_cls + loss_reg + hinge_loss + 0.05 * var_loss

                else:
                    loss_all = loss_cls + loss_reg

                loss_all.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                relative_scores = self.grouper.inference(leaf_probs_2.detach().cpu().numpy(), delta_2.detach().cpu().numpy())

                score_pred = (relative_scores.to(self.device) + score_2)
                if self.dataset == 'MLT_AQA':
                    score_pred = score_pred * data['difficulty'].float().unsqueeze(-1).to(self.device)
                score_pred_all.append(score_pred)

                if self.multi_gpu:
                    loss_cls_record = self.reduce_tensor(loss_cls.data).item()
                    loss_reg_record = self.reduce_tensor(loss_reg.data).item()
                    if self.num_cluster > 0 and self.args.hinge_loss:
                        loss_att_record = self.reduce_tensor(hinge_loss.data).item()
                        loss_var_record = self.reduce_tensor(var_loss.data).item()

                else:
                    loss_cls_record = loss_cls.item()
                    loss_reg_record = loss_reg.item()
                    if self.num_cluster > 0 and self.args.hinge_loss:
                        loss_att_record = hinge_loss.item()
                        loss_var_record = var_loss.item()

                if self.local_rank < 1:
                    loss_cls_all += loss_cls_record / batch_size
                    loss_reg_all += loss_reg_record / batch_size
                    self.writer.add_scalar('Loss/class_loss', loss_cls_record / batch_size, batch_idx + epoch * data_len)
                    self.writer.add_scalar('Loss/reg_loss', loss_reg_record / batch_size, batch_idx + epoch * data_len)

                    if self.num_cluster > 0 and self.args.hinge_loss:
                        loss_att_all += loss_att_record / batch_size
                        loss_var_all += loss_var_record / batch_size
                        self.writer.add_scalar('Loss/att_loss', loss_att_record / batch_size, batch_idx + epoch * data_len)

                    # self.writer.add_scalar('Loss/class_loss', loss_class_record, i + epoch * num_iter)
            # scheduler.step()]
            if self.local_rank < 1:
                score_all = torch.cat(score_all)
                score_pred_all = torch.cat(score_pred_all)

                score_all = score_all.detach().cpu().numpy().reshape(-1,1)
                score_pred_all = score_pred_all.detach().cpu().numpy()
                rho_curr, p = stats.spearmanr(score_pred_all, score_all)
                r_l2_curr = (np.power((score_pred_all - score_all) / (score_all.max() - score_all.min()) ,2).sum() /
                             score_all.shape[0]) * 100
                self.writer.add_scalar('R_train/rho', rho_curr, epoch * data_len)
                self.writer.add_scalar('R_train/r_l2', r_l2_curr, epoch * data_len)

                results_string = f'EPOCH:{epoch}, Training:'
                results_string += f'loss_cls: {loss_cls_all / data_len:.4f}, loss_reg: {loss_reg_all / data_len:.4f}, ' \
                                  f'loss_att: {loss_att_all / data_len:.4f}, loss_var: {loss_var_all / data_len:.4f}, ' \
                                  f'correlation: {rho_curr:.4f}, r_l2: {r_l2_curr:.4f}'
                if self.cross_dd:
                    results_string += f'loss_dd: {loss_dd_all / data_len:.4f}'
                print(results_string, flush=True)

            """ Evaluation """
            if (epoch+1)<150:
                if epoch % 40 == 0 and epoch > 0:
                    self.evaluate(epoch)
            elif (epoch+1)%10 == 0:
                self.evaluate(epoch)
            elif (epoch+1)>180 and (epoch+1)%3==0:
                self.evaluate(epoch)

            #if (epoch + 1) % 10 == 0:
            #    self.evaluate(epoch)
            #elif (epoch + 1) > 50 and (epoch + 1) % 7 == 0:
            #    self.evaluate(epoch)
            #elif (epoch + 1) > 100 and (epoch + 1) % 3 == 0:
            #    self.evaluate(epoch)

    def get_mask(self, d_len):
        mask = []
        for i in d_len:
            mask_c = torch.ones(i)
            if i < self.voter_number:
                mask_c = torch.cat([mask_c, torch.zeros(self.voter_number - i)])
            mask_c /= i
            mask.append(mask_c)
        return torch.stack(mask).to(self.device)

    def get_att_loss(self, logits_all, hinge_loss):
        if self.args.cluster_norm_dim == -1:
            logits_all = logits_all.transpose(-1,-2)
        softmax_dim = logits_all.shape[1]
        temp_idx = repeat(torch.arange(1, softmax_dim + 1), 't -> b t k', b=logits_all.shape[0], k=logits_all.shape[-1]).float().to(self.device)
        cluster_mean = (logits_all * temp_idx).sum(1)
        var = (torch.abs(temp_idx - repeat(cluster_mean, 'b k -> b t k', t=softmax_dim)) * logits_all).sum(1) 
        loss_all = 0.
        for i in range(logits_all.size(-1) - 1):
            cluster_mean_former = cluster_mean[:, i]
            cluster_mean_latter = cluster_mean[:, i + 1]
            loss = hinge_loss(cluster_mean_latter, cluster_mean_former, torch.ones_like(cluster_mean_former))
            if i == 0:
                loss += hinge_loss(cluster_mean_former, torch.ones_like(cluster_mean_former), torch.ones_like(cluster_mean_former))
            if i == logits_all.size(-1) - 2:
                loss += hinge_loss(torch.ones_like(cluster_mean_former) * softmax_dim, cluster_mean_latter, torch.ones_like(cluster_mean_former))

            loss_all += loss
        # loss_all = loss_all * (5/self.args.num_cluster)
        loss_all = loss_all * (5 / 5)
        return loss_all, var.mean()

    def get_att_loss_2(self, logits_all, hinge_loss):
        if self.args.cluster_norm_dim == -1:
            logits_all = logits_all.transpose(-1,-2)
        softmax_dim = logits_all.shape[1]
        tensor = torch.tensor([i for i in range(softmax_dim)]).to(logits_all.device).float().view(-1, 1).T + 1
        cluster_mean = torch.matmul(tensor, logits_all).squeeze(1)
        loss_all = 0.
        residual_all = 0.
        for i in range(logits_all.size(-1) - 1):
            cluster_mean_former = cluster_mean[:, i]
            cluster_mean_latter = cluster_mean[:, i + 1]
            residual_all += cluster_mean_former - cluster_mean_latter
            loss = hinge_loss(cluster_mean_latter, cluster_mean_former, torch.ones_like(cluster_mean_former))
            # if i == 0:
            #     # residual_all += torch.ones_like(cluster_mean_former) - cluster_mean_former
            #     loss += hinge_loss(cluster_mean_former, torch.ones_like(cluster_mean_former), torch.ones_like(cluster_mean_former))
            # if i == logits_all.size(-1) - 2:
            #     # residual_all += cluster_mean_latter - torch.ones_like(cluster_mean_former) * softmax_dim
            #     loss += hinge_loss(torch.ones_like(cluster_mean_former) * softmax_dim, cluster_mean_latter, torch.ones_like(cluster_mean_former))
            loss_all += loss
        # loss_all = loss_all * (5/self.args.num_cluster)
        global_hinge_loss = residual_all + (self.args.margin_factor-6)
        global_hinge_loss[global_hinge_loss < 0] = 0.
        loss_all = loss_all + global_hinge_loss.mean()
        return loss_all

    # def get_att_loss(self, logits_all, kl_loss):
    #     logits_all = logits_all
    #     loss_all = 0.
    #     for i in range(logits_all.shape[-1] - 1):
    #         att_1 = logits_all[:, :, i]
    #         for j in range(i+1, logits_all.shape[-1]):
    #             # print(f'current comb {i}, {j}')
    #             att_2 = logits_all[:, :, j]
    #             loss = kl_loss(torch.log(att_1), att_2)
    #             loss_all -= loss
    #     return torch.max(torch.zeros_like(loss_all), 5. + loss_all)

    def attention_plot(self, logits_all, video_ids=None, num_layer=1, tensorboard=False):
        # logits_all: bs * clip_len * num_cluster
        bs = logits_all.shape[0]
        clip_len = logits_all.shape[1]
        num_cluster = logits_all.shape[2]
        att_save_path = os.path.join('./visualization', self.base_name)
        os.makedirs(att_save_path, exist_ok=True)
        for i in range(1):
            figure = plt.figure()
            logits_curr = logits_all[i].reshape(-1)
            video_id_i = video_ids[i]
            cluster_ids = np.repeat(np.array([i for i in range(num_cluster)]).reshape(1, -1), clip_len, axis=0).reshape(-1)
            clip_ids = np.repeat(np.array([i for i in range(clip_len)]).reshape(1, -1), num_cluster, axis=1).reshape(-1)
            data_plot = pd.DataFrame(data=[logits_curr, cluster_ids, clip_ids]).T
            data_plot.columns = ['value', 'cluster_ids', 'clip_ids']
            sns.lineplot(data=data_plot, x="clip_ids", y="value", hue="cluster_ids")
            plt.title(f'{video_id_i}_layer:{num_layer}')
            if tensorboard is False:
                plt.show()
            # plt.savefig(os.path.join(att_save_path, f'{video_id_i}.png'))
            plt.close()
            # plt.show()
        return figure

    def weighted_plot(self, weighted_logits, video_ids=None):
        bs = weighted_logits.shape[0]
        num_voter = weighted_logits.shape[1]
        num_cluster = weighted_logits.shape[2]
        att_save_path = os.path.join('./visualization', 'weighted', self.base_name)
        os.makedirs(att_save_path, exist_ok=True)
        for i in range(1):
            logits_curr = weighted_logits[i].reshape(-1)
            video_id_i = video_ids[i]
            cluster_ids = np.repeat(np.array([i for i in range(num_cluster)]).reshape(1, -1), num_voter, axis=0).reshape(
                -1)
            voter_ids = np.repeat(np.array([i for i in range(num_voter)]).reshape(1, -1), num_cluster, axis=1).reshape(-1)
            data_plot = pd.DataFrame(data=[logits_curr, cluster_ids, voter_ids]).T
            data_plot.columns = ['value', 'cluster_ids', 'voter_ids']
            sns.lineplot(data=data_plot, x="cluster_ids", y="value", hue="voter_ids")
            plt.title(video_id_i)
            # plt.savefig(os.path.join(att_save_path, f'{video_id_i}.png'))
            # plt.close()
            plt.show()


    def evaluate(self, epoch=0):
        self.model.eval()
        if not self.use_pretrain:
            self.backbone.eval()
        score_key = 'score' if self.dataset == 'AQA_7' else 'raw_score'
        with torch.no_grad():
            score_all = []
            score_pred_all = []
            score_pred_all_2 = []
            logits_all = []
            video_ids = []
            class_all = []
            v_m_1 = 0
            v_m_2 = 0
            # groups = [[] for i in range(16)]
            # groups_2 = [[] for i in range(16)]
            for (data, targets) in self.test_loader:
                video = data['video'].to(self.device)
                # score = data['score'].unsqueeze(-1).to(self.device)
                score_raw = data[score_key].unsqueeze(-1).to(self.device)
                video_id = data['id']
                score_voting = []
                distance = []
                mask_all = self.get_mask(data['t_len'])
                mask_dy = []
                leaf_id_all = []
                for i, target in enumerate(targets):
                    video_t = target['video'].to(self.device)
                    score_t = target[score_key].unsqueeze(-1).to(self.device)
                    # mask_curr = mask_all[i].unsqueeze(-1)
                    # print(f'mask_curr = {mask_curr.shape}')
                    if not self.use_pretrain:
                        video_c, video_t = torch.chunk(self.backbone(torch.cat([video, video_t], 0)), 2)
                    else:
                        video_c = video
                    #
                    out_prob, delta, logits = self.model(video_c, video_t, train=False)
                    leaf_probs = out_prob[-1].reshape(video_c.shape[0], -1)
                    leaf_id = leaf_probs.argmax(dim=-1)
                    glabel_2, _  = self.grouper.produce_label(score_raw - score_t)
                    leaf_id_g = glabel_2.argmax(0)
                    # print(delta.shape)
                    relative_scores = self.grouper.inference(leaf_probs.detach().cpu().numpy(), delta.detach().cpu().numpy()).to(self.device).float()
                    # mask_curr = ((leaf_id > 6) & (leaf_id < 9))
                    mask_curr = ((leaf_id_g > 6) & (leaf_id_g < 15))
                    mask_dy.append(mask_curr)
                    # print(f'relative_scores = {relative_scores.shape}')
                    # score_voting.append((relative_scores.to(self.device) + score_t) * dd)
                    distance.append((score_raw - score_t).abs())
                    # distance.append(relative_scores.abs())
                    # if leaf_id > 13 or leaf_id < 2:
                    # distance.append(relative_scores.abs())
                    score_pred_curr = (relative_scores + score_t)
                    if self.dataset == 'MLT_AQA':
                        score_pred_curr = score_pred_curr * data['difficulty'].float().unsqueeze(-1).to(self.device)
                    error = score_raw - score_t - relative_scores
                    score_voting.append(score_pred_curr)
                    # groups[leaf_id_g].append((score_pred_curr, score))
                    # groups_2[leaf_id_g].append(error)
                    # groups_2[]
                    # print(f'score_voting = {score_voting.shape}')
                # print(score_voting)
                distance = torch.stack(distance).squeeze(-1).T
                score_voting = torch.stack(score_voting).squeeze(-1).T
                mask_dy = torch.stack(mask_dy, dim=-1).float()
                for i, mask_curr in enumerate(mask_dy):
                    if mask_curr.sum() < 1:
                        mask_dy[i] = mask_all[i]
                mask_dy /= mask_dy.sum(dim=-1).unsqueeze(-1)

                v_m_1 += (mask_all > 0).sum().item()
                v_m_2 += (mask_dy > 0).sum().item()

                score_voting_2 = (score_voting * mask_dy).sum(-1)
                score_voting = (score_voting * mask_all).sum(-1)
                # score_voting = score_voting / len(targets)
                logits_all.append(logits)
                score_pred_all.append(score_voting)
                score_pred_all_2.append(score_voting_2)
                score_all.append(data['score'].unsqueeze(-1).to(self.device))
                video_ids.append(video_id)
                class_all.append(data['class'].to(self.device))
            # self.attention_plot(logits_all)
            score_all = torch.cat(score_all, dim=0).detach().cpu().numpy()
            if self.num_cluster > 0:
                logits_all = torch.cat(logits_all, dim=0).detach().cpu().numpy()

            score_pred_all = torch.cat(score_pred_all, dim=0).detach().cpu().numpy()
            score_pred_all_2 = torch.cat(score_pred_all_2, dim=0).detach().cpu().numpy()
            class_all = torch.cat(class_all, dim=0).detach().cpu().numpy()
            video_ids = np.concatenate(np.array(video_ids,dtype=object))
            if self.multi_gpu and self.multi_gpu_test:
                score_all = self.gather_results(score_all)
                score_pred_all = self.gather_results(score_pred_all)
                score_pred_all_2 = self.gather_results(score_pred_all_2)
                class_all = self.gather_results(class_all)
                video_ids = self.gather_results(video_ids)
                if self.num_cluster > 0 and self.args.hinge_loss:
                    logits_all = self.gather_results(logits_all)

            if self.local_rank < 1:
                if self.multi_gpu:
                    # print(f'unique video id length = {len(np.unique(video_ids))}')
                    m = np.zeros_like(video_ids, dtype=bool)
                    m[np.unique(video_ids, return_index=True)[1]] = True
                    # print(f'mask length = {len(m)}')
                    score_all = score_all[m]
                    # print(f'length after mask = {len(score)}')
                    score_pred_all = score_pred_all[m]
                    score_pred_all_2 = score_pred_all_2[m]
                    class_all = class_all[m]
                # if epoch % 5 == 0:
                if self.plot:
                    len_att = logits_all.shape[1]//self.args.num_layers
                    for i in range(self.args.num_layers):
                        self.attention_plot(logits_all[:,i*len_att:(i+1)*len_att, :], video_ids, i)

                rho_curr, r_l2_curr = self.result_handler.update_results(score_all, score_pred_all, class_all, epoch)
                if self.local_rank < 1:
                    self.writer.add_scalar('Results/rho', rho_curr, epoch * len(self.train_loader))
                    self.writer.add_scalar('Results/r_l2_curr', r_l2_curr, epoch * len(self.train_loader))
                    len_att = logits_all.shape[1] // 2
                    img_2 = self.attention_plot(logits_all[:, :len_att, :], video_ids, 2, True)
                    img_1 = self.attention_plot(logits_all[:, -len_att:, :], video_ids, 1, True)
                    self.writer.add_figure('layer1', img_1, epoch * len(self.train_loader))
                    self.writer.add_figure('layer2', img_2, epoch * len(self.train_loader))
                # self.result_handler.update_results(score_all, score_pred_all_2, None, epoch)

        return score_all, score_pred_all, video_ids

    def gather_results(self, results):
        results_multi = [None for _ in range(distributed.get_world_size())]
        distributed.all_gather_object(results_multi, results)
        return np.concatenate(results_multi)

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
        rt /= distributed.get_world_size()
        return rt


def init_seeds(seed=0, cuda_deterministic=True, multi_gpu=True):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    if multi_gpu:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = parse_opt()
    if args.local_rank < 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')
    multi_gpu = False if args.local_rank < 0 else True
    grouper = None
    if args.dataset == 'AQA_7':
        train_loader, test_loader, grouper = get_AQA7Pair_dataloader(args)
    else:
        train_loader, test_loader, grouper = get_MLTPair_dataloader(args)

    # random_seed = 12
    # init_seeds(random_seed + args.local_rank, True, multi_gpu)
    # np.random.seed(seed)  # Numpy module.
    # random.seed(seed)  # Python random module.
    # torch.manual_seed(seed)
    # if multi_gpu:
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    random_seed = 12
    np.random.seed(random_seed)  # Numpy module.
    random.seed(random_seed)  # Python random module.
    torch.manual_seed(random_seed)
    if multi_gpu:
        torch.cuda.manual_seed(random_seed + args.local_rank)
        torch.cuda.manual_seed_all(random_seed + args.local_rank)

    run = RunPair(args, device, train_loader, test_loader, grouper, multi_gpu=multi_gpu)
    run.train()
