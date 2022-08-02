import cv2
from torchvideotransforms import video_transforms, volume_transforms
import pandas as pd
#from torchvideotransforms import video_transforms, volume_transforms
import numpy as np
from scipy import stats
import time
import torch
import os

def denormalize(label, class_idx, upper=100.0):
    label_ranges = {
        1 : (21.6, 102.6),
        2 : (12.3, 16.87),
        3 : (8.0, 50.0),
        4 : (8.0, 50.0),
        5 : (46.2, 104.88),
        6 : (49.8, 99.36)}
    label_range = label_ranges[class_idx]

    true_label = (label.float() / float(upper)) * (label_range[1] - label_range[0]) + label_range[0]
    return true_label


def normalize(label, class_idx, upper=100.0):
    label_ranges = {
        1 : (21.6, 102.6),
        2 : (12.3, 16.87),
        3 : (8.0, 50.0),
        4 : (8.0, 50.0),
        5 : (46.2, 104.88),
        6 : (49.8, 99.36)}
    label_range = label_ranges[class_idx]

    norm_label = ((label - label_range[0]) / (label_range[1] - label_range[0]) ) * float(upper)
    return norm_label

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def fisher_z_mean_correlation(rho_list):
    z_list = [np.arctanh(value) for value in rho_list]
    mean_z = np.array(z_list).mean()
    return (np.exp(mean_z) - np.exp(-mean_z))/(np.exp(mean_z) + np.exp(-mean_z))

def get_video_trans():
    train_trans = video_transforms.Compose([
        video_transforms.RandomHorizontalFlip(),
        video_transforms.Resize((455,256)),
        video_transforms.RandomCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_trans = video_transforms.Compose([
        video_transforms.Resize((455,256)),
        video_transforms.CenterCrop(224),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_trans, test_trans


def distributed_concat(tensor):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat

class ResultsHandler:
    def __init__(self, model, backbone, dataset, save_name, checkpoint, local_rank=-1, multi_gpu=False, class_idx=7):
        super(ResultsHandler, self).__init__()
        self.last_epoch = 0
        self.print = local_rank <= 0
        self.model = model
        self.backbone = backbone
        self.saving_name = save_name
        self.multi_gpu = multi_gpu
        self.checkpoint_path = os.path.join('./results', self.saving_name, 'checkpoint.pth')

        self.optimizer = None
        # establish results tracker
        self.results_tracker = pd.DataFrame({'Corr.': ['Curr_rho', 'Best_rho', 'Curr_rl2', 'Best_rl2', 'Best_corr_epoch', 'Best_l2_epoch']})
        multi_action = True if (dataset == 'AQA_7' and class_idx > 6) else False

        self.epoch_start_time = None
        self.epoch_end_time = None
        self.multi_action = multi_action
        if multi_action:
            self.classname = ['diving', 'gym_vault', 'ski_big_air', 'snowboard_big_air', 'sync_diving_3m', 'sync_diving_10m']
            for action_name in self.classname + ['Avg']:
                self.results_tracker[action_name] = -1
        else:
            self.results_tracker['metrics'] = -1
            if checkpoint is not None:
                self.results_tracker.iloc[1, -1] = checkpoint['rho_best']
                self.results_tracker.iloc[3, -1] = checkpoint['RL2_min']
                self.results_tracker.iloc[4, -1] = checkpoint['epoch_best']
        if self.print:
            print(self.saving_name)

    def start_epoch(self):
        if self.print:
            self.epoch_start_time = time.time()

    def _save_check_point(self, epoch, best=False):
        save_dic = {
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'epoch_best': self.results_tracker.iloc[4, -1],
            'rho_best': self.results_tracker.iloc[1, -1],
            'RL2_min': self.results_tracker.iloc[3, -1]
        }
        if self.multi_gpu:
            save_dic['model'] = self.model.module.state_dict()
        else:
            save_dic['model'] = self.model.state_dict()
        if self.backbone is not None:
            if self.multi_gpu:
                save_dic['backbone'] = self.backbone.module.state_dict()
            else:
                save_dic['backbone'] = self.backbone.state_dict()
        if os.path.exists(os.path.join('./results', self.saving_name)) is False:
            os.makedirs(os.path.join('./results', self.saving_name), exist_ok=True)

        if best:
            value_list = [float(i[:-4]) for i in os.listdir(os.path.join('./results', self.saving_name)) if i.split('.')[0].isnumeric()]
            best_value = 0 if len(value_list) == 0 else max(value_list)
            if best_value < self.results_tracker.iloc[1, -1]:
                pre_best_path = os.path.join('./results', self.saving_name, f'{best_value}.pth')
                if os.path.exists(pre_best_path):
                    os.remove(pre_best_path)
                torch.save(save_dic, os.path.join('./results', self.saving_name, f'{self.results_tracker.iloc[1, -1]:.4f}.pth'))
        else:
            torch.save(save_dic, os.path.join('./results', self.saving_name, f'checkpoint_{epoch}.pth'))

    def update_results(self, score, score_pred, action_class, epoch):
        if self.print:
            # self.epoch_end_time = time.time()
            # print('train time: %.3fs' % (self.epoch_end_time - self.epoch_start_time))
            # save check point
            if self.multi_action:
                rho_all = []
                rl2_all = []
                rho_all_ind = []
                rl2_all_ind = []
                classes = list(np.unique(action_class))
                classes.sort()
                for class_curr in classes:
                    index_curr = action_class == class_curr
                    score_curr = score[index_curr]
                    score_pred_curr = score_pred[index_curr]
                    rho_curr, p = stats.spearmanr(score_pred_curr, score_curr)
                    rl2_curr = (np.power((score_pred_curr.reshape(-1, 1) - score_curr) / (score_curr.max() - score_curr.min()), 2).sum() /
                                 score_curr.shape[0]) * 100
                    rho_pre = self.results_tracker.iloc[1, int(class_curr) + 1]
                    rl2_pre = self.results_tracker.iloc[3, int(class_curr) + 1]
                    if rho_pre > rho_curr:
                        rho_all_ind.append(rho_pre)
                        rl2_all_ind.append(rl2_pre)
                    else:
                        rho_all_ind.append(rho_curr)
                        rl2_all_ind.append(rl2_curr)
                    rho_all.append(rho_curr)
                    rl2_all.append(rl2_curr)
                rho_mean = fisher_z_mean_correlation(rho_all)
                rho_mean_ind = fisher_z_mean_correlation(rho_all_ind)
                rl2_mean = np.array(rl2_all).mean()
                rl2_mean_ind = np.array(rl2_all_ind).mean()
                rho_all.append(rho_mean)
                rho_all_ind.append(rho_mean_ind)
                rl2_all.append(rl2_mean)
                rl2_all_ind.append(rl2_mean_ind)
                # print(len(rho_all))
                # print(len(rho_all_ind))
                # print(self.results_tracker.shape)
                self.results_tracker.iloc[0, 1:] = rho_all
                self.results_tracker.iloc[2, 1:] = rl2_all
                rho_best = self.results_tracker.iloc[1, -1]
                if rho_mean_ind > rho_best:
                    self.results_tracker.iloc[1, 1:] = rho_all_ind
                    self.results_tracker.iloc[3, 1:] = rl2_all_ind
                    print('results updated')
                print(f'-------------epoch {epoch}')
                print(self.results_tracker)
                return rho_mean_ind
            else:
                rho_curr, p = stats.spearmanr(score_pred, score)
                print(score.shape)
                r_l2_curr = (np.power((score_pred.reshape(-1,1) - score) / (score.max() - score.min()) ,2).sum() /
                             score.shape[0]) * 100
                self.results_tracker.iloc[0, -1] = rho_curr
                self.results_tracker.iloc[2, -1] = r_l2_curr
                rho_best = self.results_tracker.iloc[1, -1]
                if rho_curr > rho_best:
                    self.results_tracker.iloc[1, -1] = rho_curr
                    # self.results_tracker.iloc[3, -1] = r_l2_curr
                    self.results_tracker.iloc[4, -1] = int(epoch)
                    self._save_check_point(epoch, best=True)
                    print('results updated')
                r_l2_best = self.results_tracker.iloc[3, -1]
                if r_l2_best < 0:
                    r_l2_best = 100
                if r_l2_curr < r_l2_best:
                    self.results_tracker.iloc[3, -1] = r_l2_curr
                    self.results_tracker.iloc[5, -1] = int(epoch)

                # if epoch % 5 == 0 or rho_curr > rho_best:
                print(f'-------------epoch {epoch}')
                print(self.results_tracker)
                if (epoch + 1) % 5 == 0:
                    self._save_check_point(epoch)
                return rho_curr, r_l2_curr


class Group_helper(object):
    def __init__(self, dataset, depth, Symmetrical=True, Max=None, Min=None):
        '''
            dataset : list of deltas (CoRe method) or list of scores (RT method)
            depth : depth of the tree
            Symmetrical: (bool) Whether the group is symmetrical about 0.
                        if symmetrical, dataset only contains th delta bigger than zero.
            Max : maximum score or delta for a certain sports.
        '''
        self.dataset = sorted(dataset)
        self.length = len(dataset)
        self.num_leaf = 2 ** (depth - 1)
        self.symmetrical = Symmetrical
        self.max = Max
        self.min = Min
        self.Group = [[] for _ in range(self.num_leaf)]
        self.build()

    def build(self):
        '''
            separate region of each leaf
        '''
        if self.symmetrical:
            # delta in dataset is the part bigger than zero.
            for i in range(self.num_leaf // 2):
                # bulid positive half first
                Region_left = self.dataset[int((i / (self.num_leaf // 2)) * (self.length - 1))]

                if i == 0:
                    if self.min != None:
                        Region_left = self.min
                    else:
                        Region_left = self.dataset[0]
                Region_right = self.dataset[int(((i + 1) / (self.num_leaf // 2)) * (self.length - 1))]
                if i == self.num_leaf // 2 - 1:
                    if self.max != None:
                        Region_right = self.max
                    else:
                        Region_right = self.dataset[-1]
                self.Group[self.num_leaf // 2 + i] = [Region_left, Region_right]
            for i in range(self.num_leaf // 2):
                self.Group[i] = [-i for i in self.Group[self.num_leaf - 1 - i]]
            for group in self.Group:
                group.sort()
        else:
            for i in range(self.num_leaf):
                Region_left = self.dataset[int((i / self.num_leaf) * (self.length - 1))]
                if i == 0:
                    if self.min != None:
                        Region_left = self.min
                    else:
                        Region_left = self.dataset[0]
                Region_right = self.dataset[int(((i + 1) / self.num_leaf) * (self.length - 1))]
                if i == self.num_leaf - 1:
                    if self.max != None:
                        Region_right = self.max
                    else:
                        Region_right = self.dataset[-1]
                self.Group[i] = [Region_left, Region_right]

    def produce_label(self, scores):
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy().reshape(-1,)
        glabel = []
        rlabel = []
        for i in range(self.num_leaf):
            # if in one leaf : left == right
            # we should treat this leaf differently
            leaf_cls = []
            laef_reg = []
            for score in scores:
                if score >= 0 and (score < self.Group[i][1] and score >= self.Group[i][0]):
                    leaf_cls.append(1)
                elif score < 0 and (score <= self.Group[i][1] and score > self.Group[i][0]):
                    leaf_cls.append(1)
                else:
                    leaf_cls.append(0)

                if leaf_cls[-1] == 1:
                    if self.Group[i][1] == self.Group[i][0]:
                        rposition = score - self.Group[i][0]
                    else:
                        rposition = (score - self.Group[i][0]) / (self.Group[i][1] - self.Group[i][0])
                else:
                    rposition = -1
                laef_reg.append(rposition)
            glabel.append(leaf_cls)
            rlabel.append(laef_reg)
        glabel = torch.tensor(glabel).cuda()
        rlabel = torch.tensor(rlabel).cuda()
        return glabel, rlabel

    def inference(self, probs, deltas):
        '''
            probs: bs * leaf
            delta: bs * leaf
        '''
        predictions = []
        for n in range(probs.shape[0]):
            prob = probs[n]
            delta = deltas[n]
            leaf_id = prob.argmax()
            if self.Group[leaf_id][0] == self.Group[leaf_id][1]:
                prediction = self.Group[leaf_id][0] + delta[leaf_id]
            else:
                try:
                    prediction = self.Group[leaf_id][0] + (self.Group[leaf_id][1] - self.Group[leaf_id][0]) * delta[leaf_id]
                except:

                    print(self.Group)
                    print(delta.shape)
                    print(f'leaf_id = {leaf_id}------')
                    print(delta)
                    prediction = None
            predictions.append(prediction)
        return torch.tensor(predictions).reshape(-1, 1)

    def get_Group(self):
        return self.Group

    def number_leaf(self):
        return self.num_leaf
