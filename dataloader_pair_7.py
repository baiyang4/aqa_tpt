import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils.util import normalize, denormalize, get_video_trans, Group_helper
import glob
from PIL import Image
import pickle as pkl
from scipy import stats
import glob
import scipy.io


class SevenPair(Dataset):
    def __init__(self, args, subset, transform):
        super(SevenPair, self).__init__()
        # random.seed(0)
        self.subset = subset  # train or test
        # loading annotations
        self.args = args
        self.class_name = ['diving', 'gym_vault', 'ski_big_air', 'snowboard_big_air', 'sync_diving_3m',
                           'sync_diving_10m']

        self.data_root = os.path.join(args.data_root, args.dataset)
        self.transforms = transform
        self.use_pretrain = args.use_pretrain
        # final_annotations_dict_with_dive_number
        self.keys_train = scipy.io.loadmat(os.path.join(self.data_root, 'info', 'split_4_train_list.mat'))['consolidated_train_list']
        class_idx = float(args.class_idx)
        if class_idx < 7:
            self.keys_train = self.keys_train[self.keys_train[:, 0] == class_idx]

        self.frame_length = 103
        self.voter_number = args.num_voting
        self.contrastive_dict = {}
        self.preprocess()
        self.check()
        # self.delta()
        if self.subset == 'test':
            self.keys_test = scipy.io.loadmat(os.path.join(self.data_root, 'info', 'split_4_test_list.mat'))['consolidated_test_list']
            if class_idx < 7:
                self.keys_test = self.keys_test[self.keys_test[:, 0] == class_idx]

    def score_normalization(self):
        pass

    def load_video(self, action_class, sample_idx):
        data_path = os.path.join(self.data_root, 'frames', '{}-out'.format(self.class_name[int(action_class - 1)]))
        video_path = os.path.join(data_path, '%03d' % sample_idx)
        # something wrong with this part
        video = [Image.open(os.path.join(video_path, 'img_%05d.jpg' % (i + 1))) for i in range(self.frame_length)]
        return self.transforms(video)

    def preprocess(self):
        for item in self.keys_train:
            contrastive_item = item[0]
            if self.contrastive_dict.get(contrastive_item) is None:
                self.contrastive_dict[contrastive_item] = []
            self.contrastive_dict[contrastive_item].append(item)

    def check(self):
        for key in sorted(list(self.contrastive_dict.keys())):
            file_list = self.contrastive_dict[key]
            for item in file_list:
                assert item[0] == key
        print('check done')

    def delta(self):
        delta = []
        for key in list(self.contrastive_dict.keys()):
            file_list = self.contrastive_dict[key]
            for i in range(len(file_list)):
                for j in range(i + 1, len(file_list)):
                    delta.append(abs(
                        normalize(file_list[i][2], file_list[i][0]).astype(np.float32) -
                        normalize(file_list[j][2], file_list[j][0]).astype(np.float32)
                        ))
        return delta

    # def class_dist(self):
    def get_score_list(self, id_list, anchor_score):
        score_list = []
        for id_curr in id_list:
            score = np.array(self.annotations.get(id_curr).get('final_score')).astype(np.float32)
            difficulty = self.annotations.get(id_curr).get('difficulty')
            raw_score = np.abs(((score / difficulty).astype(np.float32) - anchor_score))
            score_list.append(raw_score)
        return score_list

    def __getitem__(self, ix):
        if self.subset == 'test':
            key = self.keys_test[ix]
        else:
            key = self.keys_train[ix]
        data = self.load_data(key)
        contrastive_list = self.contrastive_dict[key[0]].copy()

        if self.subset == 'train':
            if len(contrastive_list) > 1:
                contrastive_list.pop(list(np.sum((key == contrastive_list), axis=-1) == 3).index(True))
            idx = random.randint(0, len(contrastive_list) - 1)
            key_2 = contrastive_list[idx]
            targets = self.load_data(key_2)
        else:
            random.shuffle(contrastive_list)
            data['t_len'] = self.voter_number

            targets = []
            for key_curr in contrastive_list[:self.voter_number]:
                data_curr = self.load_data(key_curr)
                targets.append(data_curr)

        return data, targets

    def load_data(self, curr_key):
        action_class = curr_key[0]
        sample_idx = curr_key[1]
        sample_score = curr_key[2]
        data = {}
        if self.use_pretrain:
            data['video'] = torch.load(os.path.join(self.data_root, 'features', f'{self.class_name[int(action_class) - 1]}-out', f'{int(sample_idx):03d}.pt'))
        else:
            video = self.load_video(action_class, sample_idx)
            start_idx = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
            video = torch.stack([video[:, i: i + 8] for i in start_idx])  # 10*N, c, 16, h, w
            data['video'] = video
        data['score'] = normalize(sample_score, action_class).astype(np.float32)
        data['class'] = int(action_class) - 1
        data['id'] = f'{int(action_class)}_{int(sample_idx)}'
        return data

    def proc_label(self, data):
        tmp = stats.norm.pdf(np.arange(101),
                             loc=data['score'] * (101 - 1) / 104.5,
                             scale=5).astype(
            np.float32)
        data['soft_label'] = tmp / tmp.sum()

    def __len__(self):
        if self.subset == 'train':
            sample_pool = len(self.keys_train)
        else:
            sample_pool = len(self.keys_test)
        return sample_pool

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_AQA7Pair_dataloader(args):
    train_trans, test_trans = get_video_trans()
    train_dataset = SevenPair(args, transform=train_trans, subset='train')
    test_dataset = SevenPair(args, transform=test_trans, subset='test')

    grout_helper = Group_helper(train_dataset.delta(), depth=5, Max=100, Min=0)

    train_sampler = None
    test_sampler = None

    if args.local_rank > -1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        if args.multi_gpu_test:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs_train,
                                               shuffle=False if args.local_rank >= 0 else True,
                                               num_workers=int(args.workers),
                                               pin_memory=True, sampler=train_sampler,
                                               worker_init_fn=worker_init_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bs_test,
                                              num_workers=int(args.workers),
                                              pin_memory=True,
                                              shuffle=False,
                                              sampler=test_sampler
                                              )
    return train_loader, test_loader, grout_helper

