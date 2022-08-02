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


class MLTPair(Dataset):
    def __init__(self, args, subset, transform):
        super(MLTPair, self).__init__()
        # random.seed(0)
        self.subset = subset  # train or test
        # loading annotations
        self.args = args
        self.data_root = os.path.join(args.data_root, args.dataset)
        self.transforms = transform
        self.use_pretrain = args.use_pretrain
        self.use_dd = args.use_dd
        self.multi_e = args.multi_e
        self.annotations = pkl.load(open(os.path.join(self.data_root, 'info', 'augmented_final_annotations_dict.pkl'), 'rb'))
        # final_annotations_dict_with_dive_number
        self.keys_train = pkl.load(open(os.path.join(self.data_root, 'info', 'train_split_0.pkl'), 'rb'))
        self.frame_length = 103
        self.voter_number = args.num_voting
        self.contrastive_dict = {}
        self.contrastive_key = 'rotation_type' # difficulty or dive_number or rotation_type
        if self.use_dd:
            self.contrastive_key = 'difficulty'
        self.preprocess()
        self.check()
        if self.subset == 'test':
            self.keys_test = pkl.load(open(os.path.join(self.data_root, 'info', 'test_split_0.pkl'), 'rb'))
            self.score_partition()

    def load_video(self, sample_idx, exemplar=False):
        video_path = os.path.join(self.data_root, 'frames_long', f'{sample_idx[0]:02d}_{sample_idx[1]:02d}')
        image_list = sorted((glob.glob(os.path.join(video_path, '*.jpg'))))

        frame_start_idx = 0
        if self.subset == 'train':
            temporal_aug_shift = random.randint(-3, 3)
            frame_start_idx = 3 + temporal_aug_shift
        if exemplar:
            frame_start_idx = 3

        video = [Image.open(image_path) for image_path in image_list[frame_start_idx:frame_start_idx+self.frame_length]]
        return self.transforms(video)

    def preprocess(self):
        for item in self.keys_train:
            contrastive_item = self.annotations.get(item)[self.contrastive_key]
            if self.contrastive_dict.get(contrastive_item) is None:
                self.contrastive_dict[contrastive_item] = []
            self.contrastive_dict[contrastive_item].append(item)

    def check(self):
        for key in sorted(list(self.contrastive_dict.keys())):
            file_list = self.contrastive_dict[key]
            for item in file_list:
                assert self.annotations[item][self.contrastive_key] == key
        print('check done')

    def delta(self):
        delta = []
        for key in list(self.contrastive_dict.keys()):
            file_list = self.contrastive_dict[key]
            for i in range(len(file_list)):
                for j in range(i + 1, len(file_list)):
                    delta.append(abs(
                        self.annotations[file_list[i]]['final_score'] / self.annotations[file_list[i]]['difficulty'] -
                        self.annotations[file_list[j]]['final_score'] / self.annotations[file_list[j]]['difficulty']))
        return delta

    def score_partition(self):
        scores = []
        for key in self.keys_test:
            final_score = self.annotations[key]['final_score']
            scores.append(final_score)
        scores = np.array(scores)
        print('hh')

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
        data['id'] = f'{key[0]}_{key[1]}'
        dd_current = self.annotations[key][self.contrastive_key]
        contrastive_list = self.contrastive_dict[dd_current].copy()
        # contrastive_list = self.keys_train.copy()
        # contrastive_list = self.contrastive_dict[self.annotations[(22, 16)][self.contrastive_key]].copy()
        # score_list = self.get_score_list(contrastive_list, data['raw_score'])
        # contrastive_list = [x for _, x in sorted(zip(score_list, contrastive_list))]
        if self.subset == 'train':
            if len(contrastive_list) > 1:
                contrastive_list.pop(contrastive_list.index(key))
            if self.multi_e:
                random.shuffle(contrastive_list)
                data['t_len'] = len(contrastive_list) if len(
                    contrastive_list) < self.voter_number else self.voter_number
                if len(contrastive_list) < self.voter_number:
                    dd_list = list(self.contrastive_dict.keys())
                    dd_list.pop(dd_list.index(dd_current))
                    dd_borrow = min(dd_list, key=lambda x: abs(x - dd_current))
                    need = self.voter_number - len(contrastive_list)
                    contrastive_list = contrastive_list + self.contrastive_dict[dd_borrow].copy()[:need]
                    if len(contrastive_list) < self.voter_number:
                        dd_list.pop(dd_list.index(dd_borrow))
                        dd_borrow = min(dd_list, key=lambda x: abs(x - dd_current))
                        need = self.voter_number - len(contrastive_list)
                        contrastive_list = contrastive_list + self.contrastive_dict[dd_borrow].copy()[:need]
                targets = []
                for key_curr in contrastive_list[:self.voter_number]:
                    data_curr = self.load_data(key_curr, exemplar=True)
                    targets.append(data_curr)

            else:
                idx = random.randint(0, len(contrastive_list) - 1)
                key_2 = contrastive_list[idx]
                targets = self.load_data(key_2)
        else:
            random.shuffle(contrastive_list)
            data['t_len'] = len(contrastive_list) if len(contrastive_list) < self.voter_number else self.voter_number
            if len(contrastive_list) < self.voter_number:
                dd_list = list(self.contrastive_dict.keys())
                dd_list.pop(dd_list.index(dd_current))
                dd_borrow = min(dd_list, key=lambda x: abs(x - dd_current))
                need = self.voter_number - len(contrastive_list)
                contrastive_list = contrastive_list + self.contrastive_dict[dd_borrow].copy()[:need]
                if len(contrastive_list) < self.voter_number:
                    dd_list.pop(dd_list.index(dd_borrow))
                    dd_borrow = min(dd_list, key=lambda x: abs(x - dd_current))
                    need = self.voter_number - len(contrastive_list)
                    contrastive_list = contrastive_list + self.contrastive_dict[dd_borrow].copy()[:need]
            # data['t_len'] = len(contrastive_list)
            targets = []
            for key_curr in contrastive_list[:self.voter_number]:
                data_curr = self.load_data(key_curr, exemplar=True)
                targets.append(data_curr)
            # print(key, len(targets), len(contrastive_list))
            # if len(targets) < 10:
            #     print(key, len(targets), len(contrastive_list))
        return data, targets

    def load_data(self, curr_key, exemplar=False):
        data = {}
        if self.use_pretrain:
            feature = torch.load(os.path.join(self.data_root, 'features_20_958', f'{curr_key[0]:02d}_{curr_key[1]:02d}.pt'))
            # feature = torch.repeat_interleave(feature, 2, 0)
            data['video'] = feature
        else:
            # data['video'] = self.load_video(curr_key, exemplar)
            video = self.load_video(curr_key, exemplar)
            start_idx = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
            video = torch.stack([video[:, i: i + 8] for i in start_idx])  # 10*N, c, 16, h, w
            data['video'] = video

        data['score'] = np.array(self.annotations.get(curr_key).get('final_score')).astype(np.float32)
        data['difficulty'] = self.annotations.get(curr_key).get('difficulty')
        data['raw_score'] = (data['score'] / data['difficulty']).astype(np.float32)
        data['class'] = 1
        return data

    # def load_list_data(self, keys):
    #     videos = []
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
    # worker_seed = torch.initial_seed() % 2 ** 32
    # np.random.seed(worker_seed)
    # random.seed(worker_seed)

def get_MLTPair_dataloader(args):
    train_trans, test_trans = get_video_trans()
    train_dataset = MLTPair(args, transform=train_trans, subset='train')
    test_dataset = MLTPair(args, transform=test_trans, subset='test')

    grout_helper = Group_helper(train_dataset.delta(), depth=5, Max=30, Min=0)

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

