# aqa_tpt
implementation of "Action Quality Assessment with Temporal Parsing Transformer"

## Usage 

### Requirement
- Python >= 3.8
- Pytorch >= 1.7.1
- cuda 10.1

### Dataset Preparation
- Please download the dataset from the [[link]](https://durhamuniversity-my.sharepoint.com/:u:/g/personal/fsvd68_durham_ac_uk/EfCexAQT19xArquObWijcaAB3xgRUpps50vfbezDh9wgAA?e=OOuVUH), and unzip the file. (the processed data is based on this [repo](https://github.com/yuxumin/CoRe))
1. put the folder named "data_preprocessed" in ./data
2. put the file named "data_preprocessed/model_rgb.pth" in ./data

The data structure should be:
```
$DATASET_ROOT
├── data/
    ├── model_rgb.pth
    ├── data_preprocessed/
        ├── MTL_AQA/
            ├── frames_long
                ├── 01_01/
                    ├── 00017977.jpg
                    ...
                ...
                └── 07_25/
                    ├── 00040170.jpg
                    ...
            ├── info
                ├── final_annotations_dict_with_dive_number
                ├── test_split_0.pkl
                └── train_split_0.pkl
        ├── AQA_7/
            ├── frames
                ├── diving-out
                    ├── 001
                        ├── img_00001.jpg
                        ...
                ...
            ├── info
                ├── split_4_test_list.mat
                ├── split_4_train_list.mat
```

### how to train
```
python -u -m torch.distributed.launch --nproc_per_node=8 train_pairencode1_decoder_1selfatt_self8head_ffn_sp_new.py --epoch_num=250 --dataset=MLT_AQA --bs_train=3 --bs_test=3 --use_pretrain=False --num_cluster=5 --margin_factor=3.2 --encode_video=False --hinge_loss=True --multi_hinge=True --d_model=512 --d_ffn=512 --exp_name=sp_new_103_5_3
```

## Acknowledgement
Our code is based on [CoRe](https://github.com/yuxumin/CoRe). Thanks for their great work!
