# aqa_tpt
implementation of "Action Quality Assessment with Temporal Parsing Transformer"

### Dataset Preparation
- Please download the dataset from the [[link]](https://durhamuniversity-my.sharepoint.com/:u:/g/personal/fsvd68_durham_ac_uk/EfCexAQT19xArquObWijcaAB3xgRUpps50vfbezDh9wgAA?e=OOuVUH), and 
1. put the folder named "data_preprocessing" in ./data
2. put the 


# how to train
```
python -u -m torch.distributed.launch --nproc_per_node=8 train_pairencode1_decoder_1selfatt_self8head_ffn_sp_new.py --epoch_num=250 --dataset=MLT_AQA --bs_train=3 --bs_test=3 --use_pretrain=False --num_cluster=7 --margin_factor=2 --encode_video=False --hinge_loss=True --multi_hinge=True --d_model=512 --d_ffn=512 --exp_name=sp_new_103_7_2
```
