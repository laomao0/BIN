#!/usr/bin/env bash

# use single GPU
python train.py \
        --opt /home/shenwang/PycharmProjects/Deblur/Deblur_Interp/src_github/BIN/options/train/train_adobe_stage4.

# use multiple GPUs, for example 2
# python -m torch.distributed.launch \
#      --nproc_per_node=2 \
#      --master_port=1589 \
#      train.py \
#      --opt /home/shenwang/PycharmProjects/Deblur/Deblur_Interp/src_github/BIN/options/train/train_adobe_stage4.yml \ 
#      --launcher pytorch
