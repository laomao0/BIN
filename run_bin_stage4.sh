#!/usr/bin/env bash



arg="
    --netName bin_stage4
    --input_path /DATA/wangshen_data/Adobe_240fps_dataset/Adobe_240fps_blur/test_blur
    --gt_path /DATA/wangshen_data/Adobe_240fps_dataset/Adobe_240fps_blur/test
    --output_path /DATA/wangshen_data/Adobe_240fps_dataset/Adobe_240fps_blur
    --time_step 0.5
    --gpu_id 2
    --opt /home/shenwang/PycharmProjects/Deblur/Deblur_Interp/src_github/BIN/options/train/train_adobe_stage4.yml
    "

python test_bin_Adobe.py $arg
