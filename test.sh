#!/usr/bin/env bash

arg="
    --netName bin_stage4
    --input_path  ./Adobe_240fps_dataset/test_blur
    --gt_path     ./Adobe_240fps_dataset/test
    --output_path ./Adobe_240fps_dataset/
    --time_step 0.5
    --gpu_id 2
    --opt ./options/train/train_adobe_stage4.yml
    "

python test.py $arg

