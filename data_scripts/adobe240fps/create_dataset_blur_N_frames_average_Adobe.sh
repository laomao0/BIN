#!/usr/bin/env bash

# we use the static build ffmpeg to de-compress the videos

python create_dataset_blur_N_frames_average.py \
        --ffmpeg_dir /home/shenwang/Software/ffmpeg-git-amd64-static/ffmpeg-git-20190701-amd64-static \
        --dataset adobe240fps_blur \
        --window_size 11 \
        --enable_train 1 \
        --dataset_folder ./Adobe_240fps_dataset/Adobe_240fps_blur \
        --videos_folder  ./Adobe_240fps_dataset/Adobe_240fps_original_high_fps_videos \
        --img_width 640 \
        --img_height 352

