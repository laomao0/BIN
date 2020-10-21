# demo: joint frame interpolation and deblurring

arg="
    --netName bin_stage4
    --input_path ./demo_videos/demo_blur
    --output_path ./demo_videos/demo_results
    --time_step 0.5
    --gpu_id 2
    --opt ./options/train/train_adobe_stage4.yml
    "

python demo.py $arg
