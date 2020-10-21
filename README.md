# BIN (Blurry Video Frame Interpolation)
[Project]() **|** [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shen_Blurry_Video_Frame_Interpolation_CVPR_2020_paper.pdf)

<!-- # PRF (Video Frame Interpolation and Enhancement via Pyramid Recurrent Framework)
[Project]() **|** [Paper](To be published in TIP) -->


[Wang Shen](https://sites.google.com/view/wangshen94),
[Wenbo Bao](https://sites.google.com/view/wenbobao/home),
[Guangtao Zhai](https://scholar.google.ca/citations?user=E6zbSYgAAAAJ&hl=zh-CN),
Li Chen,
[Xiongkuo Min](https://sites.google.com/site/minxiongkuo/home),
and
Zhiyong Gao

IEEE Conference on Computer Vision and Pattern Recognition, Seattle, CVPR 2020


### Table of Contents
1. [Introduction](#introduction)
1. [Citation](#citation)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)
1. [Testing Pre-trained Models](#testing-pre-trained-models)
1. [Downloading Results](#downloading-results)
1. [Training New Models](#training-new-models) 

### Introduction

We propose a **B**lurry video frame **IN**terpolation method to reduce motion blur and up-convert frame rate simultaneously.
We provide videos [here](https://www.youtube.com/watch?v=C_bL9YQJU1w).

Futher more, in the journal version (accepted by TIP), we also extend our model for joint frame interpolation and deblurring with compression artifacts, joint frame interpolation and super-resolution.
We provide videos [here](https://www.youtube.com/watch?v=IjRvaAe1HME&ab_channel=MAOLAO).

### Citation

If you find the code and datasets useful in your research, please cite:

[Frame interpolation for blurry video](https://github.com/laomao0/BIN)
     @inproceedings{BIN,
        author    = {Shen, Wang and Bao, Wenbo and Zhai, Guangtao and Chen, Li and Min, Xiongkuo and Gao, Zhiyong}, 
        title     = {Blurry Video Frame Interpolation},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
        year      = {2020}
    }

[Frame interpolation and enhancement](https://github.com/laomao0/BIN)
     @inproceedings{BIN,
        author    = {Shen, Wang and Bao, Wenbo and Zhai, Guangtao and Chen, Li and Min, Xiongkuo and Gao, Zhiyong}, 
        title     = {Video Frame Interpolation and Enhancement via Pyramid Recurrent Framework},
        booktitle = {IEEE Transactions on Image Processing},
        year      = {2020}
    }

[Frame interpolation for normal video](https://github.com/baowenbo/DAIN/)
    @inproceedings{DAIN,
        author    = {Bao, Wenbo and Lai, Wei-Sheng and Ma, Chao and Zhang, Xiaoyun and Gao, Zhiyong and Yang, Ming-Hsuan},
        title     = {Depth-Aware Video Frame Interpolation},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
        year      = {2019}
    }

[Frame interpolation MEMC architecture](https://github.com/baowenbo/MEMC-Net)
    @article{MEMC-Net,
         title={MEMC-Net: Motion Estimation and Motion Compensation Driven Neural Network for Video Interpolation and Enhancement},
         author={Bao, Wenbo and Lai, Wei-Sheng, and Zhang, Xiaoyun and Gao, Zhiyong and Yang, Ming-Hsuan},
         journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
         doi={10.1109/TPAMI.2019.2941941},
         year={2018}
    }

### Requirements and Dependencies
- Ubuntu (We test with Ubuntu = 16.04.5 LTS)
- Python (We test with Python = 3.6.8 in Anaconda3 = 4.1.1)
- Cuda & Cudnn (We test with Cuda = 10.0 and Cudnn = 7.4)
- PyTorch >= 1.0.0 (We test with Pytorch = 1.3.0)
- FFmpeg (We test with the static build version = ffmpeg-git-20190701-amd64-static)
- GCC (Compiling PyTorch 1.0.0 extension files (.c/.cu) requires gcc = 4.9.1 and nvcc = 10.0 compilers)
- NVIDIA GPU (We use RTX-2080 Ti with compute = 7.5)

### Installation
Download repository:

    $ git clone https://github.com/laomao0/BIN.git


### Make Adobe240 blur dataset

   If you want to directly download the testset, please refer to 5.

1. Download the Adobe240 original [videos](http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset_Original_High_FPS_Videos.zip)

2. Then de-compress those videos into a folder: Adobe_240fps_dataset/Adobe_240fps_original_high_fps_videos

The structure of the folder is as following:

    Adobe_240fps_original_high_fps_videos   -- 720p_240fps_1.mov
                                            -- 720p_240fps_2.mov
                                            -- 720p_240fps_3.mov
                                            -- ...

3. Make the Adobe240 blur dataset by averaging N frames.

We averaging 11 consecutive frames to synthesize 1 blur image.

For example, the frame indexs of a 240-fps video are 0 1 2 3 4 5 6 7 8 9 10 11 12...
We average 0-11 frames to synthesize the blur frame 0, average 8-19 frames to synthesize the blur frame 1.
The frame rate of synthesized blur video is 30-fps.

    $ cd data_scripts/adobe240fps
    $ ./create_dataset_blur_N_frames_average_Adobe.sh

If you do not want to create the training set, setting --enable_train to be 0.

4. Check the dataset

The script of step 3 will create the dataset at path specified at --dataset_folder
It contains 7 folders, including full_sharp, test, test_blur, test_list, train, train_blur, train_list

    full_sharp: contain all de-compressed frames, not used in this project.
    test/train: contain the sharp test/train frams at 240-fps.
    test_blur/train_blur: contain the blur test/train frames at 30-fps.
    test_list/train_list: contain im_list files used for dataloader.

    test/train structure:
                            folder_1 -- 00001.png 00002.png ....
                            folder_2 -- 00001.png 00002.png ....
    test_blur/train_blur structure:
                            folder_1 -- 00017.png 00025.png ....
                            folder_2 -- 00017.png 00025.png ....

5. For those who only want the Adobe240 blur testset with ground-truth frames, we provide download links.
   For the Adobe240 blur train set, which is too large, we suggest users to use high-fps vidoes to generate.

   Adobe_240fps_dataset/test_blur: [link](https://drive.google.com/file/d/1Lt__BO1cshm6rayCVmDJfgMxT2cLqS5s/view?usp=sharing)
   Adobe_240fps_dataset/test: [link](https://drive.google.com/file/d/11QLEfgh6JMf1FRy9XY798IkzwAOWXNk_/view?usp=sharing)
   Adobe_240fps_dataset/test_list: [link](https://drive.google.com/file/d/1Bf_Mp_ny2N1bhUkVCo3kBYgLCWPe5Xf7/view?usp=sharing)


### Demo using Pre-trained Models

1. Download pretrained model trained on Adobe240 blur training set,
    
        $ cd model_weights
        $ download the model for Adobe240 dataset
    
    [download link](https://drive.google.com/open?id=1KGu8bLcIHODGQKw8fZ4NCVB1VgSVepo9)

2. Download the demo vidoes

        $ cd demo_vidoes
        $ mkdir demo_blur
        $ download the data at the following link, then put it into demo_blur folder 

    [download link](https://drive.google.com/file/d/10c6jMuBCQmXzEtoRRZt90IMfXq6weqrM/view?usp=sharing)


3. Run the script

        $ cd ..
        $ mkdir demo_results
        $ cd ..
        $ bash demo.sh


### Testing Pre-trained Models (Performance Evaluation)

1. Download pretrained model trained on Adobe240 blur training set,

        $ cd model_weights
        $ download the model for Adobe240 dataset

    [download link]( https://drive.google.com/file/d/1FtuZTKeExX2rrlyNGnWpxd8wGWZTftMg/view?usp=sharing)

2. Run the script

        $ bash test.sh

3. Check the results

The logging file and images are saved at --output_path/60fps_test_results/adobe_stage4

test_summary.log records PSNR, SSIM of each video folders

We get the following performance:

        Frame Interpolation PSNR/SSIM : 33.31/0.9372
        Frame Deblurring    PSNR/SSIM : 33.33/0.9319

4. Besides, we also provide our results on adobe240 blur test set [here](https://drive.google.com/file/d/1Bf3PokOqXol2z6_W819bLwU_tZrxDoxe/view?usp=sharing): 
   
The downloaded zip file includes:

        a. image folders contains results: 720p_240fps_1 -- 00021.png 00025.png ....
                                            GOPR9635      -- 00021.png 00025.png ....
                                            ....          -- 00021.png 00025.png ....
                                        
        b. test.log : records each img's evaluation performance

        c. test_summary.log : records the folder's average performance


### Training New Models

If you want to train the model on our own data

    $ bash train.sh ( to be added)

    

### For joint frame interpolation and super-resolution task

In our extented work, we extend our model for joint frame interpolation and super-resolution task.

    We provide the operations for user to evaluate our model on Vimeo90K dataset.

1. Download Vimeo_septuplet dataset

        $ cd Vimeo90k_SR
        $ mkdir vimeo_septuplet
        $ download data

    [download link](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip) [82G]

2. Create dataset using matlab bicubic function

We generate the data using matlab2015b installed on the Ubuntu system.

        $ cd data_scripts
        $ cd vimeo90k_sr
        $ matlab -nodesktop -nosplash -r generate_LR_Vimeo90K


3. Download the model trained on Vimeo90k-septuplet set.

        To be added

4. Run the script

        To be added


### Contact
[Wang Shen](mailto:wangshen834@gmail.com); [Wenbo Bao](mailto:bwb0813@gmail.com); 

### License
