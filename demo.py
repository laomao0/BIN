"""
    This script evals the deblur and interpolation results.
"""
import time
import os
import threading
import glob
import logging
import torch
from torch.autograd import Variable
from torch.autograd import gradcheck
import sys
import getopt
import math
import numpy
import torch
import random
import numpy as np
import os
import numpy
import utils.AverageMeter as AverageMeter
import shutil
import  time
import utils.util as util
import data.util as data_util
import argparse
import options.options as option
import cv2
from models import create_model


def read_image(img_path):
    '''read one image from img_path
    Return  CHW torch [0,1] RGB
    '''
    # img: HWC, BGR, [0,1], numpy
    img_GT = cv2.imread(img_path)
    img = img_GT.astype(np.float32) / 255.

    # BGR to RGB
    img = img[:,:,[2, 1, 0]]
    # HWC to CHW, to torch  
    img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
    return img

def read_image_np(img_path):
    '''read one image from img_path
    Return HWC RGB [0,255]
    '''
    # img: HWC, BGR, [0,1], numpy
    img = cv2.imread(img_path)
    # BGR to RGB
    img = img[:,:,[2, 1, 0]]
    return img

def count_network_parameters(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    N = sum([numpy.prod(p.size()) for p in parameters])
    return N

def main():

    #################
    # configurations
    #################

    parser = argparse.ArgumentParser()
    parser.add_argument("--netName", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--deblur_model_path", type=str)
    parser.add_argument("--interp_model_path", type=str)
    parser.add_argument("--gpu_id", type=str, required=True)
    parser.add_argument("--time_step", type=float, default=0.5)
    parser.add_argument("--direct_interp", type=bool,default=False)
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    if not args.opt==None: # not our method do not need an opt file
        opt = option.parse(args.opt, is_train=False)
        if args.launcher == 'none':  # disabled distributed training
            opt['dist'] = False
            rank = -1
            print('Disabled distributed training.')
        else:
            opt['dist'] = True
            # init_dist()
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        model_path = "Joint Model:" + opt['path']['pretrain_model_G']
    else:
        opt={}
        opt['name'] = args.netName
        model_path = 'Interp:' + args.interp_model_path + ' Deblur:' + args.deblur_model_path


    val_fps = 30
    BLUR_TYPE = 'blur'  # or blur_gamma
    N_frames = round(1/args.time_step)

    # saving pathd
    INPUT_PATH = args.input_path
    RESULT_PATH = args.output_path
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH, exist_ok=True)
    gen_dir = RESULT_PATH


    print("We interp the " + str(val_fps) + " fps blurry video to " + str(round(1/args.time_step)*val_fps) + " fps slow-motion video!")

    flip_test = False
    PAD = 32

    #### set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    print("Num of GPU ", torch.cuda.device_count())
    device = torch.device('cuda')
    dtype = torch.cuda.FloatTensor

    # ============================================================== #
    #                       Set Models                               #
    # ============================================================== #
    ####
    which_model = args.netName
    if 'bin' in which_model:
        model = create_model(opt)

    torch.backends.cudnn.benchmark = True  

    our_model = False
    if 'bin' in which_model:
        our_model = True

    subdir = sorted(os.listdir(INPUT_PATH))  # folder 0 1 2 3...
    

    print('In Data: {} '.format(INPUT_PATH))
    print('Padding mode: {}'.format(PAD))
    print('Model path: {}'.format(model_path))
    print('Save images: {}'.format(RESULT_PATH))

    # ============================================================== #
    #                       Initialize                               #
    # ============================================================== #
    total_run_time =    AverageMeter()
    tot_timer =         AverageMeter()
    proc_timer =        AverageMeter()
    end = time.time()

    time_offsets_all = [kk * 1.0 / N_frames for kk in range(1, int(N_frames), 1)]
    time_step = range(0, N_frames - 1)

    if our_model:
        version = opt['network_G']['version']
        nframes = opt['network_G']['nframes']
        if nframes == 6:
            version = 1  # limit to lstm 

    assert version == 1
    assert our_model == True

    for dir in subdir:

        cnt = 0
        model.prev_state = None
        model.hidden_state = None

        if not os.path.exists(os.path.join(gen_dir, dir)):
            os.mkdir(os.path.join(gen_dir, dir))

        frames_path = os.path.join(INPUT_PATH,dir)
        frames = sorted(os.listdir(frames_path))

        shift_file = 1
        offset_file = 0

        for index, frame in enumerate(frames):

            if index == 0:
                first_frame_num = int(frame[:-4])

            if index >= len(frames)-1:
                break

            first_5_blurry_list = [max(index-2,0), max(index-1,0), min(index, len(frames)-1), min(index+1, len(frames)-1), min(index+2, len(frames)-1)]
            second_5_blurry_list = [max(index - 1, 0), max(index - 0, 0), min(index +1, len(frames)-1), min(index +2, len(frames)-1),min(index + 3, len(frames)-1)]

            first_5_blurry_list = [i*8 for i in first_5_blurry_list]
            second_5_blurry_list = [i*8 for i in second_5_blurry_list]

            # list the input two blurry frames
            arguments_strFirst = []

            for i in first_5_blurry_list:
                tmp_num = int(first_frame_num + i)
                tmp_num_name = str(tmp_num).zfill(5) + '.png'
                arguments_strFirst.append(os.path.join(frames_path, tmp_num_name))

            arguments_strSecond = []
            for i in second_5_blurry_list:
                tmp_num = int(first_frame_num + i)
                tmp_num_name = str(tmp_num).zfill(5) + '.png'
                arguments_strSecond.append( os.path.join(frames_path, tmp_num_name))

            first_sharp_name = str(int(first_frame_num + first_5_blurry_list[2])).zfill(5) + '.png'
            second_sharp_name = str(int(first_frame_num + second_5_blurry_list[2])).zfill(5) + '.png'

            second_frame_num = int( int(frame[:-4])+8)

            first_gt_deblur = int(int(frame[:-4]) * shift_file + offset_file + 4)
            second_gt_deblur = int(second_frame_num * shift_file + offset_file + 4)

            first_gt_deblur_name = str(first_gt_deblur).zfill(5) + '.png'
            second_gt_deblur_name = str(second_gt_deblur).zfill(5) + '.png'

            # blurry
            first_blurry_path = arguments_strFirst[2]  # the middle blurry frames
            second_blurry_path = arguments_strSecond[2]

            interpolated_sharp_list = range(first_gt_deblur+1, second_gt_deblur)
            first_blurry_path = arguments_strSecond[2]  # the middle blurry frames
            second_blurry_path = arguments_strSecond[3]

            if len(time_step) == 1:  #
                print("interpolate middle frame")
                frame_indexs = [3]
                if our_model:
                    if nframes == 6:
                        frame_indexs = [3]
            else:
                assert len(time_step) == len(time_offsets_all)
                frame_indexs = range(0,7)
                print("interpolate all 7 frames")

            for frame_time_step,frame_index  in zip(time_step,frame_indexs):

                middle_frame_num = interpolated_sharp_list[frame_index]  # set 4 as the middle
                middle_frame_name = str(middle_frame_num).zfill(5) + '.png'
                arguments_strOut = os.path.join(gen_dir, dir, middle_frame_name)

                arguments_strOut_first_res_deblur_path = os.path.join(gen_dir, dir, first_gt_deblur_name)
                arguments_strOut_second_res_deblur_path = os.path.join(gen_dir, dir, second_gt_deblur_name)

                testData = []

                list_tmp = [arguments_strFirst[0], arguments_strFirst[1], arguments_strFirst[2], arguments_strFirst[3], arguments_strFirst[4],arguments_strSecond[4]] 
                for i in list_tmp:
                    testData.append(read_image(i).to(device))

                y_ = torch.FloatTensor()
                intWidth = testData[0].size(2)
                intHeight = testData[0].size(1)
                channel = testData[0].size(0)
                if not channel == 3:
                    continue

                assert ( intWidth <= 1280)
                assert ( intHeight <= 720)

                if intWidth != ((intWidth >> 7) << 7):
                    intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
                    intPaddingLeft =int(( intWidth_pad - intWidth)/2)
                    intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
                else:
                    intWidth_pad = intWidth
                    intPaddingLeft = 32
                    intPaddingRight= 32

                if intHeight != ((intHeight >> 7) << 7):
                    intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
                    intPaddingTop = int((intHeight_pad - intHeight) / 2)
                    intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
                else:
                    intHeight_pad = intHeight
                    intPaddingTop = 32
                    intPaddingBottom = 32

                pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

                torch.set_grad_enabled(False)


                testData = [pader(Variable(torch.unsqueeze(u,0))) for u in testData]
                testData.append(torch.unsqueeze(torch.tensor(frame_index), 0))

                proc_end = time.time()

                if not os.path.exists(arguments_strOut):

                    model.test_set_input(testData)
                    model.test_forward()
                    y_ = model.Ft_p[13]     # I7_prime_prime_prime
                    x0_s = model.Ft_p[8]  # I6_prime_prime
                    x1_s = model.Ft_p[12]  # I8_prime_prime
                    s2 = model.Ft_p[7]  # I4_prime_prime
                    s3 = model.Ft_p[9]    # I5_prime_prime_prime_prime

                    # if index >=3:
                    proc_timer.update(time.time() -proc_end)
                    tot_timer.update(time.time() - end)
                    end  = time.time()
                    print("*****************current image process time \t " + str(time.time()-proc_end )+"s ******************" )
                    total_run_time.update(time.time()-proc_end,1)

                    # HWC BGR
                    x0_s = util.tensor2img(x0_s.squeeze(0))[intPaddingTop:intPaddingTop + intHeight,intPaddingLeft: intPaddingLeft + intWidth,:]
                    x1_s = util.tensor2img(x1_s.squeeze(0))[intPaddingTop:intPaddingTop + intHeight,intPaddingLeft: intPaddingLeft + intWidth,:]
                    y_ = util.tensor2img(y_.squeeze(0))[intPaddingTop:intPaddingTop + intHeight,intPaddingLeft: intPaddingLeft + intWidth,:]


                    s2 = util.tensor2img(s2.squeeze(0))[intPaddingTop:intPaddingTop + intHeight,intPaddingLeft: intPaddingLeft + intWidth,:]
                    s3 = util.tensor2img(s3.squeeze(0))[intPaddingTop:intPaddingTop + intHeight,intPaddingLeft: intPaddingLeft + intWidth,:]

                    cv2.imwrite(arguments_strOut,  np.round(y_).astype(numpy.uint8))

                if index < len(frames)-2:
                    if not os.path.exists(arguments_strOut_second_res_deblur_path):
                        cv2.imwrite(arguments_strOut_second_res_deblur_path, np.round(x1_s).astype(numpy.uint8))


                if not os.path.exists(arguments_strOut_first_res_deblur_path):
                    cv2.imwrite(arguments_strOut_first_res_deblur_path, np.round(x0_s).astype(numpy.uint8) )


if __name__ == '__main__':
    main()
