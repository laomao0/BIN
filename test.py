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

use_default_ssim = 1
if use_default_ssim == 1:
    from skimage.measure import compare_ssim,compare_psnr
    def my_compare_ssim(img1,img2):
        ssim = compare_ssim(img1, img2, multichannel=True)
        return ssim
    ssim_msg = 'skimage.measure.ssim'
else:
    from  utils.util import calculate_ssim as my_compare_ssim
    from  utils.util import calculate_psnr as compare_psnr
    ssim_msg = 'our ssim'


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
    parser.add_argument("--gt_path", type=str, required=True)
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
    GT_PATH = args.gt_path
    RESULT_PATH = os.path.join(args.output_path, str(N_frames* val_fps)+"fps_test_results")
    if not os.path.exists(RESULT_PATH):
        os.makedirs(RESULT_PATH, exist_ok=True)


    print("We interp the " + str(val_fps) + " fps blurry video to " + str(round(1/args.time_step)*val_fps) + " fps slow-motion video!")
    print("We check the our interpolated results using the test dataset to check psnr and ssim!")

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
    else:
        print("Error in Model")
        assert 1 == 0


    torch.backends.cudnn.benchmark = True  # to speed up the

    # ============================================================== #
    #                       Load Net Model                           #
    # ============================================================== #

    our_model = False
    if 'bin' in which_model:
        our_model = True

    subdir = sorted(os.listdir(INPUT_PATH))  # folder 0 1 2 3...
    gen_dir = os.path.join(RESULT_PATH, opt['name'])
    if not os.path.exists(gen_dir):
        os.mkdir(gen_dir)


    # ============================================================== #
    #                       Logger                                   #
    # ============================================================== #

    util.setup_logger('base', gen_dir, 'test', level=logging.INFO, screen=True, tofile=True)
    util.setup_logger('base_summary', gen_dir, 'test_summary', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger_summary = logging.getLogger('base_summary')

    #### log info
    logger.info('In Data: {} '.format(INPUT_PATH))
    logger.info('Padding mode: {}'.format(PAD))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(RESULT_PATH))
    logger.info('Flip test: {}'.format(flip_test))
    logger.info('Use ssin method {}'.format(ssim_msg))

    if our_model:
        pstring_model_size = 'Num. of model parameters is : {}'.format(str(count_network_parameters(model.netG)))
    else:
        pstring_model_size = 'Num. of model parameters is : {}'.format(str(count_network_parameters(model)))
    logger.info(pstring_model_size)

    # ============================================================== #
    #                       Initialize                               #
    # ============================================================== #
    total_run_time =    AverageMeter()
    interp_error =      AverageMeter()
    psnr_interp_total = AverageMeter()  #  interp total psnr
    ssim_interp_total = AverageMeter()
    psnr_deblur_total = AverageMeter()  # deblur psnr
    ssim_deblur_total = AverageMeter()  # deblur ssim
    psnr_blurry_total = AverageMeter()
    ssim_blurry_total = AverageMeter()
    tot_timer =         AverageMeter()
    proc_timer =        AverageMeter()
    end = time.time()

    interp_error_set =      AverageMeter()
    psnr_interp_total_set = AverageMeter()  # interp total psnr for a folder
    ssim_interp_total_set = AverageMeter()
    psnr_deblur_total_set = AverageMeter()  # deblur psnr
    ssim_deblur_total_set = AverageMeter()  # deblur ssim
    psnr_blurry_total_set = AverageMeter()
    ssim_blurry_total_set = AverageMeter()

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

        interp_error.reset()
        psnr_interp_total.reset()
        ssim_interp_total.reset()
        psnr_deblur_total.reset()
        ssim_deblur_total.reset()
        psnr_blurry_total.reset()
        ssim_blurry_total.reset()

        if not os.path.exists(os.path.join(gen_dir, dir)):
            os.mkdir(os.path.join(gen_dir, dir))

        logger.info("The results for dir:{}".format(dir))
        logger_summary.info("The results for dir:{}".format(dir))

        frames_path = os.path.join(INPUT_PATH,dir)  # blur path
        sharp_path = os.path.join(GT_PATH,dir)
        frames = sorted(os.listdir(frames_path)) #[0:5] #debug

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
                if our_model and args.direct_interp == True:
                    arguments_strFirst.append(os.path.join(sharp_path, tmp_num_name))
                else:
                    arguments_strFirst.append(os.path.join(frames_path, tmp_num_name))


            arguments_strSecond = []
            for i in second_5_blurry_list:
                tmp_num = int(first_frame_num + i)
                tmp_num_name = str(tmp_num).zfill(5) + '.png'
                if our_model and args.direct_interp == True:
                    arguments_strSecond.append(os.path.join(sharp_path, tmp_num_name))
                else:
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

            # handle each mid-frames

            for frame_time_step,frame_index  in zip(time_step,frame_indexs):

                middle_frame_num = interpolated_sharp_list[frame_index]  # set 4 as the middle
                middle_frame_name = str(middle_frame_num).zfill(5) + '.png'
                arguments_strOut = os.path.join(gen_dir, dir, middle_frame_name)
                # arguments_strUpload = os.path.join(RESULT_PATH_UPLOAD, upload_frame_name)

                # gt_path = os.path.join(GT_PATH, dir, "frame10i11.png")
                gt_middle_path = os.path.join(GT_PATH, dir, middle_frame_name)
                first_gt_deblur_path = os.path.join(GT_PATH, dir, first_gt_deblur_name)
                second_gt_deblur_path = os.path.join(GT_PATH, dir, second_gt_deblur_name)
                first_gt_sharp_path = os.path.join(sharp_path, first_sharp_name)
                second_gt_sharp_path = os.path.join(sharp_path, second_sharp_name)
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

                assert ( intWidth <= 1280)  # while our approach works with larger images, we do not recommend it unless you are aware of the implications
                assert ( intHeight <= 720)  # while our approach works with larger images, we do not recommend it unless you are aware of the implications

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
                        gt = read_image_np(second_gt_deblur_path)
                        res = read_image_np(arguments_strOut_second_res_deblur_path)
                        psnr_tmp = compare_psnr(res, gt)
                        ssim_tmp = my_compare_ssim(res, gt)
                        psnr_deblur_total.update(psnr_tmp, 1)
                        ssim_deblur_total.update(ssim_tmp, 1)
                        pstring = "Interp PSNR : " + str(round(psnr_tmp, 4)) + " Interp SSIM : "+ str(round(ssim_tmp, 4))
                        logger.info(pstring)


                #----------------------------------------second img------------------------------------#
                if not os.path.exists(arguments_strOut_first_res_deblur_path):
                    cv2.imwrite(arguments_strOut_first_res_deblur_path, np.round(x0_s).astype(numpy.uint8) )
                    gt = read_image_np(first_gt_deblur_path )
                    res = read_image_np(arguments_strOut_first_res_deblur_path )
                    psnr_tmp = compare_psnr(res, gt)
                    ssim_tmp = my_compare_ssim(res, gt)
                    psnr_deblur_total.update(psnr_tmp, 1)
                    ssim_deblur_total.update(ssim_tmp, 1)
                    pstring = "Interp PSNR : " + str(round(psnr_tmp, 4)) + " Interp SSIM : " + str(round(ssim_tmp, 4))
                    logger.info(pstring)


                #----------------------------------------third img------------------------------------#
                rec_rgb = read_image_np(arguments_strOut )
                gt_rgb = read_image_np(gt_middle_path )

                diff_rgb = 128.0 + rec_rgb - gt_rgb
                avg_interp_error_abs = np.mean(np.abs(diff_rgb - 128.0))

                interp_error.update(avg_interp_error_abs, 1)

                mse = numpy.mean((diff_rgb - 128.0) ** 2)
                if mse == 0:
                    return 100.0
                PIXEL_MAX = 255.0
                psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
                psnr_ = compare_psnr(rec_rgb, gt_rgb)
                ssim_tmp = my_compare_ssim(rec_rgb, gt_rgb)
                psnr_interp_total.update(psnr, 1)
                ssim_interp_total.update(ssim_tmp, 1)

                pstring = "deblur error / PSNR : " + str(round(avg_interp_error_abs, 4)) + " / " + str(round(psnr, 4))
                logger.info(pstring)
                # check blurry
                blur = read_image_np(first_blurry_path )
                psnr_tmp = compare_psnr(blur, gt_rgb)
                ssim_tmp = my_compare_ssim(blur, gt_rgb)
                psnr_blurry_total.update(psnr_tmp, 1)
                ssim_blurry_total.update(ssim_tmp, 1)
                pstring = "blurry PSNR : " + str(round(psnr_tmp, 4)) + " blurry SSIM : " + str(round(ssim_tmp, 4)) + '\n' + first_blurry_path
                logger.info(pstring)




        pstring = "The results for dir:" + dir
        logger_summary.info(pstring)
        pstring = "The average interpolation error " + str(round(interp_error.avg, 4))
        logger_summary.info(pstring)
        # end for folders

        pstring = "Avg. folder" + \
                    " blurry psnr " + str(psnr_blurry_total.avg) + \
                    " deblur psnr " + str(psnr_interp_total.avg) + \
                    " interp psnr " + str(psnr_deblur_total.avg) + \
                    " blurry ssim " + str(ssim_blurry_total.avg) + \
                    " deblur ssim " + str(ssim_interp_total.avg) + \
                    " interp ssim " + str(ssim_deblur_total.avg)

        logger_summary.info(pstring)
        interp_error_set.update(interp_error.avg, 1)
        psnr_interp_total_set.update(psnr_interp_total.avg, 1)  # interp total psnr
        ssim_interp_total_set.update(ssim_interp_total.avg, 1)
        psnr_deblur_total_set.update(psnr_deblur_total.avg, 1)  # deblur psnr
        ssim_deblur_total_set.update(ssim_deblur_total.avg, 1)  # deblur ssim
        psnr_blurry_total_set.update(psnr_blurry_total.avg, 1)
        ssim_blurry_total_set.update(ssim_blurry_total.avg, 1)

    pstring = "The results for Adobe dataset"
    logger_summary.info(pstring)
    pstring = "The average interpolation error " + str(round(interp_error_set.avg, 4))
    logger_summary.info(pstring)
    # end for folders
    pstring = "Avg. testset " + \
                " interp psnr " + str(psnr_deblur_total_set.avg) + \
                " blurry psnr" + str(psnr_blurry_total_set.avg) + \
                " deblur psnr" + str(psnr_interp_total_set.avg) + \
                " interp ssim " + str(ssim_deblur_total_set.avg) + \
                " blurry ssim" + str(ssim_blurry_total_set.avg) + \
                " deblur ssim" + str(ssim_interp_total_set.avg)


    logger_summary.info(pstring)

    pstring = "runtime per image [s] : %.4f\n" % total_run_time.avg + \
                "CPU[1] / GPU[0] : 1 \n" + \
                "Extra Data [1] / No Extra Data [0] : 1"
    logger_summary.info(pstring)
    logger_summary.info(pstring_model_size)

if __name__ == '__main__':
    main()
