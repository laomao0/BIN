import os.path as osp
import os.path
import torch
import torch.utils.data as data
import data.util as util
import json
import os
import math
import random
import numpy as np

class BINDataset(data.Dataset):
    """
    A video test dataset. Support:
    Vid4
    REDS4
    Vimeo90K-Test

    no need to prepare LMDB files
    """
    def __init__(self, opt):
        super(BINDataset, self).__init__()
        self.opt = opt
        self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
        self.data_type = self.opt['data_type']
        self.input_frame_size = self.opt['LQ_size']
        self.all_paths , _ = self._make_dataset_deep_long_(dir=self.LQ_root,sharp_index=(2,3),mode=opt['name'])
        print("\n Dataset Initialized\n")

    def __getitem__(self, index):

        path = self.all_paths[index]

        LQs, GTenh, GTinp, key = self.Adobe_BIN_loader(im_path_pair=path, input_frame_size=self.input_frame_size)

        # stack images to NHWC, N is the frame number
        img_LQs = np.stack(LQs, axis=0)  # N H W C
        img_GTenh = np.stack(GTenh, axis=0)
        img_GTinp = np.stack(GTinp, axis=0)

        # BGR to RGB, cv2 default
        img_LQs = img_LQs[:, :, :, [2, 1, 0]]
        img_GTenh = img_GTenh[:, :, :, [2, 1, 0]]
        img_GTinp = img_GTinp[:, :, :, [2, 1, 0]]

        # HWC to CHW, numpy to tensor
        img_LQs = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQs, (0, 3, 1, 2)))).float()
        img_GTenh = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTenh, (0, 3, 1, 2)))).float()
        img_GTinp = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GTinp, (0, 3, 1, 2)))).float()

        return {'LQs': img_LQs,
                'GTenh': img_GTenh,
                'GTinp': img_GTinp,
                'key': key }



    def __len__(self):
        return len(self.all_paths)


    @staticmethod
    def Adobe_BIN_loader(im_path_pair, input_frame_size=(3, 128, 256), data_aug=True, transform=None):
        # ([blurry_frame_path, sharp_frame_path, interp_frame_path])

        key = im_path_pair[3] # 'IMG_0034_00809'

        if data_aug and random.randint(0, 1):
            # right order
            B1_path = im_path_pair[0][0]
            B3_path = im_path_pair[0][1]
            B5_path = im_path_pair[0][2]
            B7_path = im_path_pair[0][3]
            B9_path = im_path_pair[0][4]
            B11_path = im_path_pair[0][5]

            I1_path = im_path_pair[1][0]
            I3_path = im_path_pair[1][1]
            I5_path = im_path_pair[1][2]
            I7_path = im_path_pair[1][3]
            I9_path = im_path_pair[1][4]
            I11_path = im_path_pair[1][5]

            I2_path = im_path_pair[2][0]
            I4_path = im_path_pair[2][1]
            I6_path = im_path_pair[2][2]
            I8_path = im_path_pair[2][3]
            I10_path = im_path_pair[2][4]
        else:
            # inverse order
            B1_path = im_path_pair[0][5]
            B3_path = im_path_pair[0][4]
            B5_path = im_path_pair[0][3]
            B7_path = im_path_pair[0][2]
            B9_path = im_path_pair[0][1]
            B11_path = im_path_pair[0][0]

            I1_path = im_path_pair[1][5]
            I3_path = im_path_pair[1][4]
            I5_path = im_path_pair[1][3]
            I7_path = im_path_pair[1][2]
            I9_path = im_path_pair[1][1]
            I11_path = im_path_pair[1][0]

            I2_path = im_path_pair[2][4]
            I4_path = im_path_pair[2][3]
            I6_path = im_path_pair[2][2]
            I8_path = im_path_pair[2][1]
            I10_path = im_path_pair[2][0]

        B1 = util.read_img(B1_path)
        B3 = util.read_img(B3_path)
        B5 = util.read_img(B5_path)
        B7 = util.read_img(B7_path)
        B9 = util.read_img(B9_path)
        B11 = util.read_img(B11_path)

        I1 = util.read_img(I1_path)
        I3 = util.read_img(I3_path)
        I5 = util.read_img(I5_path)
        I7 = util.read_img(I7_path)
        I9 = util.read_img(I9_path)
        I11 = util.read_img(I11_path)

        I2 = util.read_img(I2_path)

        I4 = util.read_img(I4_path)
        I6 = util.read_img(I6_path)
        I8 = util.read_img(I8_path)
        I10 = util.read_img(I10_path)

        h_offset = random.choice(range(352 - input_frame_size[1] + 1))
        w_offset = random.choice(range(640 - input_frame_size[2] + 1))

        B1 = B1[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2],:]  # imresize(im_pre, (128, 424))[:, 20:404, :]
        B3 = B3[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2],:]  # imresize(im_pre, (128, 424))[:, 20:404, :]
        B5 = B5[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2],:]  # imresize(im_pre, (128, 424))[:, 20:404, :]
        B7 = B7[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2],:]  # imresize(im_pre, (128, 424))[:, 20:404, :]
        B9 = B9[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2],:]  # imresize(im_pre, (128, 424))[:, 20:404, :]
        B11 = B11[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2],:]  # imresize(im_pre, (128, 424))[:, 20:404, :]

        I1 = I1[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2],:]  # imresize(im_pre, (128, 424))[:, 20:404, :]
        I3 = I3[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2],:]  # imresize(im_pre, (128, 424))[:, 20:404, :]
        I5 = I5[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2],:]  # imresize(im_pre, (128, 424))[:, 20:404, :]
        I7 = I7[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2],:]  # imresize(im_pre, (128, 424))[:, 20:404, :]
        I9 = I9[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2],:]  # imresize(im_pre, (128, 424))[:, 20:404, :]
        I11 = I11[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2],:]  # imresize(im_pre, (128, 424))[:, 20:404, :]

        I2 = I2[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2],:]  # imresize(im_pre, (128, 424))[:, 20:404, :]
        I4 = I4[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2],:]  # imresize(im_pre, (128, 424))[:, 20:404, :]
        I6 = I6[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2],:]  # imresize(im_pre, (128, 424))[:, 20:404, :]
        I8 = I8[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2],:]  # imresize(im_pre, (128, 424))[:, 20:404, :]
        I10 = I10[h_offset:h_offset + input_frame_size[1], w_offset: w_offset + input_frame_size[2],:]  # imresize(im_pre, (128, 424))[:, 20:404, :]

        if data_aug:
            if random.randint(0, 1):
                
                B1 = np.fliplr(B1)
                B3 = np.fliplr(B3)
                B5 = np.fliplr(B5)
                B7 = np.fliplr(B7)
                B9 = np.fliplr(B9)
                B11 = np.fliplr(B11)

                I1 = np.fliplr(I1)
                I3 = np.fliplr(I3)
                I5 = np.fliplr(I5)
                I7 = np.fliplr(I7)
                I9 = np.fliplr(I9)
                I11 = np.fliplr(I11)


                I2 = np.fliplr(I2)
                I4 = np.fliplr(I4)
                I6 = np.fliplr(I6)
                I8 = np.fliplr(I8)
                I10 = np.fliplr(I10)


        return [B1, B3, B5, B7, B9, B11], \
            [I1, I3, I5, I7, I9, I11], \
                [I2, I4, I6, I8, I10], \
                    key

    @staticmethod
    def _make_dataset_deep_long_(dir,sharp_index, split=100, mode='train'):
        """
            This script make the dataset of the Adobe240 dataset.
            The dataset has the following feature.

                Before use this dataloader, we must ensure use src/data/create_dataset_blur_N_frames_average.py create
            the dataset. This create_dataset_blur_N_frames_average.py use N consecutive frames to do average to form the
            blurry frame. And the blurry output is 30fps.

            Our dataset has the following structure.
            test: contains all the sharp test frames in 240fps.
            train: contains all the sharp train frames in 240fps.
            test_blur: contains the blurry test frames in 30fps.
            train_blur: contains the blurry train frames in 30fps.
        """
        # load the train list
        framesPath = []

        num_win_per_bunch = 4

        dir_train = os.path.join(dir, mode)
        dir_train_blur = os.path.join(dir, mode+'_blur')

        dir_train_blur_list = os.path.join(dir, mode+'_list')

        # list all the blurry in dir
        for index ,folder in enumerate(os.listdir(dir_train_blur)):  # GOPRxxxx_11_xx ...


            shift = 1
            offset = 0

            sharp_folder = os.path.join(dir_train, folder)
            blur_folder = os.path.join(dir_train_blur, folder)

            blur_pics = sorted(os.listdir(blur_folder))
            blur_pics_len = len(blur_pics)

            num_win = int(blur_pics_len-num_win_per_bunch-1)

            blur_list_path = os.path.join(dir_train_blur_list, folder+'_im_list.txt')
            f = open(blur_list_path, 'r')
            blur_pic_list = f.read().split('\n')
            blur_pic_list = sorted(blur_pic_list)

            first_frama_name = sorted(os.listdir(blur_folder))[0]
            first_frame_index = int(first_frama_name[:-4])

            basic_index_for_bunch_sharp =  [0, 8, 16, 24, 32, 40]
            basic_index_for_interp_sharp = [ 4, 12, 20, 28, 36]
            basic_index_for_bunch_blurry = [0, 8, 16, 24, 32, 40]

            for _ in range(num_win):
                blurry_frame_path = []
                sharp_frame_path = []
                interp_frame_path = []
                for index, ind in enumerate(basic_index_for_bunch_blurry):  # blurry
                    blurry_indx = first_frame_index + ind
                    blurry_indx = str(blurry_indx)
                    blurry_indx = blurry_indx.zfill(5)
                    if index == 0:
                        name_b = blurry_indx
                    blurry_indx = blurry_indx + ".png"
                    blurry_indx_path = os.path.join(blur_folder, blurry_indx)
                    blurry_frame_path.append(blurry_indx_path)

                # sharp
                for ind in basic_index_for_bunch_sharp:
                    sharp_indx = int(np.floor((ind + first_frame_index)* shift + offset))
                    sharp_indx = str(sharp_indx)
                    sharp_indx = sharp_indx.zfill(5)
                    sharp_indx = sharp_indx + ".png"
                    sharp_path = os.path.join(sharp_folder, sharp_indx)
                    sharp_frame_path.append(sharp_path)

                # interp
                for ind in basic_index_for_interp_sharp:
                    interp_indx = int(np.floor((ind + first_frame_index)* shift + offset))
                    interp_indx = str(interp_indx)
                    interp_indx = interp_indx.zfill(5)
                    interp_indx = interp_indx + ".png"
                    interp_path = os.path.join(sharp_folder, interp_indx)
                    interp_frame_path.append(interp_path)

                key = folder + '_' + name_b

                add_frame = True
                for blur_frame in blurry_frame_path:
                    if not os.path.split(blur_frame)[1] in blur_pic_list:
                        add_frame = False
                if add_frame == True:
                    framesPath.append([blurry_frame_path, sharp_frame_path, interp_frame_path, key])

                basic_index_for_bunch_blurry = [i+8 for i in basic_index_for_bunch_blurry]
                basic_index_for_bunch_sharp =  [i+8 for i in basic_index_for_bunch_sharp]
                basic_index_for_interp_sharp = [i+8 for i in basic_index_for_interp_sharp]

        random.shuffle(framesPath)

        split_index = int(math.floor(len(framesPath) * split / 100.0))
        assert (split_index >= 0 and split_index <= len(framesPath))

        return (framesPath[:split_index], framesPath[split_index:]) if split_index < len(framesPath) else (framesPath, [])