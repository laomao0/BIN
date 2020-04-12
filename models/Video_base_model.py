import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss
from models.loss import CharbonnierLossPlusSSIM
import itertools
import numpy as np
# import torch.optim.lr_scheduler.ReduceLROnPlateau as ReduceLROnPlateau

import utils.util as util
import data.util as data_util

logger = logging.getLogger('base')


class VideoBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoBaseModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='sum').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            elif loss_type == 'cb+ssim':
                self.cri_pix = CharbonnierLossPlusSSIM().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['ft_tsa_only']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))

            elif train_opt['lr_scheme'] == 'ReduceLROnPlateau':
                for optimizer in self.optimizers:  # optimizers[0] =adam
                    self.schedulers.append(  # schedulers[0] = ReduceLROnPlateau
                        torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, 'min', factor=train_opt['factor'], patience=train_opt['patience'],verbose=True))
                print('Use ReduceLROnPlateau')
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)
        if need_GT:
            self.real_H = data['GT'].to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0


    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L) # 1 x 5 x 3 x 64 x 64

        loss, loss_tmp = self.cri_pix(self.fake_H, self.real_H)

        l_pix = self.l_pix_w * loss

        if l_pix.item() > 1e-1:
            print('stop!')

        l_pix.backward()
        self.optimizer_G.step()

        self.log_dict['l_pix'] = l_pix.item()

        if loss_tmp == None:
            self.log_dict['l_pix'] = l_pix.item()
        else:
            self.log_dict['total_loss'] = l_pix.item()
            self.log_dict['l_pix'] = loss_tmp[0].item()
            self.log_dict['ssim_loss'] = loss_tmp[1].item()


    def optimize_parameters_without_schudlue(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L) # 1 x 5 x 3 x 64 x 64

        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)

        if l_pix.item() > 1e-1:
            print('stop!')

        l_pix.backward()

        self.optimizer_G.step()

        # for scheduler in self.schedulers:
        #     scheduler.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def test_stitch(self):
        """
                To hande the 4k output, we have no much GPU memory
                :return:
                """
        self.netG.eval()

        with torch.no_grad():
            imgs_in = self.var_L  # 1 NC HW

            # crop
            gtWidth = 3840
            gtHeight = 2160
            intWidth_ori = 960  # 960
            intHeight_ori = 540  # 540
            split_lengthY = 180
            split_lengthX = 320
            scale = 4
            PAD = 32

            intPaddingRight_ = int(float(intWidth_ori) / split_lengthX + 1) * split_lengthX - intWidth_ori
            intPaddingBottom_ = int(float(intHeight_ori) / split_lengthY + 1) * split_lengthY - intHeight_ori

            intPaddingRight_ = 0 if intPaddingRight_ == split_lengthX else intPaddingRight_
            intPaddingBottom_ = 0 if intPaddingBottom_ == split_lengthY else intPaddingBottom_

            pader0 = torch.nn.ReplicationPad2d([0, intPaddingRight_, 0, intPaddingBottom_])
            # print("Init pad right/bottom " + str(intPaddingRight_) + " / " + str(intPaddingBottom_))

            intPaddingRight = PAD  # 32# 64# 128# 256
            intPaddingLeft = PAD  # 32#64 #128# 256
            intPaddingTop = PAD  # 32#64 #128#256
            intPaddingBottom = PAD  # 32#64 # 128# 256

            pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight, intPaddingTop, intPaddingBottom])

            imgs_in = torch.squeeze(imgs_in, 0)  # N C H W

            imgs_in = pader0(imgs_in)  # N C 540 960

            imgs_in = pader(imgs_in)  # N C 604 1024

            assert (split_lengthY == int(split_lengthY) and split_lengthX == int(split_lengthX))
            split_lengthY = int(split_lengthY)
            split_lengthX = int(split_lengthX)
            split_numY = int(float(intHeight_ori) / split_lengthY)
            split_numX = int(float(intWidth_ori) / split_lengthX)
            splitsY = range(0, split_numY)
            splitsX = range(0, split_numX)

            intWidth = split_lengthX
            intWidth_pad = intWidth + intPaddingLeft + intPaddingRight
            intHeight = split_lengthY
            intHeight_pad = intHeight + intPaddingTop + intPaddingBottom

            # print("split " + str(split_numY) + ' , ' + str(split_numX))
            # y_all = np.zeros((1, 3, gtHeight, gtWidth), dtype="float32")  # HWC
            y_all = torch.zeros((1, 3, gtHeight, gtWidth)).to(self.device)
            for split_j, split_i in itertools.product(splitsY, splitsX):
                # print(str(split_j) + ", \t " + str(split_i))
                X0 = imgs_in[:, :,
                     split_j * split_lengthY:(split_j + 1) * split_lengthY + intPaddingBottom + intPaddingTop,
                     split_i * split_lengthX:(split_i + 1) * split_lengthX + intPaddingRight + intPaddingLeft]

                # y_ = torch.FloatTensor()

                X0 = torch.unsqueeze(X0, 0)  # N C H W -> 1 N C H W

                output = self.netG(X0) # 1 N C H W ->  1 C H W

                # if flip_test:
                #     output = util.flipx4_forward(model, X0)
                # else:
                #     output = util.single_forward(model, X0)

                output_depadded = output[:, :, intPaddingTop * scale:(intPaddingTop + intHeight) * scale,  # 1 C H W
                                  intPaddingLeft * scale: (intPaddingLeft + intWidth) * scale]

                # output_depadded = output_depadded.squeeze(0)  # C H W

                # output = util.tensor2img(output_depadded)  # C H W -> HWC

                # y_all[split_j * split_lengthY * scale:(split_j + 1) * split_lengthY * scale,
                # split_i * split_lengthX * scale:(split_i + 1) * split_lengthX * scale, :] = \
                #     np.round(output_depadded).astype(np.uint8)

                y_all[:, :, split_j * split_lengthY * scale:(split_j + 1) * split_lengthY * scale,
                split_i * split_lengthX * scale:(split_i + 1) * split_lengthX * scale] = output_depadded

            self.fake_H = y_all  # 1 N x c x 2160 x 3840

            self.netG.train()


    def get_current_log(self):
        return self.log_dict

    def get_loss(self):
        if (self.opt['train']['pixel_criterion'] == 'cb+ssim' or self.opt['train']['pixel_criterion'] == 'cb'):
            loss_temp,_ = self.cri_pix(self.fake_H, self.real_H)
            l_pix = self.l_pix_w * loss_temp
        else:
            l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        return l_pix

    # def get_loss_ssim(self):
    #     l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
    #     # todo
    #     return l_pix



    def get_current_visuals(self, need_GT=True, save=False, name=None, save_path=None):

        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        if save == True:
            import os.path as osp
            import cv2
            img = out_dict['rlt']
            img = util.tensor2img(img)
            cv2.imwrite(osp.join(save_path, '{}.png'.format(name)), img)

        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
