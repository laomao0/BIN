import torch.nn.functional as F
import utils.AverageMeter as AverageMeter
import numpy
import torch.nn.init as weight_init
import math
import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss
import itertools
import utils.util as util
import data.util as data_util

logger = logging.getLogger('base')

class bin_model(BaseModel):
    """
        The model for Blurry Video Frame Interpolation 
    """
    def __init__(self, opt):
        super(bin_model, self).__init__(opt)

        self.nframes = int(opt['network_G']['nframes'])
        self.version = int(opt['network_G']['version'])

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1 # non dist training
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
            self.loss_type = train_opt['pixel_criterion']

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='sum').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
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

            self.avg_log_dict = OrderedDict()
            self.inst_log_dict = OrderedDict()
    
    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()
        
        self.optimizer_G.zero_grad()
        self.Ft_p = self.forward()
        self.loss, self.loss_list = self.get_loss(ret=1)

        l_pix = self.l_pix_w * self.loss

        l_pix.backward()
        self.optimizer_G.step()

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def feed_data(self, trainData, need_GT=True):

        # Read all inputs

        LQs =   trainData['LQs']  # B N C H W
        GTenh = trainData['GTenh']
        GTinp = trainData['GTinp']

        # print('LQs.size', LQs.shape)  # NCHW

        B1 =  LQs[:,0,...]
        B3 =  LQs[:,1,...]
        B5 =  LQs[:,2,...]
        B7 =  LQs[:,3,...]
        B9 =  LQs[:,4,...]
        B11 =  LQs[:,5,...]

        I1 =  GTenh[:,0,...]
        I3 =  GTenh[:,1,...]
        I5 =  GTenh[:,2,...]
        I7 =  GTenh[:,3,...]
        I9 =  GTenh[:,4,...]
        I11 =  GTenh[:,5,...]

        I2  = GTinp[:,0,...]
        I4  = GTinp[:,1,...]
        I6  = GTinp[:,2,...]
        I8  = GTinp[:,3,...]
        I10  = GTinp[:,4,...]

        self.B1 = B1.to(self.device)
        self.B3 = B3.to(self.device)
        self.B5 = B5.to(self.device)
        self.B7 = B7.to(self.device)
        self.B9 = B9.to(self.device)
        self.B11 = B11.to(self.device)

        self.I1 = I1.to(self.device)
        self.I3 = I3.to(self.device)
        self.I5 = I5.to(self.device)
        self.I7 = I7.to(self.device)
        self.I9 = I9.to(self.device)
        self.I11 = I11.to(self.device)

        self.I2 = I2.to(self.device)
        self.I4 = I4.to(self.device)
        self.I6 = I6.to(self.device)
        self.I8 = I8.to(self.device)
        self.I10 = I10.to(self.device)


        # shape
        self.batch = self.I1.size(0)
        self.channel = self.I1.size(1)
        self.height = self.I1.size(2)
        self.width = self.I1.size(3)

    def test_set_input(self, testData):

        # Read all inputs
        if self.nframes == 1:

            B1, B3, frame_index = testData

            self.B1 = B1.to(self.device)
            self.B3 = B3.to(self.device)

        elif self.nframes == 3:

            B1, B3, B5, _ = testData

            self.B1 = B1.to(self.device)
            self.B3 = B3.to(self.device)
            self.B5 = B5.to(self.device)

        elif self.nframes == 4 and self.version == 1:  # long-term LSTM

            B1, B3, B5, _ = testData

            self.B1 = B1.to(self.device)
            self.B3 = B3.to(self.device)
            self.B5 = B5.to(self.device)

        elif (self.nframes == 4 and self.version == 2) or (self.nframes == 4 and self.version == 3):  # short-term LSTM

            B1, B3, B5, B7, _ = testData

            self.B1 = B1.to(self.device)
            self.B3 = B3.to(self.device)
            self.B5 = B5.to(self.device)
            self.B7 = B7.to(self.device)

        elif (self.nframes == 4 and self.version == 4) or (self.nframes == 4 and self.version == 5):

            B1, B3, B5, B7, _ = testData

            self.B1 = B1.to(self.device)
            self.B3 = B3.to(self.device)
            self.B5 = B5.to(self.device)
            self.B7 = B7.to(self.device)

        elif self.nframes == 5:

            B1, B3, B5, B7, B9, _ = testData

            self.B1 = B1.to(self.device)
            self.B3 = B3.to(self.device)
            self.B5 = B5.to(self.device)
            self.B7 = B7.to(self.device)
            self.B9 = B9.to(self.device)

        elif self.nframes == 6:

            B1, B3, B5, B7, B9, B11, _ = testData

            self.B1 = B1.to(self.device)
            self.B3 = B3.to(self.device)
            self.B5 = B5.to(self.device)
            self.B7 = B7.to(self.device)
            self.B9 = B9.to(self.device)
            self.B11 = B11.to(self.device)


        # shape
        self.batch = self.B1.size(0)
        self.channel = self.B1.size(1)
        self.height = self.B1.size(2)
        self.width = self.B1.size(3)

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            if self.nframes == 1:
                if self.opt['network_G']['which_model_G'] == 'deep_long_stage1_memc':
                    indata = torch.stack((self.B1, self.B3), dim=0)
                    Ft_p = self.netG(indata)[0]
                    Ft_p = [Ft_p[-1]]
                else:
                    Ft_p = self.netG(self.B1, self.B3)
            elif self.nframes == 3:
                Ft_p = self.netG(self.B1, self.B3, self.B5)
            elif self.nframes == 4:
                Ft_p = self.netG(self.B1, self.B3, self.B5, self.B7)
            elif self.nframes == 5:
                Ft_p = self.netG(self.B1, self.B3, self.B5, self.B7, self.B9)
            elif self.nframes == 6:
                Ft_p = self.netG(self.B1, self.B3, self.B5, self.B7, self.B9, self.B11)

        self.netG.train()
        self.Ft_p = Ft_p

        return Ft_p

    def forward(self):

        if self.nframes == 1:
            if self.opt['network_G']['which_model_G'] == 'deep_long_stage1_memc':
                indata = torch.stack((self.B1, self.I2, self.B3), dim=0)
                Ft_p = self.netG(indata)[-1]
                Ft_p = [Ft_p]
            else:
                Ft_p = self.netG(self.B1, self.B3)
        elif self.nframes == 3:
            Ft_p = self.netG(self.B1, self.B3, self.B5)
        elif self.nframes == 4:
            Ft_p = self.netG(self.B1, self.B3, self.B5, self.B7)
        elif self.nframes == 5:
            Ft_p = self.netG(self.B1, self.B3, self.B5, self.B7, self.B9)
        elif self.nframes == 6:
            Ft_p = self.netG(self.B1, self.B3, self.B5, self.B7, self.B9, self.B11)

        self.Ft_p = Ft_p

        return Ft_p

    def reset_state(self):
        self.netG.prev_state = None
        self.netG.hidden_state = None

    def get_current_log(self, mode='train'):
        # get the averaged loss
        num = self.get_info()

        self.avg_log_dict = OrderedDict()
        self.avg_psnr_dict = OrderedDict()
        self.inst_log_dict = OrderedDict()

        if mode == 'train':
            for i in range(num):
                self.avg_log_dict[str(i)] = self.train_loss_total[i].avg
                self.inst_log_dict[str(i)] = self.loss_list[i].item()
            # the total train loss
            self.avg_log_dict['Al'] = self.train_loss_total[-1].avg

            return self.inst_log_dict,  self.avg_log_dict

        elif mode == 'val':
            psnr_total_avg = 0
            ssim_total_avg = 0
            val_loss_total_avg = 0
            for i in range(num):
                self.avg_log_dict['Al'+str(i)] = self.val_loss_total[i].avg
                self.avg_psnr_dict['Ap'+str(i)] = self.psnr_interp[i].avg
                # self.avg_log_dict['Avg. ssim'+str(i)] = self.ssim_interp[i].avg
                psnr_total_avg = psnr_total_avg + self.psnr_interp[i].avg
                ssim_total_avg = ssim_total_avg + self.ssim_interp[i].avg

            self.avg_log_dict['Al'] = self.val_loss_total[-1].avg
            self.avg_psnr_dict['Ap'] = psnr_total_avg/num

            val_loss_total_avg = self.val_loss_total[-1].avg

            return self.avg_log_dict, self.avg_psnr_dict, psnr_total_avg/num, ssim_total_avg/num, val_loss_total_avg

    def test_forward(self):

        if self.nframes == 1:
            self.Ft_p = self.netG(self.B1, self.B3)
        elif self.nframes == 3:
            self.Ft_p = self.netG(self.B1, self.B3, self.B5)
        elif self.nframes == 4:
            if self.version == 1:
                self.Ft_p = self.netG.test_forward(self.B1, self.B3, self.B5)
            elif self.version == 2 or self.version == 3:
                self.Ft_p = self.netG(self.B1, self.B3, self.B5, self.B7)
            elif self.version == 4 or self.version == 5:
                self.Ft_p = self.netG(self.B1, self.B3, self.B5, self.B7)
        elif self.nframes == 5:
            if self.version == 2:
                self.Ft_p = self.netG(self.B1, self.B3, self.B5, self.B7, self.B9)
            else:
                self.Ft_p = self.netG(self.B1, self.B3, self.B5, self.B7, self.B9)
        elif self.nframes == 6:
            self.Ft_p = self.netG(self.B1, self.B3, self.B5, self.B7, self.B9, self.B11)

    def test_sharp_forward(self):
        """
            Direct interp use sharp frames.
        """
        if self.nframes == 1:
            self.Ft_p = self.netG(self.I1, self.I3)
        elif self.nframes == 3:
            self.Ft_p = self.netG(self.I1, self.I3, self.I5)
        elif self.nframes == 4:
            self.Ft_p = self.netG(self.I1, self.I3, self.I5, self.I7)
        elif self.nframes == 5:
            self.Ft_p = self.netG(self.I1, self.I3, self.I5, self.I7, self.I9)

    def get_loss(self, ret=0):

        loss_list = []
        num, gt_list = self.get_info(mode=1)
        assert num == len(gt_list)  # if num == 1, todo  modify model
        for idx, gt in enumerate(gt_list):
            loss = self.cri_pix(self.Ft_p[idx], gt)
            loss_list.append(loss)
        loss = sum(loss_list) / len(loss_list)

        if self.nframes == 4 and self.version == 5:
            cyc_loss_I4 = self.cri_pix(self.Ft_p[1], self.Ft_p[5])
            loss_list.append(cyc_loss_I4)
            loss = sum(loss_list) / len(loss_list)
        if self.nframes == 6 and self.version == 2:
            cyc_loss_I4 = self.cri_pix(self.Ft_p[1], self.Ft_p[7])
            cyc_loss_I5 = self.cri_pix(self.Ft_p[5], self.Ft_p[9])
            cyc_loss_I6 = self.cri_pix(self.Ft_p[2], self.Ft_p[8])
            loss_list.append(cyc_loss_I4)
            loss_list.append(cyc_loss_I5)
            loss_list.append(cyc_loss_I6)
            loss = sum(loss_list) / len(loss_list)

        loss_list = loss_list[:num]


        if ret == 1:
            return loss, loss_list
        else:
            self.loss = loss
            self.loss_list = loss_list

    def get_current_visuals(self, need_GT=True):
        """
            For validation, the batchsize is always 1
        """ 
        self.Restored_IMG = []
        self.Restored_GT_IMG = [] 

        num, gt_list, lq_list = self.get_info(mode=2)
        rlt_list = self.Ft_p

        assert num == len(gt_list)

        out_dict = OrderedDict()
        out_dict['LQ'] = [data.detach()[0].float().cpu() for data in lq_list]
        out_dict['rlt'] = [rlt_list[idx].detach()[0].float().cpu() for idx in range(num)]
        if need_GT:
            out_dict['GT'] = [data.detach()[0].float().cpu() for data in gt_list]

        return out_dict

    def train_AverageMeter(self):
        num = self.get_info() + 1
        self.train_loss_total = []
        for i in range(num):
            self.train_loss_total.append(AverageMeter())

    def train_AverageMeter_update(self):

        num = len(self.loss_list)
        for i in range(num):
            self.train_loss_total[i].update(self.loss_list[i].item(), 1)
        # the total train loss
        self.train_loss_total[num].update(self.loss.item(), 1)

    def train_AverageMeter_reset(self):

        num = self.get_info() + 1
        for i in range(num):
            self.train_loss_total[i].reset()

    def val_loss_AverageMeter(self):
        num = self.get_info() + 1
        self.val_loss_total = []
        for i in range(num):
            self.val_loss_total.append(AverageMeter())

    def val_loss_AverageMeter_update(self, loss_list, avg_loss):
        num = len(loss_list)
        for i in range(num):
            self.val_loss_total[i].update(loss_list[i].item(), 1)

        # the total train loss
        self.val_loss_total[num].update(avg_loss.item(), 1)

    def val_loss_AverageMeter_reset(self):
        num = len(self.loss_list) + 1
        for i in range(num):
            self.val_loss_total[i].reset()

    def get_info(self, mode=0):
        if self.nframes == 1:
            num = 1
        elif self.nframes == 3:
            num = 3
        elif self.nframes == 4:
            if self.version == 4 or self.version == 5:
                num = 6
            else:
                num = 5
        elif self.nframes == 5:
            if self.version == 2:
                num = 9
            else:
                num = 10
        elif self.nframes == 6:
            num = 14

        if not mode == 0:
            if self.nframes == 1:
                gt_list = [self.I2]
                lq_list = [self.B1, self.B3]
            elif self.nframes == 3:
                gt_list = [self.I2, self.I4, self.I3]
                lq_list = [self.B1, self.B3, self.B5]
            elif self.nframes == 4:
                if self.version == 4 or self.version == 5:
                    gt_list = [self.I2, self.I4, self.I6, 
                            self.I3, self.I5, self.I4]
                else:
                    gt_list = [self.I2, self.I4, self.I3, 
                            self.I6, self.I5]
                lq_list = [self.B1, self.B3, self.B5, self.B7]
            elif self.nframes == 5:
                if self.version == 2:
                    gt_list = [self.I2, self.I4, self.I6, 
                                self.I3, self.I5, self.I4, 
                                self.I8, self.I7, self.I6]
                else:
                    gt_list =[self.I2, self.I4, self.I6,
                            self.I8, self.I3, self.I5,
                            self.I7, self.I4, self.I6, self.I5]
                lq_list = [self.B1, self.B3, self.B5, self.B7, self.B9]
            elif self.nframes == 6:
                gt_list = [self.I2, self.I4, self.I6,
                                self.I8, self.I3, self.I5,
                                self.I7, self.I4, self.I6,
                                self.I5, self.I10, self.I9,
                                self.I8, self.I7]
                lq_list = [self.B1, self.B3, self.B5, self.B7, self.B9, self.B11]

        if mode == 0:
            return num
        elif mode == 1:
            return num, gt_list
        elif mode == 2:
            return num, gt_list, lq_list

    def val_AverageMeter_para(self):
        num = self.get_info()
        self.psnr_interp = []
        self.ssim_interp = []
        for i in range(num):
            self.psnr_interp.append(AverageMeter())
            self.ssim_interp.append(AverageMeter())

    def val_AverageMeter_para_update(self, psnr_interp_t, ssim_interp_t):
        num = len(self.psnr_interp)
        for i in range(num):
            self.psnr_interp[i].update(psnr_interp_t[i], 1)
            self.ssim_interp[i].update(ssim_interp_t[i], 1)

    def val_AverageMeter_para_reset(self):
        num = len(self.psnr_interp)
        for i in range(num):
            self.psnr_interp[i].reset()
            self.ssim_interp[i].reset()

    def compute_current_psnr_ssim(self, save=False, name=None, save_path=None):

        """
             compute ssim, psnr when validate the model
        """
        num = self.get_info()
        visuals = self.get_current_visuals()

        psnr_interp_t_t = []
        ssim_interp_t_t = []
        for i in range(num):
            rlt_img = util.tensor2img(visuals['rlt'][i])
            gt_img = util.tensor2img(visuals['GT'][i])
            psnr = util.calculate_psnr(rlt_img, gt_img)
            ssim = util.calculate_ssim(rlt_img, gt_img)

            psnr_interp_t_t.append(psnr)
            ssim_interp_t_t.append(ssim)

            if save == True:
                import os.path as osp
                import cv2
                cv2.imwrite(osp.join(save_path, 'rlt_{}_{}.png'.format(name, i)), rlt_img)
                cv2.imwrite(osp.join(save_path, 'gt_{}_{}.png'.format(name, i)), gt_img)

        return psnr_interp_t_t, ssim_interp_t_t

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

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

