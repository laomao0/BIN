import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler
from data.balancedsampler import RandomBalancedSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import time


def init_dist(backend='nccl', **kwargs):
    """initialization for distributed training"""
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])

    num_gpus = torch.cuda.device_count()

    print("Num GPUs ", num_gpus)
    print("RANK ", rank)

    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)



def main():

    ############################################
    #
    #           set options
    #
    ############################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    ############################################
    #
    #           distributed training settings
    #
    ############################################


    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        print("Rank:", rank)
        print("World Size", world_size)
        print("------------------DIST-------------------------")


    ############################################
    #
    #           loading resume state if exists
    #
    ############################################


    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None


    ############################################
    #
    #           mkdir and loggers
    #
    ############################################
    if 'debug' in opt['name']:
        debug_mode = True
    else:
        debug_mode = False


    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(opt['path']['experiments_root'])  # rename experiment folder if exists

            util.mkdirs((path for key, path in opt['path'].items() if
                         not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)

        util.setup_logger('base_val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)

        logger = logging.getLogger('base')
        logger_val = logging.getLogger('base_val')

        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
         # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_', level=logging.INFO, screen=True)
        print("set train log")
        util.setup_logger('base_val', opt['path']['log'], 'val_', level=logging.INFO, screen=True)
        print("set val log")
        logger = logging.getLogger('base')
        logger_val = logging.getLogger('base_val')


    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


    ############################################
    #
    #           create train and val dataloader
    #
    ############################################
    ####


    # dataset_ratio = 200  # enlarge the size of each epoch
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            if opt['datasets']['train'].get('split', None):
                train_set, val_set = create_dataset(dataset_opt)
            else:
                train_set = create_dataset(dataset_opt)   
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            # total_iters = int(opt['train']['niter'])
            # total_epochs = int(math.ceil(total_iters / train_size))
            total_iters = train_size
            total_epochs = int(opt['train']['epoch'])
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                # total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
                total_epochs = int(opt['train']['epoch'])
                if opt['train']['enable'] == False:
                    total_epochs = 1
            else:
                # train_sampler = None
                train_sampler = RandomBalancedSampler(train_set, train_size)
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler,vscode_debug=debug_mode)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            if not opt['datasets']['train'].get('split', None):
                val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None,vscode_debug=debug_mode)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))

    assert train_loader is not None


    ############################################
    #
    #          create model
    #
    ############################################
    ####

    model = create_model(opt)

    print("Model Created! ")

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0
        print("Not Resume Training")

    ############################################
    #
    #          training
    #
    ############################################


    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    model.train_AverageMeter()
    saved_total_loss = 10e10
    saved_total_PSNR = -1
    saved_total_SSIM = -1

    for epoch in range(start_epoch, total_epochs):

        ############################################
        #
        #          Start a new epoch
        #
        ############################################

        current_step = 0

        if opt['dist']:
            train_sampler.set_epoch(epoch)


        for train_idx, train_data in enumerate(train_loader):

            # print('current_step', current_step)

            if 'debug' in opt['name']:
                img_dir = os.path.join(opt['path']['train_images'])
                util.mkdir(img_dir)

                LQs = train_data['LQs']  # B N C H W

                if not 'sr' in opt['name']:
                    GTenh = train_data['GTenh']
                    GTinp = train_data['GTinp']

                    for imgs, name in zip([LQs, GTenh, GTinp], ['LQs', 'GTenh', 'GTinp']):
                        num = imgs.size(1)
                        for i in range(num):
                            img = util.tensor2img(imgs[0, i, ...])  # uint8
                            save_img_path = os.path.join(img_dir, '{:4d}_{:s}_{:1d}.png'.format(train_idx, str(name), i))
                            util.save_img(img, save_img_path)
                else:
                    if 'GT' in train_data:
                        GT_name = 'GT'
                    elif  'GTs' in train_data:
                        GT_name = 'GTs'

                    GT = train_data[GT_name]
                    for imgs, name in zip([LQs, GT], ['LQs', GT_name]):
                        if name == 'GT' :
                            num = imgs.size(0)
                            img = util.tensor2img(imgs[0, ...])  # uint8
                            save_img_path = os.path.join(img_dir, '{:4d}_{:s}_{:1d}.png'.format(train_idx, str(name), 0))
                            util.save_img(img, save_img_path)
                        elif name == 'GTs':
                            num = imgs.size(1)
                            for i in range(num):
                                img = util.tensor2img(imgs[:, i, ...])  # uint8
                                save_img_path = os.path.join(img_dir, '{:4d}_{:s}_{:1d}.png'.format(train_idx, str(name), i))
                                util.save_img(img, save_img_path)
                        else:
                            num = imgs.size(1)
                            for i in range(num):
                                img = util.tensor2img(imgs[:, i, ...])  # uint8
                                save_img_path = os.path.join(img_dir, '{:4d}_{:s}_{:1d}.png'.format(train_idx, str(name), i))
                                util.save_img(img, save_img_path)

                if (train_idx >= 3): # set to 0, just do validation
                    break

            # if pre-load weight first do validation and skip the first epoch
            # if opt['path'].get('pretrain_model_G', None) and epoch == 0:
            #     epoch += 1
            #     break

            if opt['train']['enable'] == False:
                message_train_loss = 'None'
                break

            current_step += 1
            if current_step > total_iters:
                print("Total Iteration Reached !")
                break

            #### update learning rate
            if opt['train']['lr_scheme'] == 'ReduceLROnPlateau':
                pass
            else:
                model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])


            #### training
            model.feed_data(train_data)

            model.optimize_parameters(current_step)

            model.train_AverageMeter_update()

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs_inst, logs_avg = model.get_current_log()  # training loss  mode='train'
                message = '[epoch:{:3d}, iter:{:8,d}, lr:('.format(epoch, current_step)
                for v in model.get_current_learning_rate():
                    message += '{:.3e},'.format(v)
                message += ')] '
                # if 'debug' in opt['name']:  # debug model print the instant loss
                #     for k, v in logs_inst.items():
                #         message += '{:s}: {:.4e} '.format(k, v)
                #         # tensorboard logger
                #         if opt['use_tb_logger'] and 'debug' not in opt['name']:
                #             if rank <= 0:
                #                 tb_logger.add_scalar(k, v, current_step)
                # for avg loss
                current_iters_epoch = epoch * total_iters + current_step
                for k, v in logs_avg.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_iters_epoch)
                if rank <= 0:
                    logger.info(message)

        # saving models
        if epoch == 1:
            save_filename = '{:04d}_{}.pth'.format(0, 'G')
            save_path = os.path.join(opt['path']['models'], save_filename)
            if os.path.exists(save_path):
                os.remove(save_path)

        save_filename = '{:04d}_{}.pth'.format(epoch-1, 'G')
        save_path = os.path.join(opt['path']['models'], save_filename)
        if os.path.exists(save_path):
            os.remove(save_path)

        if rank <= 0:
            logger.info('Saving models and training states.')
            save_filename = '{:04d}'.format(epoch)
            model.save(save_filename)


        # ======================================================================= #
        #                  Main validation loop                                   #
        # ======================================================================= #

        if opt['datasets'].get('val', None):
            if opt['dist']: 
                # multi-GPU testing
                psnr_rlt = {}  # with border and center frames
                psnr_rlt_avg = {}
                psnr_total_avg = 0.

                ssim_rlt = {}  # with border and center frames
                ssim_rlt_avg = {}
                ssim_total_avg = 0.

                val_loss_rlt = {}  # the averaged loss
                val_loss_rlt_avg ={}
                val_loss_total_avg = 0.

                if rank == 0:
                    pbar = util.ProgressBar(len(val_set))

                for idx in range(rank, len(val_set), world_size):  # distributed parallel validation
                    # print('idx', idx)

                    if 'debug' in opt['name']:
                        if (idx >= 3):
                            break

                    if (idx >= 1000):
                        break
                    val_data = val_set[idx]
                    # use idx method to fetch must extend batch dimension
                    val_data['LQs'].unsqueeze_(0)
                    val_data['GTenh'].unsqueeze_(0)
                    val_data['GTinp'].unsqueeze_(0)

                    key = val_data['key'][0]  # IMG_0034_00809
                    max_idx = len(val_set)
                    val_name = 'val_set'
                    num = model.get_info() # each model has different number of loss 

                    if psnr_rlt.get(val_name, None) is None:
                        psnr_rlt[val_name] = torch.zeros([num, max_idx], dtype=torch.float32, device='cuda')

                    if ssim_rlt.get(val_name, None) is None:
                        ssim_rlt[val_name] = torch.zeros([num, max_idx], dtype=torch.float32, device='cuda')

                    if val_loss_rlt.get(val_name, None) is None:
                        val_loss_rlt[val_name] = torch.zeros([num, max_idx], dtype=torch.float32, device='cuda')

                    model.feed_data(val_data)

                    model.test()

                    avg_loss, loss_list = model.get_loss(ret=1)

                    save_enable = True
                    if idx >= 100: 
                        save_enable = False

                    psnr_list, ssim_list = model.compute_current_psnr_ssim(save=save_enable, name=key, save_path=opt['path']['val_images'])

                    # print('psnr_list',psnr_list)

                    assert len(loss_list) == num
                    assert len(psnr_list) == num

                    for i in range(num):
                        psnr_rlt[val_name][i, idx] = psnr_list[i]
                        ssim_rlt[val_name][i, idx] = ssim_list[i]
                        val_loss_rlt[val_name][i, idx] = loss_list[i]
                        # print('psnr_rlt[val_name][i, idx]',psnr_rlt[val_name][i, idx])
                        # print('ssim_rlt[val_name][i, idx]',ssim_rlt[val_name][i, idx])
                        # print('val_loss_rlt[val_name][i, idx] ',val_loss_rlt[val_name][i, idx] )


                    if rank == 0:
                        for _ in range(world_size):
                            pbar.update('Test {} - {}/{}'.format(key, idx, max_idx))

                # # collect data
                for _, v in psnr_rlt.items():
                    for i in v:
                        dist.reduce(i, 0)

                for _, v in ssim_rlt.items():
                    for i in v:
                        dist.reduce(i, 0)

                for _, v in val_loss_rlt.items():
                    for i in v:
                        dist.reduce(i, 0)

                dist.barrier()

                if rank == 0:
                    psnr_rlt_avg = {}
                    psnr_total_avg = 0.
                    for k, v in psnr_rlt.items():  # key, value
                        # print('k', k, 'v', v, 'v.shape', v.shape)
                        psnr_rlt_avg[k] = []
                        for i in range(num):
                            non_zero_idx = v[i,:].nonzero()
                            # logger.info('non_zero_idx {}'.format(non_zero_idx.shape)) # check 
                            matrix =  v[i,:][non_zero_idx]
                            # print('matrix', matrix)
                            value = torch.mean(matrix).cpu().item()
                            # print('value', value)
                            psnr_rlt_avg[k].append(value) 
                            psnr_total_avg += psnr_rlt_avg[k][i]
                    psnr_total_avg = psnr_total_avg / (len(psnr_rlt)*num)
                    log_p = '# Validation # Avg. PSNR: {:.2f},'.format(psnr_total_avg)
                    for k, v in psnr_rlt_avg.items():
                        for i, it in enumerate(v):
                            log_p += ' {}: {:.2f}'.format(i, it)
                    logger.info(log_p)
                    logger_val.info(log_p)

                    # ssim
                    ssim_rlt_avg = {}
                    ssim_total_avg = 0.
                    for k, v in ssim_rlt.items():
                        ssim_rlt_avg[k] = []
                        for i in range(num):
                            non_zero_idx = v[i,:].nonzero()
                            # print('non_zero_idx', non_zero_idx)
                            matrix =  v[i,:][non_zero_idx]
                            # print('matrix', matrix)
                            value = torch.mean(matrix).cpu().item()
                            # print('value', value)
                            ssim_rlt_avg[k].append(torch.mean(matrix).cpu().item()) 
                            ssim_total_avg += ssim_rlt_avg[k][i]
                    ssim_total_avg /= (len(ssim_rlt)*num)
                    log_s = '# Validation # Avg. SSIM: {:.2f},'.format(ssim_total_avg)
                    for k, v in ssim_rlt_avg.items():
                        for i,it in enumerate(v):
                            log_s += ' {}: {:.2f}'.format(i, it)
                    logger.info(log_s)
                    logger_val.info(log_s)

                    # added
                    val_loss_rlt_avg = {}
                    val_loss_total_avg = 0.
                    for k, v in val_loss_rlt.items():
                        # k, key, the folder name
                        # v, value, the torch matrix
                        val_loss_rlt_avg[k] = []  # loss0 - loss_N
                        for i in range(num):
                            non_zero_idx = v[i,:].nonzero()
                            # print('non_zero_idx', non_zero_idx)
                            matrix =  v[i,:][non_zero_idx]
                            # print('matrix', matrix)
                            value = torch.mean(matrix).cpu().item()
                            # print('value', value)
                            val_loss_rlt_avg[k].append(torch.mean(matrix).cpu().item())
                            val_loss_total_avg += val_loss_rlt_avg[k][i]
                    val_loss_total_avg /= (len(val_loss_rlt)*num)
                    log_l = '# Validation # Avg. Loss: {:.4e},'.format(val_loss_total_avg)
                    for k, v in val_loss_rlt_avg.items():
                        for i,it in enumerate(v):
                            log_l += ' {}: {:.4e}'.format(i, it)
                    logger.info(log_l)
                    logger_val.info(log_l)

                    message = ''
                    for v in model.get_current_learning_rate():
                        message += '{:.5e}'.format(v)

                    logger_val.info(
                        'Epoch {:02d}, LR {:s}, PSNR {:.4f}, SSIM {:.4f}, Val Loss {:.4e}'.format(
                            epoch, message, psnr_total_avg, ssim_total_avg, val_loss_total_avg
                        ))

            else:
                pbar = util.ProgressBar(len(val_loader))

                model.val_loss_AverageMeter()
                model.val_AverageMeter_para()

                for val_inx, val_data in enumerate(val_loader):

                    # if 'debug' in opt['name']:
                    #     if (val_inx >= 10):
                    #         break

                    save_enable = True
                    if val_inx >= 100: 
                        save_enable = False
                    if val_inx >= 100:
                        break

                    key = val_data['key'][0]

                    folder = key[:-6]
                    model.feed_data(val_data)

                    model.test()

                    avg_loss, loss_list = model.get_loss(ret=1)

                    model.val_loss_AverageMeter_update(loss_list,avg_loss)

                    psnr_list, ssim_list = model.compute_current_psnr_ssim(save=save_enable, name=key, save_path=opt['path']['val_images'])

                    model.val_AverageMeter_para_update(psnr_list, ssim_list)

                    if 'debug' in opt['name']:
                        msg_psnr = ''
                        msg_ssim = ''
                        for i, psnr in enumerate(psnr_list): 
                            msg_psnr += '{} :{:.02f} '.format(i,psnr)
                        for i,ssim in enumerate(ssim_list): 
                            msg_ssim += '{} :{:.02f} '.format(i,ssim)

                        logger.info('{}_{:02d} {}'.format(key, val_inx, msg_psnr))
                        logger.info('{}_{:02d} {}'.format(key, val_inx, msg_ssim))

                    pbar.update('Test {} - {}'.format(key, val_inx))


                # toal validation log

                lr = ''
                for v in model.get_current_learning_rate():
                    lr += '{:.5e}'.format(v)

                logs_avg, logs_psnr_avg, psnr_total_avg, ssim_total_avg, val_loss_total_avg = model.get_current_log(mode='val')

                msg_logs_avg = ''
                for k, v in logs_avg.items():
                    msg_logs_avg += '{:s}: {:.4e} '.format(k, v)

                logger_val.info('Val-Epoch {:02d}, LR {:s}, {:s}'.format(epoch, lr, msg_logs_avg))
                logger.info('Val-Epoch {:02d}, LR {:s}, {:s}'.format(epoch, lr, msg_logs_avg))

                msg_logs_psnr_avg = ''
                for k, v in logs_psnr_avg.items():
                    msg_logs_psnr_avg += '{:s}: {:.4e} '.format(k, v)

                logger_val.info('Val-Epoch {:02d}, LR {:s}, {:s}'.format(epoch, lr, msg_logs_psnr_avg))
                logger.info('Val-Epoch {:02d}, LR {:s}, {:s}'.format(epoch, lr, msg_logs_psnr_avg))


                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('val_psnr', psnr_total_avg, epoch)
                    tb_logger.add_scalar('val_loss', val_loss_total_avg, epoch)


        ############################################
        #
        #          end of validation, save model
        #
        ############################################
        #
        if rank <= 0:
            logger.info("Finished an epoch, Check and Save the model weights")
            # we check the validation loss instead of training loss. OK~
            if saved_total_loss >= val_loss_total_avg:
                saved_total_loss = val_loss_total_avg
                #torch.save(model.state_dict(), args.save_path + "/best" + ".pth")
                model.save('best')
                logger.info("Best Weights updated for decreased validation loss")
            else:
                logger.info("Weights Not updated for undecreased validation loss")
            if saved_total_PSNR <= psnr_total_avg:
                saved_total_PSNR = psnr_total_avg
                model.save('bestPSNR')
                logger.info("Best Weights updated for increased validation PSNR")

            else:
                logger.info("Weights Not updated for unincreased validation PSNR")


        ############################################
        #
        #          end of one epoch, schedule LR
        #
        ############################################

        model.train_AverageMeter_reset()

        # add scheduler  todo
        if opt['train']['lr_scheme'] == 'ReduceLROnPlateau':
            for scheduler in model.schedulers:
                # scheduler.step(val_loss_total_avg)
                scheduler.step(val_loss_total_avg)
    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('last')
        logger.info('End of training.')
        tb_logger.close()


if __name__ == '__main__':
    main()