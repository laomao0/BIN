"""create dataset and dataloader"""
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None, vscode_debug=False):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = False #True
        if vscode_debug == True:
            num_workers = 0
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=True) #False
    else:
        if vscode_debug == True:
            num_workers = 0
        else:
            num_workers = 1
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers,  # for debug 0, normal 1
                                           pin_memory=True) #False


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    # datasets for image restoration
    if mode == 'BIN':
        from data.BIN_dataset import BINDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset



__all__ = (
        'create_dataloader',
        'create_dataset'
        )

