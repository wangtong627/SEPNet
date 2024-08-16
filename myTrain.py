# -*- coding:utf-8 -*-
# @Time: 2023-11-1 21:58
# @Author: TonyWang-SEU (tongwangnj@qq.com)
# @File: myTrain.py
# @ProjectName: PolypNet
import os
# import logging
# import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
# from lib.base_model_for_train import *
# from lib.full_model_for_train import *
from lib.model_for_train import *
from utils.trainer_for_six_logits import *
from utils.utils import adjust_lr, adjust_lr_v2, adjust_lr_v3, adjust_lr_with_warmup, init_logger
from utils.dataloader import get_loader, test_dataset
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['AdamW', 'Adam', 'SGD'])
    parser.add_argument('--batch_size', type=int, default=3)  # 本地测试
    parser.add_argument('--train_size', type=int, default=352)
    parser.add_argument('--gradient_clip', type=float, default=0.5)
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay_epoch', type=int, default=50)
    parser.add_argument('--load_checkpoint_path', type=str, default=None)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--train_data_path', type=str, default=r'E:\data\Polyp-Dataset\PolypPVT-dataset\TrainDataset')
    parser.add_argument('--val_data_path', type=str, default=r'E:\data\Polyp-Dataset\PolypPVT-dataset\TestDataset\CVC-300')
    parser.add_argument('--train_model_save_path', type=str, default=r'./checkpoint/record_0')
    parser.add_argument('--train_model_save_epoch', type=int, default=50)
    parser.add_argument('--multi_scale_training', type=bool, default=False)
    # # ---- network settings ----
    # parser.add_argument('--net_sc_type', type=str, default='sc0', choices=['sc0', 'sc1', 'sc2', 'sc3'])
    # parser.add_argument('--net_um_type', type=str, default='um0', choices=['um0', 'um1'])
    # ---- 2023.10.12 ----
    parser.add_argument('--net_type', type=str, default='train_model_arch',
                        choices=[

                            'train_model_arch',  #

                        ])
    parser.add_argument('--net_backbone', type=str, default='pvt-v2-b2',
                        choices=['res2net50', 'resnet50', 'pvt-v2-b0', 'pvt-v2-b1', 'pvt-v2-b2', 'pvt-v2-b4', 'efficientnet-b1',
                                 'efficientnet-b4'])

    # ---- param for middle channel ----
    parser.add_argument('--net_planes', type=int, default=64)
    # ---- param for rf ----
    # parser.add_argument('--net_rf_shuffles', type=list, default=[4, 8])
    parser.add_argument('--net_pgs_reduction_scale', type=int, default=4)
    parser.add_argument('--net_pgs_se_channel_reduction', type=int, default=16)
    # ---- param for df ----
    parser.add_argument('--net_df_channel_reduction', type=list, default=[4, 4])
    # ---- param for sc ----
    parser.add_argument('--net_sc_middle_channle_scale', type=int, default=2)
    parser.add_argument('--net_sc_ksize', default=[3, 3, 3])
    parser.add_argument('--net_sc_hidden_scale', type=int, default=32)
    parser.add_argument('--net_sc_gas_ksize', type=int, default=1)
    parser.add_argument('--net_sc_gas_ksize_scale', default=[15, 15, 15])
    # ---- pretrained dict ----
    parser.add_argument('--net_pretrained_backbone_path', type=str, default=r'E:/model/backbone/pvt_v2_b2.pth')
    opt = parser.parse_args()

    # ---- set logger and start training ----
    os.makedirs(opt.train_model_save_path, exist_ok=True)
    logger = init_logger(save_path=opt.train_model_save_path)
    print('=' * 35, ' Start Training ', '=' * 35)
    logger.info('>>> Start Training')
    logger.info('>>> config:{}'.format(opt))

    # ---- set device ----
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    print('>>> Using Gpu: ' + opt.gpu_id)
    logger.info('>>> Using Gpu: ' + opt.gpu_id)
    cudnn.benchmark = True

    # ---- set seed ----
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # ---- build the net ----
    # ---- GN Slim ----
    if opt.net_type == 'train_model_arch':
        print('>>> Using Train Model')

        model = SEPNet_TrainModel(backbone_name=opt.net_backbone,
                                  pretrained_backbone_path=opt.net_pretrained_backbone_path,
                                  mid_channels=opt.net_planes,
                                  rf_reduction_scale=opt.net_pgs_reduction_scale,
                                  rf_se_channel_reduction=opt.net_pgs_se_channel_reduction,
                                  df_channel_reduction=opt.net_df_channel_reduction,
                                  sc_hidden_scale=opt.net_sc_hidden_scale,
                                  sc_gas_ksize=opt.net_sc_gas_ksize,
                                  sc_middle_channle_scale=opt.net_sc_middle_channle_scale,
                                  sc_ksize=opt.net_sc_ksize,
                                  sc_gas_ksize_scale=opt.net_sc_gas_ksize_scale,
                                  ).cuda()

    else:
        raise Exception('Model not set!')

    # ---- set optimizer ----
    if opt.optimizer == 'AdamW':
        my_optimizer = optim.AdamW(params=model.parameters(), lr=opt.lr, weight_decay=1e-4)
        print('>>> Using optimizer: AdamW')
        logger.info('>>> Using optimizer: AdamW')
    elif opt.optimizer == 'Adam':
        my_optimizer = optim.Adam(params=model.parameters(), lr=opt.lr, weight_decay=1e-4)
        print('>>> Using optimizer: Adam')
        logger.info('>>> Using optimizer: Adam')
    elif opt.optimizer == 'SGD':
        my_optimizer = optim.SGD(params=model.parameters(), lr=opt.lr, weight_decay=1e-4, momentum=0.9)
        print('>>> Using optimizer: SGD')
        logger.info('>>> Using optimizer: SGD')
    else:
        raise Exception('Optimizer not set!')
    # optim_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=my_optimizer, T_max=opt.epoch, eta_min=1e-5)  # change the learning rate

    # ---- set dataloader ----
    # train_image_root = '{}/Imgs/'.format(opt.train_data_path)
    # train_gt_root = '{}/GT/'.format(opt.train_data_path)
    train_image_root = '{}/images/'.format(opt.train_data_path)
    train_gt_root = '{}/masks/'.format(opt.train_data_path)
    train_loader = get_loader(img_root=train_image_root, gt_root=train_gt_root, batch_size=opt.batch_size,
                              train_size=opt.train_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=4)
    # val_image_root = '{}/Imgs/'.format(opt.val_data_path)
    # val_gt_root = '{}/GT/'.format(opt.val_data_path)
    val_image_root = '{}/images/'.format(opt.val_data_path)
    val_gt_root = '{}/masks/'.format(opt.val_data_path)
    val_loader = test_dataset(image_root=val_image_root, gt_root=val_gt_root, test_size=opt.train_size)

    # ---- whether load from checkpoint ----
    if opt.load_checkpoint_path is not None:
        model.load_state_dict(torch.load(opt.load_checkpoint_path)['model_state_dict'])
        my_optimizer.load_state_dict(torch.load(opt.load_checkpoint_path)['optimizer_state_dict'])
        start_epoch = torch.load(opt.load_checkpoint_path)['epoch']
        print('>>> Successfully load checkpoint model at Epoch: ', start_epoch)
        logger.info('>>> Successfully load checkpoint model at Epoch:{}'.format(start_epoch))  # revised log at 2023/8/16
    else:
        # ---- set start epoch and train from scratch----
        start_epoch = 1

    # ---- set writer ----
    from datetime import datetime

    log_dir = opt.train_model_save_path + '/summary_{}'.format(datetime.now().strftime("%Y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)

    # ---- Start training ----
    for epoch in range(start_epoch, opt.epoch + 1):
        adjust_lr_with_warmup(optimizer=my_optimizer, init_lr=opt.lr, epoch=epoch, decay_rate=opt.lr_decay_rate,
                              decay_epoch=opt.lr_decay_epoch,
                              min_lr=0.1 * opt.lr, warmup_epoch=30)
        train_stage(train_loader=train_loader, model=model, optimizer=my_optimizer, epoch=epoch, opt=opt, logger=logger, writer=writer)
        # schedule
        # adjust_lr_v3(optimizer=my_optimizer, init_lr=opt.lr, epoch=epoch, decay_rate=opt.lr_decay_rate, decay_epoch=opt.lr_decay_epoch,
        #              min_lr=0.1 * opt.lr)
        # optim_schedule.step()
        # if epoch == start_epoch or (epoch % 2 == 1 and epoch > 10) or epoch > 100:
        # if epoch == start_epoch or (epoch % 10 == 0) or epoch > 100:
        val_stage_with_multiMetrics(test_loader=val_loader, model=model, epoch=epoch, opt=opt, logger=logger, start_epoch=start_epoch,
                                    writer=writer)
    print('>>> Training finished!')
    logger.info('>>> Training finished!')
