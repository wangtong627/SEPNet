# -*- coding:utf-8 -*-
# @Time: 2023-10-12 20:53
# @Author: TonyWang-SEU (tongwangnj@qq.com)
# @File: myTest.py
# @ProjectName: PolypNet
import os
import cv2

import torch.backends.cudnn as cudnn

from utils.dataloader import test_dataset
# ---- models ----

from lib.model_for_train import *

from lib.model_for_eval import *

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--test_size', type=int, default=352)
    parser.add_argument('--model_state_dict_path', type=str,
                        default=
                        r'.\checkpoint\Weight_of_SEPNet.pth')
    parser.add_argument('--test_data_path', type=str,
                        default=
                        r'E:\data\Polyp-Dataset\TestDataset')
    parser.add_argument('--pred_save_path', type=str,
                        default=
                        r'.\result\Model_SEPNet')
    # ---- 2023.10.12 ----
    parser.add_argument('--net_type', type=str, default='eval_model_arch',
                        choices=[
                            # 'HYT_Model', 'QYL_Model',
                            'train_model_arch',
                            'eval_model_arch',

                        ])

    # ---- network params ----
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
    parser.add_argument('--net_pretrained_backbone_path', type=str, default=None)
    opt = parser.parse_args()

    # ---- set device ----
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    print('=' * 35, ' Start Testing ', '=' * 35)
    cudnn.benchmark = True
    # ---- set seed ----
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # ---- 2023.10.12 baseModel ----
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


    # ---- 2024-2-27 ablation ----

    elif opt.net_type == 'eval_model_arch':
        print('>>> Using Eval Model')
        model = SEPNet_EvalModel(backbone_name=opt.net_backbone,
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

    model_state_dict = torch.load(opt.model_state_dict_path, map_location='cuda')['model_state_dict']  # 取出模型的训练参数
    model.load_state_dict(model_state_dict, strict=False)
    print('>>> Successfully load Training Dict!')
    model.eval()

    # for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']:
    for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
        data_path = opt.test_data_path + '/{}/'.format(_data_name)
        save_path = opt.pred_save_path + '/{}/'.format(_data_name)
        os.makedirs(save_path, exist_ok=True)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        test_loader = test_dataset(image_root=image_root, gt_root=gt_root, test_size=opt.test_size)

        for i in range(test_loader.size):
            image, gt, name, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            y = model(image)
            pred = F.interpolate(y[0], size=gt.shape, mode='bilinear', align_corners=False)  # y[0] is p1
            # pred = F.interpolate(y, size=gt.shape, mode='bilinear', align_corners=False)  # y[0] is p1
            pred = pred.sigmoid().data.cpu().numpy().squeeze()
            pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            print('> {} - {}'.format(_data_name, name))

            # save pred map
            cv2.imwrite(save_path + name, pred * 255)
        print(_data_name, 'Finish!')

    # ---- eval metrics ----
    # from measure.eval_metrics import call_eval_metric

    # call_eval_metric()
