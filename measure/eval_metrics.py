# -*- coding:utf-8 -*-
# @Time: 2023-9-7 16:39
# @Author: TonyWang-SEU (tongwangnj@qq.com)
# @File: eval_metrics.py
# @ProjectName: PolypNet

import os
import sys
import cv2
from tqdm import tqdm
import measure.metric as metric
import json
import argparse
import numpy as np


def Borders_Capture(gt, pred, dksize=15):
    gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = gt.copy()
    img[:] = 0
    cv2.drawContours(img, contours, -1, (255, 255, 255), 3)
    kernel = np.ones((dksize, dksize), np.uint8)
    img_dilate = cv2.dilate(img, kernel)

    res = cv2.bitwise_and(img_dilate, gt)
    b, g, r = cv2.split(res)
    alpha = np.rollaxis(img_dilate, 2, 0)[0]
    merge = cv2.merge((b, g, r, alpha))

    resp = cv2.bitwise_and(img_dilate, pred)
    b, g, r = cv2.split(resp)
    alpha = np.rollaxis(img_dilate, 2, 0)[0]
    mergep = cv2.merge((b, g, r, alpha))

    merge = cv2.cvtColor(merge, cv2.COLOR_RGB2GRAY)
    mergep = cv2.cvtColor(mergep, cv2.COLOR_RGB2GRAY)
    return merge, mergep, np.sum(img_dilate) / 255


def eval(args, dataset):
    # args = parser.parse_args()
    # model = args.model
    gt_root = args.GT_root
    pred_root = args.pred_root

    gt_root = os.path.join(gt_root, dataset)
    # ---- cod mask ----
    # gt_root = os.path.join(gt_root, 'GT')
    # ---- polyp mask ----
    gt_root = os.path.join(gt_root, 'masks')
    pred_root = os.path.join(pred_root, dataset)

    gt_name_list = sorted(os.listdir(pred_root))

    # ---- 实例化对象 ----
    FM = metric.Fmeasure_and_FNR()
    WFM = metric.WeightedFmeasure()
    SM = metric.Smeasure()
    EM = metric.Emeasure()
    MAE = metric.MAE()
    # ---- 增加dice和iou指标
    DICE = metric.DICE()
    IOU = metric.IoU()
    # Medical = metric.Medical(length=len(gt_name_list))

    BR_MAE = metric.MAE()
    BR_wF = metric.WeightedFmeasure()

    idx = 0
    for gt_name in tqdm(gt_name_list, total=len(gt_name_list)):
        # ---- 标签和预测结果的路径 ----
        gt_path = os.path.join(gt_root, gt_name)
        pred_path = os.path.join(pred_root, gt_name)

        # ---- 读标签 ----
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        # gt_width, gt_height = gt.shape

        # ---- 读预测结果 ----
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        # pred_width, pred_height = pred.shape
        if gt.shape != pred.shape:
            cv2.imwrite(os.path.join(pred_root, gt_name), cv2.resize(pred, gt.shape[::-1]))
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        # ---- 每个样本算一次指标 ----
        FM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        MAE.step(pred=pred, gt=gt)
        DICE.step(pred=pred, gt=gt)
        IOU.step(pred=pred, gt=gt)

        # Medical.step(pred=pred, gt=gt, idx=idx)
        # idx = +1

        if args.BR == 'on':
            BR_gt, BR_pred, area = Borders_Capture(cv2.imread(gt_path), cv2.imread(pred_path), int(args.br_rate))
            BR_MAE.step(pred=BR_pred, gt=BR_gt, area=area)
            BR_wF.step(pred=BR_pred, gt=BR_gt)

    # ---- 当前数据集的指标结果 ----
    fm = FM.get_results()[0]['fm']
    wfm = WFM.get_results()['wfm']
    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']
    fnr = FM.get_results()[1]
    # ---- cal dice and iou ----
    dice = DICE.get_results()['dice']
    iou = IOU.get_results()['iou']
    # mean_dice = Medical.get_results()['meanDice']
    # max_dice = Medical.get_results()['maxDice']
    # mean_iou = Medical.get_results()['meanIoU']
    # max_iou = Medical.get_results()['maxIoU']

    # ---- 保留精度 ----
    model_r = str(args.model)
    Smeasure_r = str(sm.round(4))
    Wmeasure_r = str(wfm.round(4))
    MAE_r = str(mae.round(4))
    adpEm_r = str(em['adp'].round(4))
    meanEm_r = str('-' if em['curve'] is None else em['curve'].mean().round(4))
    maxEm_r = str('-' if em['curve'] is None else em['curve'].max().round(4))
    adpFm_r = str(fm['adp'].round(4))
    meanFm_r = str(fm['curve'].mean().round(4))
    maxFm_r = str(fm['curve'].max().round(4))
    fnr_r = str(fnr.round(4))
    # ----  dice and iou ----
    dice_r = str(dice.round(4))
    iou_r = str(iou.round(4))
    # mean_dice_r = str(mean_dice.round(4))
    # max_dice_r = str(max_dice.round(4))
    # mean_iou_r = str(mean_iou.round(4))
    # max_iou_r = str(max_iou.round(4))

    if args.BR == 'on':
        BRmae = BR_MAE.get_results()['mae']
        BRmae_r = str(BRmae.round(3))
        BRwF = BR_wF.get_results()['wfm']
        BRwF_r = str(BRwF.round(3))
        eval_record = str(
            'Model:' + model_r + ',' +
            'Dataset:' + dataset + '||' +
            'Smeasure:' + Smeasure_r + '; ' +
            'meanEm:' + meanEm_r + '; ' +
            'wFmeasure:' + Wmeasure_r + '; ' +
            'MAE:' + MAE_r + '; ' +
            'fnr:' + fnr_r + ';' +
            'adpEm:' + adpEm_r + '; ' +
            # 'meanEm:' + meanEm_r + '; ' +
            'maxEm:' + maxEm_r + '; ' +
            'adpFm:' + adpFm_r + '; ' +
            'meanFm:' + meanFm_r + '; ' +
            'maxFm:' + maxFm_r + ';' +
            'BR' + str(args.br_rate) + '_mae:' + BRmae_r + ';' +
            'BR' + str(args.br_rate) + '_wF:' + BRwF_r
        )
    else:
        eval_record = str(
            # 'Model:' + model_r + ',' +
            'Dataset:' + dataset + '||' +
            # ---- dice and iou ----
            'DICE:' + dice_r + '; ' +
            'IOU:' + iou_r + '; ' +
            # 'mean_dice:' + mean_dice_r + '; ' +
            # 'max_dice:' + max_dice_r + '; ' +
            # 'mean_iou:' + mean_iou_r + '; ' +
            # 'max_iou:' + max_iou_r + '; ' +

            'Smeasure:' + Smeasure_r + '; ' +
            'meanEm:' + meanEm_r + '; ' +
            'wFmeasure:' + Wmeasure_r + '; ' +
            'MAE:' + MAE_r + '; ' +
            'fnr:' + fnr_r + ';' +
            'adpEm:' + adpEm_r + '; ' +
            # 'meanEm:' + meanEm_r + '; ' +
            'maxEm:' + maxEm_r + '; ' +
            'adpFm:' + adpFm_r + '; ' +
            'meanFm:' + meanFm_r + '; ' +
            'maxFm:' + maxFm_r
        )

    print(eval_record)
    print('#' * 50)
    if args.record_path is not None:
        txt = args.record_path
    else:
        txt = 'output/eval_record.txt'
    f = open(txt, 'a')
    f.write(eval_record)
    f.write("\n\n")
    f.close()


def call_eval_metric(args, dataset_list):
    datasets = dataset_list
    existed_pred = os.listdir(args.pred_root)
    for dataset in datasets:
        if dataset in existed_pred:
            eval(args, dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='PolypNet')
    parser.add_argument("--pred_root",
                        default=
                        r'E:\model\PolypSeg-Result\CFANet')
    parser.add_argument("--GT_root",
                        default=r'E:\data\Polyp-Dataset\PolypPVT-dataset\TestDataset')
    parser.add_argument("--record_path",
                        default=
                        r'E:\model\PolypSeg-Result\CFANet\eval_record_soft_label.txt')
    parser.add_argument("--BR", default='off')
    parser.add_argument("--br_rate", default=15)
    args = parser.parse_args()
    # datasets = ['NC4K', 'COD10K', 'CAMO', 'CHAMELEON']
    # datasets = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    datasets = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    # datasets = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB']
    # datasets = ['ETIS-LaribPolypDB']
    # datasets = ['CVC-300']
    existed_pred = os.listdir(args.pred_root)
    for dataset in datasets:
        if dataset in existed_pred:
            eval(args, dataset)
