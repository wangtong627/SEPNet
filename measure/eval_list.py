# -*- coding:utf-8 -*-
# @Time: 2024-1-24 14:37
# @Author: TonyWang-SEU (tongwangnj@qq.com)
# @File: eval_list.py
# @ProjectName: PolypNet
"""
实现对单个样本计算指标
"""
import argparse
import os
import cv2
from tqdm import tqdm
import measure.metric as metric
# import numpy as np
import pandas as pd


def my_eval(args, dataset):
    gt_root = os.path.join(args.GT_root, dataset, 'masks')
    pred_root = os.path.join(args.pred_root, dataset)
    gt_name_list = sorted(os.listdir(pred_root))

    # 存放容器
    # model_dataFrame = pd.DataFrame(columns=['Sample', 'Sm', 'Em', 'Fm', 'Sum'])
    model_dataFrame = pd.DataFrame(columns=['Sample', args.model])

    dataFrame_idx = 0
    for gt_name in tqdm(gt_name_list, total=len(gt_name_list)):

        # if gt_name != '153.png':
        #     continue

        # 获取单个样本名称
        current_sample_name = '[' + dataset + ']_' + gt_name

        gt_path = os.path.join(gt_root, gt_name)
        pred_path = os.path.join(pred_root, gt_name)

        # ---- 读标签 ----
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        # ---- 读预测结果 ----
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if gt.shape != pred.shape:
            cv2.imwrite(os.path.join(pred_root, gt_name), cv2.resize(pred, gt.shape[::-1]))
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        # 实例化指标
        # FM = metric.Fmeasure_and_FNR()
        # SM = metric.Smeasure()
        # EM = metric.Emeasure()
        # ---- 增加dice和iou指标
        DICE = metric.DICE()

        # ---- 每个样本算一次指标 ----
        # FM.step(pred=pred, gt=gt)
        # SM.step(pred=pred, gt=gt)
        # EM.step(pred=pred, gt=gt)
        DICE.step(pred=pred, gt=gt)

        # ---- 当前数据集的指标结果 ----
        # fm = FM.get_results()[0]['fm']
        # sm = SM.get_results()['sm']
        # em = EM.get_results()['em']
        dice = DICE.get_results()['dice']

        # Smeasure_r = sm.round(4)
        # adpFm_r = fm['adp'].round(4)
        # mEm_r = em['curve'].mean().round(4)
        dice_r = dice.round(4)
        # # 统计综合指标
        # score_sum = Smeasure_r + adpFm_r + mEm_r

        # 记录到容器
        data_row = [current_sample_name, dice_r]
        model_dataFrame.loc[dataFrame_idx] = data_row
        dataFrame_idx += 1

    # 导出为CSV文件
    csv_filename = 'Dice_{}_{}.csv'.format(args.model, dataset)
    model_dataFrame.to_csv(os.path.join(args.pred_root, csv_filename), index=False)  # 如果不希望保存索引列，设置index=False
    print(f'已导出为CSV文件: {csv_filename}')

    # 导出为Excel文件
    excel_filename = 'Dice_{}_{}.xlsx'.format(args.model, dataset)
    model_dataFrame.to_excel(os.path.join(args.pred_root, excel_filename), index=False)
    print(f'已导出为Excel文件: {excel_filename}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='T315')
    parser.add_argument('--pred_root', default=r'E:\model\My-Model\PolypNet_Code\result\T315')
    parser.add_argument('--GT_root', default=r'E:\model\PolypSeg-Result\MyResults\Image')

    args = parser.parse_args()
    # datasets = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    datasets = ['CVC-ClinicDB', 'Kvasir']
    # datasets = ['CVC-300']
    existed_pred = os.listdir(args.pred_root)
    for dataset in datasets:
        if dataset in existed_pred:
            my_eval(args, dataset)
