# -*- coding:utf-8 -*-
# @Time: 2023-11-1 21:47
# @Author: TonyWang-SEU (tongwangnj@qq.com)
# @File: trainer_for_six_logits.py
# @ProjectName: PolypNet
import torch
# import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from utils.loss_func import structure_loss, region_loss, GroupLoss
from utils.utils import clip_gradient
from datetime import datetime
import measure.metric as Measure


def train_stage(train_loader, model, optimizer, epoch, opt, logger, writer=None):
    model.train()
    print('=' * 35, 'Training Epoch: ' + str(epoch), '=' * 35)

    # ---- whether use opt.multi_scale_training ----
    if opt.multi_scale_training:
        # size_rate = [0.75, 1, 1.25]
        size_rate = [224/352, 256/352, 288/352, 320/352, 1]
    else:
        size_rate = [1]

    # ---- set epoch info ----
    epoch_loss = 0
    epoch_step = 0

    # ---- set model save path ----
    model_save_path = opt.train_model_save_path
    os.makedirs(model_save_path, exist_ok=True)

    # ---- revised full branch, return 8 logit, train the model by iter ----
    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            for rate in size_rate:
                optimizer.zero_grad()
                # ---- data prepare ----
                images = images.cuda()
                gts = gts.cuda()
                # ---- rescale ----
                train_size = int(round(opt.train_size * rate / 32) * 32)
                if rate != 1:
                    images = F.interpolate(images, size=(train_size, train_size), mode='bilinear', align_corners=True)
                logit1, logit2, logit3, logit4, deep_logit2, deep_logit3 = model(images)
                # ---- loss func ----
                loss_logit4 = structure_loss(pred=logit4,
                                             mask=F.interpolate(gts, size=logit4.shape[2:], mode='bilinear', align_corners=True))
                loss_logit3 = structure_loss(pred=logit3,
                                             mask=F.interpolate(gts, size=logit3.shape[2:], mode='bilinear', align_corners=True))
                loss_logit2 = structure_loss(pred=logit2,
                                             mask=F.interpolate(gts, size=logit2.shape[2:], mode='bilinear', align_corners=True))
                loss_logit1 = structure_loss(pred=logit1,
                                             mask=F.interpolate(gts, size=logit1.shape[2:], mode='bilinear', align_corners=True))

                loss_deep_logit3 = structure_loss(pred=deep_logit3,
                                                  mask=F.interpolate(gts, size=deep_logit3.shape[2:], mode='bilinear',
                                                                     align_corners=True))
                loss_deep_logit2 = structure_loss(pred=deep_logit2,
                                                  mask=F.interpolate(gts, size=deep_logit2.shape[2:], mode='bilinear',
                                                                     align_corners=True))
                # loss_deep_logit1 = structure_loss(pred=deep_logit1,
                #                                   mask=F.interpolate(gts, size=deep_logit1.shape[2:], mode='bilinear',
                #                                                      align_corners=True))
                batch_loss = loss_logit4 + loss_logit3 + loss_logit2 + 2 * loss_logit1 + loss_deep_logit3 + loss_deep_logit2
                # ---- backward ----
                batch_loss.backward()
                clip_gradient(optimizer=optimizer, grad_clip=opt.gradient_clip)
                optimizer.step()
                # ---- update epoch loss ----
                if rate == 1:
                    epoch_loss += batch_loss.data
                    epoch_step += 1

                    # ---- record train info ----
                    if epoch_step == 1 or epoch_step % 10 == 0 or epoch_step == len(train_loader):
                        print('{} [Epoch: {:03d}/{:03d}] => [Batch: {:04d}/{:04d}] => [BatchLoss: {:.4f}] => [UpdatedEpochLoss: {:.4f}]'
                              .format(datetime.now(), epoch, opt.epoch, epoch_step, len(train_loader), batch_loss,
                                      epoch_loss / epoch_step))
                        print(
                            '=> Shallow: [loss_logit4: {:.4f}] => [loss_logit3: {:.4f}] => [loss_logit2: {:.4f}] => [loss_logit1: {:.4f}], [Lr: {}]'
                            .format(loss_logit4.data, loss_logit3.data, loss_logit2.data, loss_logit1.data,
                                    optimizer.state_dict()['param_groups'][0]['lr']))
                        print(
                            '=> Deep: [loss_deep_logit3: {:.4f}] => [loss_deep_logit2: {:.4f}]'
                            .format(loss_deep_logit3.data, loss_deep_logit2.data))
    except KeyboardInterrupt:
        # ---- 某个iter被终止 存中断iter的checkpoint ----
        print('>>> KeyBoard Interrupt: save model and exit!')
        logger.info('>>> KeyBoard Interrupt: save model and exit!')
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 'loss': epoch_loss / epoch_step # loss 可以不存，节省空间
                    }, model_save_path + '/interrupted_checkpoint_epoch_' + str(epoch) + '.pth')
        print('>>> Successfully save interrupted checkpoint at epoch ' + str(epoch) + '.')
        logger.info('>>> Successfully save interrupted checkpoint at epoch ' + str(epoch) + '.')
        raise

    # ---- print/save the info of model in each epoch ----
    print('[Train Info]: [Epoch {:03d}/{:03d}], [AvgEpochLoss: {:.4f}], [Lr: {}]'
          .format(epoch, opt.epoch, epoch_loss / epoch_step, optimizer.state_dict()['param_groups'][0]['lr']))
    logger.info('[Train Info]: [Epoch {:03d}/{:03d}], [AvgEpochLoss: {:.4f}], [Lr: {}]'
                .format(epoch, opt.epoch, epoch_loss / epoch_step, optimizer.state_dict()['param_groups'][0]['lr']))

    # ---- add tb record ----
    writer.add_scalar('Train/learning_rate', optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
    writer.add_scalar('Train/Loss', epoch_loss / epoch_step, global_step=epoch)

    # ---- save model ----
    if epoch % opt.train_model_save_epoch == 0:
        torch.save({'epoch': epoch,
                    # 'best_mae': best_mae,
                    # 'best_epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss / epoch_step}, model_save_path + '/checkpoint_epoch_' + str(epoch) + '.pth')
        print('>>> Saving checkpoint at epoch ' + str(epoch) + '.')


# def val_stage(test_loader, model, epoch, opt, logger, start_epoch, writer=None):
#     model.eval()
#     print('=' * 35, 'Val Epoch: ' + str(epoch), '=' * 35)
#
#     # ---- set save path ----
#     model_save_path = opt.train_model_save_path  # 用于保存best model
#     os.makedirs(model_save_path, exist_ok=True)
#
#     # ---- switch to full branch ----
#     global best_mae_shallow, best_epoch_shallow, best_mae_deep, best_epoch_deep
#
#     with torch.no_grad():
#         mae_sum_shallow = 0
#         mae_sum_deep = 0
#
#         # ---- get the evaluation of the performance ----
#         for i in range(test_loader.size):
#             image, gt, name, img_for_post = test_loader.load_data()
#             gt = np.asarray(gt, np.float32)
#             gt /= (gt.max() + 1e-8)  # make gt scatter into 0~1
#             image = image.cuda()
#             logit1, logit2, logit3, logit4, deep_logit1, deep_logit2, deep_logit3 = model(image)
#             # ---- get shallow pred ----
#             pred_shallow = F.interpolate(logit1, size=gt.shape, mode='bilinear', align_corners=False)
#             pred_shallow = pred_shallow.sigmoid().data.cpu().numpy().squeeze()
#             pred_shallow = (pred_shallow - pred_shallow.min()) / (pred_shallow.max() - pred_shallow.min() + 1e-8)
#             # ---- get deep pred ----
#             pred_deep = F.interpolate(deep_logit1, size=gt.shape, mode='bilinear', align_corners=False)
#             pred_deep = pred_deep.sigmoid().data.cpu().numpy().squeeze()
#             pred_deep = (pred_deep - pred_deep.min()) / (pred_deep.max() - pred_deep.min() + 1e-8)
#             # metric mae
#             mae_sum_shallow += np.sum(np.abs(pred_shallow - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
#             mae_sum_deep += np.sum(np.abs(pred_deep - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
#         mae_shallow = mae_sum_shallow / test_loader.size
#         mae_deep = mae_sum_deep / test_loader.size
#
#         # ---- update the performance and save the best model ----
#         if epoch == start_epoch:  # record the mae of start epoch
#             best_mae_shallow = mae_shallow
#             best_mae_deep = mae_deep
#             best_epoch_shallow = epoch
#             best_epoch_deep = epoch
#         # ---- save model with best shallow mae ----
#         if mae_shallow < best_mae_shallow:
#             best_mae_shallow = mae_shallow
#             best_epoch_shallow = epoch
#             if start_epoch == 1:
#                 torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
#                            model_save_path + '/bestShallowMae_Model_train_from_scratch.pth')
#             else:
#                 torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
#                            model_save_path + '/bestShallowMae_Model_train_from_checkpoint.pth')
#             print('>>> Save best Shallow_MAE model at epoch: {}.'.format(epoch))
#         # ---- save model with best deep mae ----
#         if mae_deep < best_mae_deep:
#             best_mae_deep = mae_deep
#             best_epoch_deep = epoch
#             if start_epoch == 1:
#                 torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
#                            model_save_path + '/bestDeepMae_Model_train_from_scratch.pth')
#             else:
#                 torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
#                            model_save_path + '/bestDeepMae_Model_train_from_checkpoint.pth')
#             print('>>> Save best Deep_MAE model at epoch: {}.'.format(epoch))
#
#         # ---- print/log the performance ----
#         if start_epoch == 1:
#             # ---- shallow predict performance ----
#             print('[Val] [Epoch: {}] => [ShallowMAE: {:.4f}] => [bestShallowMAE_f_scratch: {:.4f}] => [bestShallowEpoch_f_scratch: {}]'
#                   .format(epoch, float(mae_shallow), float(best_mae_shallow), best_epoch_shallow))
#             logger.info('[Val Info]: Epoch: {}, ShallowMAE: {:.4f}, bestShallowMAE_f_scratch: {:.4f}, bestShallowEpoch_f_scratch: {}.'
#                         .format(epoch, float(mae_shallow), float(best_mae_shallow), best_epoch_shallow))
#             # ---- deep predict performance ----
#             print('[Val] [Epoch: {}] => [DeepMAE: {:.4f}] => [bestDeepMAE_f_scratch: {:.4f}] => [bestDeepEpoch_f_scratch: {}]'
#                   .format(epoch, float(mae_deep), float(best_mae_deep), best_epoch_deep))
#             logger.info('[Val Info]: Epoch: {}, DeepMAE: {:.4f}, bestDeepMAE_f_scratch: {:.4f}, bestDeepEpoch_f_scratch: {}.'
#                         .format(epoch, float(mae_deep), float(best_mae_deep), best_epoch_deep))
#         else:
#             # ---- shallow predict performance ----
#             print('[Val] [Epoch: {}] => [ShallowMAE: {:.4f}] => [bestShallowMAE_from_ckpt: {:.4f}] => [bestShallowEpoch_from_ckpt: {}]'
#                   .format(epoch, float(mae_shallow), float(best_mae_shallow), best_epoch_shallow))
#             logger.info('[Val Info]: Epoch: {}, ShallowMAE: {:.4f}, bestShallowMAE_from_ckpt: {:.4f}, bestShallowEpoch_from_ckpt: {}.'
#                         .format(epoch, float(mae_shallow), float(best_mae_shallow), best_epoch_shallow))
#             # ---- deep predict performance ----
#             print('[Val] [Epoch: {}] => [DeepMAE: {:.4f}] => [bestDeepMAE_from_ckpt: {:.4f}] => [bestDeepEpoch_from_ckpt: {}]'
#                   .format(epoch, float(mae_deep), float(best_mae_deep), best_epoch_deep))
#             logger.info('[Val Info]: Epoch: {}, DeepMAE: {:.4f}, bestDeepMAE_from_ckpt: {:.4f}, bestDeepEpoch_from_ckpt: {}.'
#                         .format(epoch, float(mae_deep), float(best_mae_deep), best_epoch_deep))


def val_stage_with_multiMetrics(test_loader, model, epoch, opt, logger, start_epoch, writer=None):
    model.eval()
    print('=' * 35, 'Val Epoch: ' + str(epoch), '=' * 35)

    # ---- set save path ----
    model_save_path = opt.train_model_save_path  # 用于保存best model
    os.makedirs(model_save_path, exist_ok=True)

    # ---- switch to full branch ----
    global best_shallow_metric_dict, best_shallow_score, best_epoch_shallow

    # ---- shallow ----
    shallowDice = Measure.DICE()
    shallowIOU = Measure.IoU()
    shallowMAE = Measure.MAE()
    shallow_metric_dict = dict()
    # # ---- deep ----
    # deepDice = Measure.DICE()
    # deepIOU = Measure.IoU()
    # deepMAE = Measure.MAE()
    # deep_metric_dict = dict()

    with torch.no_grad():

        # ---- get the evaluation of the performance ----
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            image = image.cuda()
            logit1, logit2, logit3, logit4, deep_logit2, deep_logit3 = model(image)

            # ---- get shallow pred ----
            pred_shallow = F.interpolate(logit1, size=gt.shape, mode='bilinear', align_corners=False)
            pred_shallow = pred_shallow.sigmoid().data.cpu().numpy().squeeze()
            pred_shallow = (pred_shallow - pred_shallow.min()) / (pred_shallow.max() - pred_shallow.min() + 1e-8) * 255  # *255还原到压缩前

            # # ---- get deep pred ----
            # pred_deep = F.interpolate(deep_logit1, size=gt.shape, mode='bilinear', align_corners=False)
            # pred_deep = pred_deep.sigmoid().data.cpu().numpy().squeeze()
            # pred_deep = (pred_deep - pred_deep.min()) / (pred_deep.max() - pred_deep.min() + 1e-8) * 255  # 同上

            # ---- 计算单个样本的指标 ----
            shallowDice.step(pred=pred_shallow, gt=gt)
            shallowIOU.step(pred=pred_shallow, gt=gt)
            shallowMAE.step(pred=pred_shallow, gt=gt)
            # deepDice.step(pred=pred_deep, gt=gt)
            # deepIOU.step(pred=pred_deep, gt=gt)
            # deepMAE.step(pred=pred_deep, gt=gt)

        # ---- 计算整个loader中样本的指标 ----
        shallow_metric_dict.update(dice=shallowDice.get_results()['dice'])
        shallow_metric_dict.update(iou=shallowIOU.get_results()['iou'])
        shallow_metric_dict.update(mae=shallowMAE.get_results()['mae'])
        # deep_metric_dict.update(dice=deepDice.get_results()['dice'])
        # deep_metric_dict.update(iou=deepIOU.get_results()['iou'])
        # deep_metric_dict.update(mae=deepMAE.get_results()['mae'])

        cur_shallow_score = shallow_metric_dict['dice'] + shallow_metric_dict['iou'] - shallow_metric_dict['mae']
        # cur_deep_score = deep_metric_dict['dice'] + deep_metric_dict['iou'] - deep_metric_dict['mae']

        # ---- update the performance and save the best model ----
        if epoch == start_epoch:  # record the mae of start epoch
            best_shallow_metric_dict = shallow_metric_dict
            # best_deep_metric_dict = deep_metric_dict
            best_shallow_score = cur_shallow_score
            # best_deep_score = cur_deep_score
            best_epoch_shallow = epoch
            # best_epoch_deep = epoch

        # ---- save model with best shallow score ----
        if cur_shallow_score > best_shallow_score:
            best_shallow_metric_dict = shallow_metric_dict
            best_shallow_score = cur_shallow_score
            best_epoch_shallow = epoch
            if start_epoch == 1:
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
                           model_save_path + '/bestShallowModel_train_from_scratch.pth')
            else:
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
                           model_save_path + '/bestShallowModel_train_from_checkpoint.pth')
            print('>>> Save best SHALLOW model at epoch: {}.'.format(epoch))

        # # ---- save model with best deep score ----
        # if cur_deep_score > best_deep_score:
        #     best_deep_metric_dict = deep_metric_dict
        #     best_deep_score = cur_deep_score
        #     best_epoch_deep = epoch
        #     if start_epoch == 1:
        #         torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
        #                    model_save_path + '/bestDeepModel_train_from_scratch.pth')
        #     else:
        #         torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
        #                    model_save_path + '/bestDeepModel_train_from_checkpoint.pth')
        #     print('>>> Save best DEEP model at epoch: {}.'.format(epoch))

        # ---- get value ----
        cur_shallow_dice = shallow_metric_dict['dice'].round(4)
        cur_shallow_iou = shallow_metric_dict['iou'].round(4)
        cur_shallow_mae = shallow_metric_dict['mae'].round(4)
        # cur_deep_dice = deep_metric_dict['dice'].round(4)
        # cur_deep_iou = deep_metric_dict['iou'].round(4)
        # cur_deep_mae = deep_metric_dict['mae'].round(4)
        best_shallow_dice = best_shallow_metric_dict['dice'].round(4)
        best_shallow_iou = best_shallow_metric_dict['iou'].round(4)
        best_shallow_mae = best_shallow_metric_dict['mae'].round(4)
        # best_deep_dice = best_deep_metric_dict['dice'].round(4)
        # best_deep_iou = best_deep_metric_dict['iou'].round(4)
        # best_deep_mae = best_deep_metric_dict['mae'].round(4)

        # TensorboardX-Loss
        writer.add_scalars('Val/Shallow_DICE_IOU',
                           {'cur_shallow_dice': cur_shallow_dice, 'cur_shallow_iou': cur_shallow_iou},
                           global_step=epoch)
        writer.add_scalar('Val/Shallow_MAE', cur_shallow_mae, global_step=epoch)
        writer.add_scalars('Val/BestShallow_DICE_IOU',
                           {'best_shallow_dice': best_shallow_dice, 'best_shallow_iou': best_shallow_iou},
                           global_step=epoch)
        writer.add_scalar('Val/BestShallow_MAE', best_shallow_mae, global_step=epoch)

        # writer.add_scalars('Val/Deep_DICE_IOU',
        #                    {'cur_deep_dice': cur_deep_dice, 'cur_deep_iou': cur_deep_iou},
        #                    global_step=epoch)
        # writer.add_scalar('Val/Deep_MAE', cur_deep_mae, global_step=epoch)
        # writer.add_scalars('Val/BestDeep_DICE_IOU',
        #                    {'best_deep_dice': best_deep_dice, 'best_deep_iou': best_deep_iou},
        #                    global_step=epoch)
        # writer.add_scalar('Val/BestDeep_MAE', best_deep_mae, global_step=epoch)

        # ---- print/log the performance ----
        if start_epoch == 1:
            # ---- shallow predict performance ----
            print('[Val] [Epoch: {}] => [ShallowScore: {:.4f}] => [bestShallowScore_f_scratch: {:.4f}] => [bestShallowEpoch_f_scratch: {}]'
                  .format(epoch, float(cur_shallow_score), float(best_shallow_score), best_epoch_shallow))
            print('[CurShallowMetrics (dice={}, iou={}, mae={})] => [BestShallowMetrics (dice={}, iou={}, mae={})]'
                  .format(cur_shallow_dice, cur_shallow_iou, cur_shallow_mae, best_shallow_dice, best_shallow_iou, best_shallow_mae))
            logger.info('[Val Info]: Epoch: {}, ShallowScore: {:.4f}, bestShallowScore_f_scratch: {:.4f}, bestShallowEpoch_f_scratch: {}.'
                        .format(epoch, float(cur_shallow_score), float(best_shallow_score), best_epoch_shallow))
            logger.info('[CurShallowMetrics (dice={}, iou={}, mae={})] => [BestShallowMetrics (dice={}, iou={}, mae={})]'
                        .format(cur_shallow_dice, cur_shallow_iou, cur_shallow_mae, best_shallow_dice, best_shallow_iou, best_shallow_mae))

            # # ---- deep predict performance ----
            # print('[Val] [Epoch: {}] => [DeepScore: {:.4f}] => [bestDeepScore_f_scratch: {:.4f}] => [bestDeepEpoch_f_scratch: {}]'
            #       .format(epoch, float(cur_deep_score), float(best_deep_score), best_epoch_deep))
            # print('[CurDeepMetrics (dice={}, iou={}, mae={})] => [BestDeepMetrics (dice={}, iou={}, mae={})]'
            #       .format(cur_deep_dice, cur_deep_iou, cur_deep_mae, best_deep_dice, best_deep_iou, best_deep_mae))
            # logger.info('[Val Info]: Epoch: {}, ShallowScore: {:.4f}, bestShallowScore_f_scratch: {:.4f}, bestShallowEpoch_f_scratch: {}.'
            #             .format(epoch, float(cur_shallow_score), float(best_shallow_score), best_epoch_shallow))
            # logger.info('[CurDeepMetrics (dice={}, iou={}, mae={})] => [BestDeepMetrics (dice={}, iou={}, mae={})]'
            #             .format(cur_deep_dice, cur_deep_iou, cur_deep_mae, best_deep_dice, best_deep_iou, best_deep_mae))
        else:
            # ---- shallow predict performance ----
            print('[Val] [Epoch: {}] => [ShallowScore: {:.4f}] => [bestShallowScore_from_ckpt: {:.4f}] => [bestShallowEpoch_from_ckpt: {}]'
                  .format(epoch, float(cur_shallow_score), float(best_shallow_score), best_epoch_shallow))
            print('[CurShallowMetrics (dice={}, iou={}, mae={})] => [BestShallowMetrics (dice={}, iou={}, mae={})]'
                  .format(cur_shallow_dice, cur_shallow_iou, cur_shallow_mae, best_shallow_dice, best_shallow_iou, best_shallow_mae))
            logger.info('[Val Info]: Epoch: {}, ShallowScore: {:.4f}, bestShallowScore_from_ckpt: {:.4f}, bestShallowEpoch_from_ckpt: {}.'
                        .format(epoch, float(cur_shallow_score), float(best_shallow_score), best_epoch_shallow))
            logger.info('[CurShallowMetrics (dice={}, iou={}, mae={})] => [BestShallowMetrics (dice={}, iou={}, mae={})]'
                        .format(cur_shallow_dice, cur_shallow_iou, cur_shallow_mae, best_shallow_dice, best_shallow_iou, best_shallow_mae))

            # # ---- deep predict performance ----
            # print('[Val] [Epoch: {}] => [DeepScore: {:.4f}] => [bestDeepScore_from_ckpt: {:.4f}] => [bestDeepEpoch_from_ckpt: {}]'
            #       .format(epoch, float(cur_deep_score), float(best_deep_score), best_epoch_deep))
            # print('[CurDeepMetrics (dice={}, iou={}, mae={})] => [BestDeepMetrics (dice={}, iou={}, mae={})]'
            #       .format(cur_deep_dice, cur_deep_iou, cur_deep_mae, best_deep_dice, best_deep_iou, best_deep_mae))
            # logger.info('[Val Info]: Epoch: {}, DeepScore: {:.4f}, bestDeepScore_from_ckpt: {:.4f}, bestDeepEpoch_from_ckpt: {}.'
            #             .format(epoch, float(cur_deep_score), float(best_deep_score), best_epoch_deep))
            # logger.info('[CurDeepMetrics (dice={}, iou={}, mae={})] => [BestDeepMetrics (dice={}, iou={}, mae={})]'
            #             .format(cur_deep_dice, cur_deep_iou, cur_deep_mae, best_deep_dice, best_deep_iou, best_deep_mae))

# def val_test_with_multiMetrics(test_loader, model, epoch, opt, logger, start_epoch, writer=None):
#     model.eval()
#     print('=' * 35, 'Val Epoch: ' + str(epoch), '=' * 35)
#
#     # ---- set save path ----
#     model_save_path = opt.train_model_save_path  # 用于保存best model
#     os.makedirs(model_save_path, exist_ok=True)
#
#     # ---- switch to full branch ----
#     global best_shallow_metric_dict, best_shallow_score, best_epoch_shallow, \
#         best_deep_metric_dict, best_deep_score, best_epoch_deep
#
#     # ---- shallow ----
#     shallowDice = Measure.DICE()
#     shallowIOU = Measure.IoU()
#     shallowMAE = Measure.MAE()
#     shallow_metric_dict = dict()
#     # ---- deep ----
#     deepDice = Measure.DICE()
#     deepIOU = Measure.IoU()
#     deepMAE = Measure.MAE()
#     deep_metric_dict = dict()
#
#     with torch.no_grad():
#
#         # ---- get the evaluation of the performance ----
#         for i in range(test_loader.size):
#             image, gt, name, img_for_post = test_loader.load_data()
#             gt = np.asarray(gt, np.float32)
#             image = image.cuda()
#             logit1, logit2, logit3, logit4, deep_logit1, deep_logit2, deep_logit3 = model(image)
#
#             # ---- get shallow pred ----
#             pred_shallow = F.interpolate(logit1, size=gt.shape, mode='bilinear', align_corners=False)
#             pred_shallow = pred_shallow.sigmoid().data.cpu().numpy().squeeze()
#             pred_shallow = (pred_shallow - pred_shallow.min()) / (pred_shallow.max() - pred_shallow.min() + 1e-8) * 255  # *255还原到压缩前
#             # import cv2
#             # my_save_path = r'E:\model\My-Model\PolypNet\result\record_68\tmp_cvc_300/'
#             # cv2.imwrite(my_save_path + name, pred_shallow)
#             pred_shallow = np.asarray(pred_shallow + 0.5, np.int16)
#
#             # ---- get deep pred ----
#             pred_deep = F.interpolate(deep_logit1, size=gt.shape, mode='bilinear', align_corners=False)
#             pred_deep = pred_deep.sigmoid().data.cpu().numpy().squeeze()
#             pred_deep = (pred_deep - pred_deep.min()) / (pred_deep.max() - pred_deep.min() + 1e-8) * 255  # 同上
#
#             # ---- 计算单个样本的指标 ----
#             shallowDice.step(pred=pred_shallow, gt=gt)
#             shallowIOU.step(pred=pred_shallow, gt=gt)
#             shallowMAE.step(pred=pred_shallow, gt=gt)
#             deepDice.step(pred=pred_deep, gt=gt)
#             deepIOU.step(pred=pred_deep, gt=gt)
#             deepMAE.step(pred=pred_deep, gt=gt)
#
#         # ---- 计算整个loader中样本的指标 ----
#         shallow_metric_dict.update(dice=shallowDice.get_results()['dice'])
#         shallow_metric_dict.update(iou=shallowIOU.get_results()['iou'])
#         shallow_metric_dict.update(mae=shallowMAE.get_results()['mae'])
#         deep_metric_dict.update(dice=deepDice.get_results()['dice'])
#         deep_metric_dict.update(iou=deepIOU.get_results()['iou'])
#         deep_metric_dict.update(mae=deepMAE.get_results()['mae'])
#
#         cur_shallow_score = shallow_metric_dict['dice'] + shallow_metric_dict['iou'] - shallow_metric_dict['mae']
#         cur_deep_score = deep_metric_dict['dice'] + deep_metric_dict['iou'] - deep_metric_dict['mae']
#
#         # ---- update the performance and save the best model ----
#         if epoch == start_epoch:  # record the mae of start epoch
#             best_shallow_metric_dict = shallow_metric_dict
#             best_deep_metric_dict = deep_metric_dict
#             best_shallow_score = cur_shallow_score
#             best_deep_score = cur_deep_score
#             best_epoch_shallow = epoch
#             best_epoch_deep = epoch
#
#         # ---- save model with best shallow score ----
#         if cur_shallow_score > best_shallow_score:
#             best_shallow_metric_dict = shallow_metric_dict
#             best_shallow_score = cur_shallow_score
#             best_epoch_shallow = epoch
#             if start_epoch == 1:
#                 torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
#                            model_save_path + '/bestShallowModel_train_from_scratch.pth')
#             else:
#                 torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
#                            model_save_path + '/bestShallowModel_train_from_checkpoint.pth')
#             print('>>> Save best SHALLOW model at epoch: {}.'.format(epoch))
#
#         # ---- save model with best deep score ----
#         if cur_deep_score > best_deep_score:
#             best_deep_metric_dict = deep_metric_dict
#             best_deep_score = cur_deep_score
#             best_epoch_deep = epoch
#             if start_epoch == 1:
#                 torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
#                            model_save_path + '/bestDeepModel_train_from_scratch.pth')
#             else:
#                 torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
#                            model_save_path + '/bestDeepModel_train_from_checkpoint.pth')
#             print('>>> Save best DEEP model at epoch: {}.'.format(epoch))
#
#         # ---- get value ----
#         cur_shallow_dice = shallow_metric_dict['dice'].round(4)
#         cur_shallow_iou = shallow_metric_dict['iou'].round(4)
#         cur_shallow_mae = shallow_metric_dict['mae'].round(4)
#         cur_deep_dice = deep_metric_dict['dice'].round(4)
#         cur_deep_iou = deep_metric_dict['iou'].round(4)
#         cur_deep_mae = deep_metric_dict['mae'].round(4)
#         best_shallow_dice = best_shallow_metric_dict['dice'].round(4)
#         best_shallow_iou = best_shallow_metric_dict['iou'].round(4)
#         best_shallow_mae = best_shallow_metric_dict['mae'].round(4)
#         best_deep_dice = best_deep_metric_dict['dice'].round(4)
#         best_deep_iou = best_deep_metric_dict['iou'].round(4)
#         best_deep_mae = best_deep_metric_dict['mae'].round(4)
#
#         # ---- print/log the performance ----
#         if start_epoch == 1:
#             # ---- shallow predict performance ----
#             print('[Val] [Epoch: {}] => [ShallowScore: {:.4f}] => [bestShallowScore_f_scratch: {:.4f}] => [bestShallowEpoch_f_scratch: {}]'
#                   .format(epoch, float(cur_shallow_score), float(best_shallow_score), best_epoch_shallow))
#             print('[CurShallowMetrics (dice={}, iou={}, mae={})] => [BestShallowMetrics (dice={}, iou={}, mae={})]'
#                   .format(cur_shallow_dice, cur_shallow_iou, cur_shallow_mae, best_shallow_dice, best_shallow_iou, best_shallow_mae))
#             # logger.info('[Val Info]: Epoch: {}, ShallowScore: {:.4f}, bestShallowScore_f_scratch: {:.4f}, bestShallowEpoch_f_scratch: {}.'
#             #             .format(epoch, float(cur_shallow_score), float(best_shallow_score), best_epoch_shallow))
#             # logger.info('[CurShallowMetrics (dice={}, iou={}, mae={})] => [BestShallowMetrics (dice={}, iou={}, mae={})]'
#             #             .format(cur_shallow_dice, cur_shallow_iou, cur_shallow_mae, best_shallow_dice, best_shallow_iou, best_shallow_mae))
#
#             # ---- deep predict performance ----
#             print('[Val] [Epoch: {}] => [DeepScore: {:.4f}] => [bestDeepScore_f_scratch: {:.4f}] => [bestDeepEpoch_f_scratch: {}]'
#                   .format(epoch, float(cur_deep_score), float(best_deep_score), best_epoch_deep))
#             print('[CurDeepMetrics (dice={}, iou={}, mae={})] => [BestDeepMetrics (dice={}, iou={}, mae={})]'
#                   .format(cur_deep_dice, cur_deep_iou, cur_deep_mae, best_deep_dice, best_deep_iou, best_deep_mae))
#             # logger.info('[Val Info]: Epoch: {}, ShallowScore: {:.4f}, bestShallowScore_f_scratch: {:.4f}, bestShallowEpoch_f_scratch: {}.'
#             #             .format(epoch, float(cur_shallow_score), float(best_shallow_score), best_epoch_shallow))
#             # logger.info('[CurDeepMetrics (dice={}, iou={}, mae={})] => [BestDeepMetrics (dice={}, iou={}, mae={})]'
#             #             .format(cur_deep_dice, cur_deep_iou, cur_deep_mae, best_deep_dice, best_deep_iou, best_deep_mae))
#         else:
#             # ---- shallow predict performance ----
#             print('[Val] [Epoch: {}] => [ShallowScore: {:.4f}] => [bestShallowScore_from_ckpt: {:.4f}] => [bestShallowEpoch_from_ckpt: {}]'
#                   .format(epoch, float(cur_shallow_score), float(best_shallow_score), best_epoch_shallow))
#             print('[CurShallowMetrics (dice={}, iou={}, mae={})] => [BestShallowMetrics (dice={}, iou={}, mae={})]'
#                   .format(cur_shallow_dice, cur_shallow_iou, cur_shallow_mae, best_shallow_dice, best_shallow_iou, best_shallow_mae))
#             # logger.info('[Val Info]: Epoch: {}, ShallowScore: {:.4f}, bestShallowScore_from_ckpt: {:.4f}, bestShallowEpoch_from_ckpt: {}.'
#             #             .format(epoch, float(cur_shallow_score), float(best_shallow_score), best_epoch_shallow))
#             # logger.info('[CurShallowMetrics (dice={}, iou={}, mae={})] => [BestShallowMetrics (dice={}, iou={}, mae={})]'
#             #             .format(cur_shallow_dice, cur_shallow_iou, cur_shallow_mae, best_shallow_dice, best_shallow_iou, best_shallow_mae))
#
#             # ---- deep predict performance ----
#             print('[Val] [Epoch: {}] => [DeepScore: {:.4f}] => [bestDeepScore_from_ckpt: {:.4f}] => [bestDeepEpoch_from_ckpt: {}]'
#                   .format(epoch, float(cur_deep_score), float(best_deep_score), best_epoch_deep))
#             print('[CurDeepMetrics (dice={}, iou={}, mae={})] => [BestDeepMetrics (dice={}, iou={}, mae={})]'
#                   .format(cur_deep_dice, cur_deep_iou, cur_deep_mae, best_deep_dice, best_deep_iou, best_deep_mae))
#             # logger.info('[Val Info]: Epoch: {}, DeepScore: {:.4f}, bestDeepScore_from_ckpt: {:.4f}, bestDeepEpoch_from_ckpt: {}.'
#             #             .format(epoch, float(cur_deep_score), float(best_deep_score), best_epoch_deep))
#             # logger.info('[CurDeepMetrics (dice={}, iou={}, mae={})] => [BestDeepMetrics (dice={}, iou={}, mae={})]'
#             #             .format(cur_deep_dice, cur_deep_iou, cur_deep_mae, best_deep_dice, best_deep_iou, best_deep_mae))
