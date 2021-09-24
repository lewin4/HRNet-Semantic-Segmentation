# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import argparse
import os
import pprint

import logging
import timeit
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter
import sys

sys.path.append("../lib")
from config import config
from config import update_config
from core.criterion import CrossEntropyLovasz, OhemCrossEntropy
from core.function import train, validate
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel
import _init_paths
import models
import datasets


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        # required=True,
                        default=r"..\experiments\cityscapes\seg_hrnet_ocr_w48.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=454)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def get_sampler(dataset):
    from utils.distributed import is_distributed
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset)
    else:
        return None


def main():
    args = parse_args()

    host_name = os.environ['COMPUTERNAME']


    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    distributed = args.local_rank >= 0
    if distributed:
        device = torch.device('cuda:{}'.format(args.local_rank))
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # build model
    model = eval('models.' + config.MODEL.NAME +
                 '.get_seg_model')(config)
    # x = torch.randn((2, 3, 768, 1024),device="cpu")
    # y = model(x)    #output is list[2 items] everyone's shape is (4, 19, 128, 256)
    # dump_input = torch.rand(
    #     (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    # )
    # logger.info(get_model_summary(model.cuda(), dump_input.cuda()))

    # copy model file
    if distributed and args.local_rank == 0:
        this_dir = os.path.dirname(__file__)
        models_dst_dir = os.path.join(final_output_dir, 'models')
        # if os.path.exists(models_dst_dir):
        #     shutil.rmtree(models_dst_dir)
        # shutil.copytree(os.path.join(this_dir, '../lib/models'), models_dst_dir)

    if distributed:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU
    else:
        batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)

    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])

    extra_epoch_iters = 0
    if config.DATASET.EXTRA_TRAIN_SET:
        extra_train_dataset = eval('datasets.'+config.DATASET.DATASET)(
                    root=config.DATASET.ROOT,
                    list_path=config.DATASET.EXTRA_TRAIN_SET,
                    num_samples=None,
                    num_classes=config.DATASET.NUM_CLASSES,
                    multi_scale=config.TRAIN.MULTI_SCALE,
                    flip=config.TRAIN.FLIP,
                    ignore_label=config.TRAIN.IGNORE_LABEL,
                    base_size=config.TRAIN.BASE_SIZE,
                    crop_size=crop_size,
                    downsample_rate=config.TRAIN.DOWNSAMPLERATE,
                    scale_factor=config.TRAIN.SCALE_FACTOR)
        extra_train_sampler = get_sampler(extra_train_dataset)
        extra_trainloader = torch.utils.data.DataLoader(
            extra_train_dataset,
            batch_size=batch_size,
            shuffle=config.TRAIN.SHUFFLE and extra_train_sampler is None,
            num_workers=config.WORKERS,
            pin_memory=True,
            drop_last=True,
            sampler=extra_train_sampler)
        extra_epoch_iters = np.int(extra_train_dataset.__len__() /
                        config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))

    # test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    # test_dataset = eval('datasets.'+config.DATASET.DATASET)(
    #                     root=config.DATASET.ROOT,
    #                     list_path=config.DATASET.TEST_SET,
    #                     num_samples=config.TEST.NUM_SAMPLES,
    #                     num_classes=config.DATASET.NUM_CLASSES,
    #                     multi_scale=False,
    #                     flip=False,
    #                     ignore_label=config.TRAIN.IGNORE_LABEL,
    #                     base_size=config.TEST.BASE_SIZE,
    #                     crop_size=test_size,
    #                     downsample_rate=1)
    #
    # test_sampler = get_sampler(test_dataset)
    # testloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=config.WORKERS,
    #     pin_memory=True,
    #     sampler=test_sampler)

    trainloader, valloader = eval('datasets.' + config.DATASET.DATASET + ".get_loaders")(
                                image_dir=config.DATASET.IMAGE_DIR,
                                mask_dir=config.DATASET.MASK_DIR,
                                batch_size=batch_size,
                                num_worker=config.WORKERS,
                                pin_memory=True,
                                img_shape=crop_size)

    # def calculate_weigths_labels(label_dir, num_classes):
    #     # Create an instance from the data loader
    #     z = np.zeros((num_classes,))
    #     imgs = os.listdir(label_dir)
    #     # Initialize tqdm
    #     tqdm_batch = tqdm(imgs)
    #     print('Calculating classes weights')
    #     for label in tqdm_batch:
    #         label_path = os.path.join(label_dir, label)
    #         labels = np.array(Image.open(label_path), dtype=np.uint8)
    #         # y = y.detach().cpu().numpy()
    #         # mask = (y >= 0) & (y < num_classes)
    #         # labels = y[mask].astype(np.uint8)
    #         count_l = np.bincount(labels, minlength=num_classes)  ##统计每幅图像中不同类别像素的个数
    #         z += count_l
    #     tqdm_batch.close()
    #     total_frequency = np.sum(z)
    #     class_weights = []
    #     for frequency in z:
    #         class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))  ##这里是计算每个类别像素的权重
    #         class_weights.append(class_weight)
    #     ret = np.array(class_weights)
    #     # classes_weights_path = os.path.join(Path.db_root_dir(dataset), dataset + '_classes_weights.npy')  ##生成权重文件
    #     # np.save(classes_weights_path, ret)  ##把各类别像素权重保存到一个文件中
    #     return ret
    #
    # weight = calculate_weigths_labels(config.DATASET.MASK_DIR, 3)


    # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=config.LOSS.CLASS_WEIGHT)
    else:
        criterion = CrossEntropyLovasz(ignore_label=config.TRAIN.IGNORE_LABEL,
                                 weight=config.LOSS.CLASS_WEIGHT)

    model = FullModel(model, criterion).to(device)

    if config.MODEL.PRETRAINED is not None:
        model.model.init_weights(config.MODEL.PRETRAINED)

    if distributed:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
    # else:
    #     model = nn.DataParallel(model, device_ids=gpus).to(device)

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':

        params_dict = dict(model.named_parameters())
        if config.TRAIN.NONBACKBONE_KEYWORDS:
            bb_lr = []
            nbb_lr = []
            nbb_keys = set()
            for k, param in params_dict.items():
                if any(part in k for part in config.TRAIN.NONBACKBONE_KEYWORDS):
                    nbb_lr.append(param)
                    nbb_keys.add(k)
                else:
                    bb_lr.append(param)
            print(nbb_keys)
            params = [{'params': bb_lr, 'lr': config.TRAIN.LR},
                      {'params': nbb_lr, 'lr': config.TRAIN.LR * config.TRAIN.NONBACKBONE_MULT}]
        else:
            params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(params,
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV,
                                    )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = np.int(len(trainloader))

    best_mIoU = 0
    last_epoch = 0
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']

            # model.module.model.load_state_dict(
            #     {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
            model.model.load_state_dict(dct)
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
        if distributed:
            torch.distributed.barrier()

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    extra_iters = config.TRAIN.EXTRA_EPOCH * extra_epoch_iters

    losses = []
    mious = []

    for epoch in range(last_epoch, end_epoch):

        current_trainloader = extra_trainloader if epoch >= config.TRAIN.END_EPOCH else trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        # valid_loss, mean_IoU, IoU_array = validate(config, 
        #             testloader, model, writer_dict)

        if epoch >= config.TRAIN.END_EPOCH:
            train(config, epoch - config.TRAIN.END_EPOCH,
                  config.TRAIN.EXTRA_EPOCH, extra_epoch_iters,
                  config.TRAIN.EXTRA_LR, extra_iters,
                  extra_trainloader, optimizer, model, writer_dict, device)
        else:
            train(config, epoch, config.TRAIN.END_EPOCH,
                  epoch_iters, config.TRAIN.LR, num_iters,
                  trainloader, optimizer, model, writer_dict, device)

        valid_loss, mean_IoU, IoU_array = validate(config, epoch,
                                                   valloader, model, writer_dict, device)
        losses.append(valid_loss)
        mious.append(mean_IoU)
        if args.local_rank <= 0:
            logger.info('=> saving checkpoint to {}'.format(
                final_output_dir + 'checkpoint.pth.tar'))
            torch.save({
                'epoch': epoch + 1,
                'best_mIoU': best_mIoU,
                'state_dict': model.model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))
            if mean_IoU > best_mIoU:
                best_mIoU = mean_IoU
                torch.save(model,
                           os.path.join(final_output_dir, 'best_model.pth'))

            msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
                valid_loss, mean_IoU, best_mIoU)
            logging.info(msg)
            logging.info(IoU_array)

    if args.local_rank <= 0:
        torch.save(model.model.state_dict(),
                   os.path.join(final_output_dir, 'final_state.pth'))

        writer_dict['writer'].close()
        end = timeit.default_timer()
        logger.info('Hours: %d' % np.int((end - start) / 3600))
        logger.info('Done')

        plt.plot(losses, label='val 1oss')
        plt.title('validation loss', fontsize=12)
        plt.xlabel('epochs', fontsize=12)
        plt.legend(fontsize=12)
        plt.savefig("loss.jpg")
        # plt.show()
        plt.close()

        plt.plot(mious, label='val miou')
        plt.title('val miou', fontsize=12)
        plt.xlabel('epochs', fontsize=12)
        plt.legend(fontsize=12)
        plt.savefig("miou.jpg")
        # plt.show()
        plt.close()


if __name__ == '__main__':
    main()
