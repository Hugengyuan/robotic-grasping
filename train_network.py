import argparse
import datetime
import json
import logging
import os
import sys

import cv2
import numpy as np
import tensorboardX
import torch
import torch.optim as optim
import torch.utils.data
from torchsummary import summary
from sklearn.model_selection import KFold

from hardware.device import get_device
from inference.models import get_network
from inference.post_process import post_process_output
from utils.data import get_dataset
from utils.dataset_processing import evaluation
from utils.visualisation.gridshow import gridshow


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    # 网络相关参数
    parser.add_argument('--network', type=str, default='grconvnet3',
                        help='设定使用的骨干网络，包括grconvnet到grconvnet4')
    parser.add_argument('--input-size', type=int, default=224,
                        help='网络第一层的输入大小，默认为224')
    parser.add_argument('--use-depth', type=int, default=1,
                        help='是否使用深度图像')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='是否使用RGB图像')
    parser.add_argument('--use-dropout', type=int, default=1,
                        help='是否使用Dropout')
    parser.add_argument('--dropout-prob', type=float, default=0.1,
                        help='设定dropout的概率，默认为10%')
    parser.add_argument('--channel-size', type=int, default=32,
                        help='输入的通道数，默认为32')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='判断为正确抓取的IOU的阈值，默认为25%重合就算做成功抓取')

    # 数据集相关参数
    parser.add_argument('--dataset', type=str,
                        help='设定使用的数据集 ("cornell" 或 "jaquard")')
    parser.add_argument('--dataset-path', type=str,
                        help='数据集的路径')
    parser.add_argument('--split', type=float, default=0.9,
                        help='设定用于训练的百分比，剩余的用于validation')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='是否开启shuffle')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='改变数据集的分割点，从而使得分割出来的测试集和训练集不同')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')

    # 训练的相关参数
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1000,
                        help='Batches per Epoch')
    parser.add_argument('--optim', type=str, default='adam',
                        help='优化器. (adam or SGD)')

    # 其他参数
    parser.add_argument('--description', type=str, default='',
                        help='Training description')
    parser.add_argument('--logdir', type=str, default='logs/',
                        help='Log的保存位置')
    parser.add_argument('--vis', action='store_true',
                        help='是否开启训练过程的可视化')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='强制使用cpu训练，默认为False')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='随机种子')

    args = parser.parse_args()
    return args


def validate(net, device, val_data, iou_threshold):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param iou_threshold: IoU threshold
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }

    ld = len(val_data)

    with torch.no_grad():
        for rgb_x, depth_x, y, didx, rot, zoom_factor in val_data:
            rgb_xc = rgb_x.to(device)
            depth_x = depth_x.repeat_interleave(3,1)
            depth_xc = depth_x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(rgb_xc, depth_xc, yc)

            loss = lossd['loss']

            results['loss'] += loss.item() / ld
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item() / ld

            q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])

            s = evaluation.calculate_iou_match(q_out,
                                               ang_out,
                                               val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                               no_grasps=1,
                                               grasp_width=w_out,
                                               threshold=iou_threshold
                                               )

            if s:
                results['correct'] += 1
            else:
                results['failed'] += 1

    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }

    net.train()

    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    while batch_idx <= batches_per_epoch:
        for rgb_x, depth_x, y, _, _, _ in train_data:
            batch_idx += 1
            if batch_idx >= batches_per_epoch:
                break

            rgb_xc = rgb_x.to(device)
            depth_x = depth_x.repeat_interleave(3,1)
            depth_xc = depth_x.to(device)
            yc = [yy.to(device) for yy in y]
            lossd = net.compute_loss(rgb_xc, depth_x, yc)

            loss = lossd['loss']

            if batch_idx % 100 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))

            results['loss'] += loss.item()
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Display the images
            if vis:
                imgs = []
                n_img = min(4, x.shape[0])
                for idx in range(n_img):
                    imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
                        x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in
                                                      lossd['pred'].values()])
                gridshow('Display', imgs,
                         [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0),
                          (0.0, 1.0)] * 2 * n_img,
                         [cv2.COLORMAP_BONE] * 10 * n_img, 10)
                cv2.waitKey(2)

    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results


def run():
    args = parse_args()

    # 设置输出的log位置以及相关的参数
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

    save_folder = os.path.join('/kaggle/working/logs', net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(save_folder)
    
    # 存储命令行的参数
    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)

    # 初始化logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename="{0}/{1}.log".format(save_folder, 'log'),
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    # 将logging的句柄绑定到console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 设置logging的格式以遍展示在console上
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # 获取设备
    device = get_device(args.force_cpu)

    
    logging.info('Log files were saved to dir {}'.format(save_folder))
    
    
    # 载入数据集
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    dataset = Dataset(args.dataset_path,
                      output_size=args.input_size,
                      ds_rotate=args.ds_rotate,
                      random_rotate=True,
                      random_zoom=True,
                      include_depth=args.use_depth,
                      include_rgb=args.use_rgb)
    logging.info('Dataset size is {}'.format(dataset.length))

    
    # Creating data indices for training and validation splits
    #如果不使用五折交叉验证
    indices = list(range(dataset.length))
    split = int(np.floor(args.split * dataset.length))
    if args.ds_shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    logging.info('Training size: {}'.format(len(train_indices)))
    logging.info('Validation size: {}'.format(len(val_indices)))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    
    ##如果要使用五折交叉验证
    #kf = KFold(n_splits=5, random_state=42, shuffle=True)  # 初始化KFold
    #for train_indices , val_indices in kf.split(indices):  # 调用split方法切分数据
    #    logging.info('train_index:%s {}'.format(len(train_indices)))
    #    logging.info('test_index:%s {}'.format(len(val_indices)))
    
    #train_files = []   # 存放5折的训练集划分
    #test_files = []     # # 存放5折的测试集集划分
    
    #for k, (Trindex, Tsindex) in enumerate(kf.split(indices)):
    #    train_files.append(np.array(indices)[Trindex].tolist())
    #    test_files.append(np.array(indices)[Tsindex].tolist())
    
    #train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_files[2])
    #val_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_files[2])

    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    val_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=val_sampler
    )
    logging.info('Done')

    # 载入网络
    logging.info('Loading Network...')
    #input_channels = 1 * args.use_depth + 3 * args.use_rgb
    input_channels = 3
    network = get_network(args.network)
    net = network(
        input_channels=input_channels,
        dropout=args.use_dropout,
        prob=args.dropout_prob,
        channel_size=args.channel_size
    )

    net = net.to(device)
    logging.info('Done')

    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(net.parameters())
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(args.optim))

    # 打印网络结构
#     summary(net, (input_channels, 1, args.input_size, args.input_size))
#     f = open(os.path.join(save_folder, 'arch.txt'), 'w')
#     sys.stdout = f
#     summary(net, (input_channels, 1, args.input_size, args.input_size))
#     sys.stdout = sys.__stdout__
#     f.close()
    
    # 使用best_iou保存
    best_iou = 0.0
    for epoch in range(args.epochs):
        # 训练一个epoch
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, vis=args.vis)

        # 训练一个epoch后将loss添加到tensorboard中
        tb.add_scalar('loss/train_loss', train_results['loss'], epoch)
        for n, l in train_results['losses'].items():
            tb.add_scalar('train_loss/' + n, l, epoch)

        # 开始验证
        logging.info('Validating...')
        test_results = validate(net, device, val_data, args.iou_threshold)
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct'] / (test_results['correct'] + test_results['failed'])))

        # 将验证的loss添加到tensorboard中
        tb.add_scalar('loss/IOU', test_results['correct'] / (test_results['correct'] + test_results['failed']), epoch)
        tb.add_scalar('loss/val_loss', test_results['loss'], epoch)
        for n, l in test_results['losses'].items():
            tb.add_scalar('val_loss/' + n, l, epoch)

        # 每10轮保存最好的iou的模型
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        if iou > best_iou or epoch == 0 or (epoch % 10) == 0:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.2f' % (epoch, iou)))
            best_iou = iou


if __name__ == '__main__':
    run()
