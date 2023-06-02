
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
import model.ResNet_linear as models
import numpy as np
from ops.os_operation import mkdir
from data_processing.IndianPines import get_dataset, Hyper2X,HyperX
from data_processing.utils import  get_device, sample_gt, count_sliding_window, compute_imf_weights, metrics, logger, display_dataset, display_goundtruth, sliding_window, grouper, PCA_data, get_rank
from tqdm import tqdm
from training.train_utils import pca_dr
# from loss import dcl
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import visdom
import datetime
from torchsummary import summary
import logging
from scipy.io import loadmat

# 什么意思？
# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))+['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base']
# print(model_names)

parser = argparse.ArgumentParser(description='PyTorch HSI Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',  # benchmark网络结构，在高光谱上十来层就够了
                    # choices=model_names,
                    # help='model architecture: ' +
                    #     ' | '.join(model_names) +
                    #     ' (default: resnet50)')
                    )
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',  # 线程数，可以人为设定的吗？
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',  # 在恢复模型训练时可用
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=32, type=int,  # 一定要手动设定，默认值跑不动
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',  # 动量m
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,  # 权重衰减
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,  # 是指打印epoch进程的频率吗
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,  # 分布式训练的节点数（GPU块数）
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,  # 第几块GPU
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,  # 人为定义覆盖了
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,  # 指定用第几块GPU跑，一般不取或取0
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to CaCo pretrained checkpoint')
parser.add_argument("--dataset", type=str, default="IndianPines", help="which dataset is used to finetune")
parser.add_argument('--folder', default="../dataset/IndianPines", type=str, metavar='DIR',
                        help='path to dataset')
parser.add_argument("--load_data", type=str, default=None,  # 加载采样好的训练集文件
                           help="Samples use of training")
parser.add_argument('--training_percentage', type=float, default=0.01,  # 按训练比例随即比例从全样本文件采样
                           help="Percentage of samples to use for training")
parser.add_argument('--sampling_mode', type=str, default='random',  # 当需要固定采样时，修改sample_gt方法中的sample_num
                           help="Sampling mode (random sampling or disjoint, default: random)")
parser.add_argument('--class_balancing', action='store_true',
                         help="Inverse median frequency class balancing (default = False)")
parser.add_argument('--patch_size', type=int, default=11,
                         help="Size of the spatial neighbourhood (optional, if "
                              "absent will be set by the model)")
parser.add_argument('--validation_percentage', type=float, default=0.2,
                           help="In the training data set, percentage of the labeled data are randomly "
                                "assigned to validation groups.")
parser.add_argument('--supervision', type=str, default='full',
                         help="full supervision or semi supervision ")    
parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")   
parser.add_argument('--test_stride', type=int, default=1,
                         help="Sliding window step stride during inference (default = 1)")  
parser.add_argument('--sample_nums', type=int, default=None,
                           help="Number of samples to use for training and validation") 
parser.add_argument('--run', type=int, default=1,
                    help="Running times.")
parser.add_argument('--save_epoch', type=int, default=5,
                         help="Training save epoch")
parser.add_argument('--fine_tune', type=str, default='no',
                         help="Choose linear prob or fine-tune")    
parser.add_argument('--desc', type=str, default=' ',
                         help="Describing current experiment with one word")     
parser.add_argument('--raw', type=str, default='no',
                         help="Use raw image or not")       
parser.add_argument('--rho', default=0.4, type=float,
                        help='rho value for KL divergence ')    
parser.add_argument('--norm', default=2, type=float,
                        help='the exponent value in the norm formulation ')           

best_acc1 = 0

def main():
    args = parser.parse_args()  # 参数解析
    if args.seed is not None:  # 训练前设置随机种子，使实验结果可复现
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        print("打印随机种子：", np.random.random(args.seed))

    if args.gpu is not None:  # 不指定GPU，取None
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed  # GPU就一块，但是超参数设置了multiprocessing_distributed，也是分布式训练吗？

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    

    # 定义全局变量
    RUN = args.run
    DATASET = args.dataset
    # 生成日志
    file_date = datetime.datetime.now().strftime('%Y-%m-%d')
    log_date = datetime.datetime.now().strftime('%Y-%m-%d:%H:%M')
    log = logger('./test_log/logs-' + file_date + DATASET +'.txt')
    logging.getLogger('matplotlib.font_manager').disabled = True  # 禁止在日志中报缺失字体的错
    # info方法同时在终端和log日志文件打印内容
    log.info("---------------------------------------------------------------------")
    log.info("-----------------------------Next run log----------------------------")
    log.info("---------------------------{}--------------------------".format(log_date))
    log.info("---------------------------------------------------------------------")
    CUDA_DEVICE = get_device(log, args.cuda)
    FOLDER = args.folder
    SAMPLE_NUMS = args.sample_nums
    EPOCH = args.epochs
    LOAD_DATA = args.load_data
    TRAINING_PERCENTAGE = args.training_percentage
    SAMPLING_MODE = args.sampling_mode
    SAMPLE_NUMS = args.sample_nums
    CLASS_BALANCING = args.class_balancing
    
    hyperparams = vars(args)
    img, gt, LABEL_VALUES, IGNORED_LABELS = get_dataset(DATASET, FOLDER)
    # img = pca_dr(img,LOAD_DATA,DATASET,3)
    N_CLASSES = len(LABEL_VALUES)
    N_BANDS = img.shape[-1]
    FINE_TUNE = args.fine_tune
    RAW = args.raw
    hyperparams.update(
        {'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'center_pixel': True, 'device': CUDA_DEVICE, 'fine_tune': FINE_TUNE})
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
    log.info("已加载{}数据集".format(DATASET))
    log.info("标签类名：{}".format(LABEL_VALUES))
    log.info("标签数量：{}".format(N_CLASSES))
    log.info("波段数：{}".format(N_BANDS))
    
    # 定义一个列表用于存储run多次的总体准确率
    acc_dataset = np.zeros([RUN, 1])
    # 定义一个列表用于存储run多次的每类平均准确率
    A = np.zeros([RUN, N_CLASSES-1])
    # 定义一个列表用于存储run多次的总体准确率
    K = np.zeros([RUN, 1])
    
    for i in range(RUN):
        log.info("==========================================================================================")
        log.info("======================================RUN:{}===============================================".format(i))
        log.info("==========================================================================================")
        model = models.resnet18(num_classes=N_CLASSES, num_bands=N_BANDS, fine_tune = FINE_TUNE)  # 一种加载模型的方法，动态传参类别数和波段数ResNet.py line273
        # print("*************************model", model)
        # 冻结除了最后FC层的所有层
        for name, param in model.named_parameters():  # 对读取的model里的参数遍历，如果是非FC层的参数，则取消梯度回传
            # print("name: ", name)
            if args.fine_tune == 'no':
                ft = False
            else:
                ft = True
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = ft
        # 初始化FC层，并不是新建
        # print("111111111111111111111", model.layer1)
        # model.conv1.weight.data.normal_(mean=0.0, std=0.01)
        # model.bn1.bias.data.zero_()
        # model.layer1[0].conv1.weight.data.normal_(mean=0.0, std=0.01)
        # # model.layer1[0].conv1.bias.data.zero_()
        # model.layer1[0].bn1.weight.data.normal_(mean=0.0, std=0.01)
        # model.layer1[0].bn1.bias.data.zero_()
        # model.layer1[0].conv2.weight.data.normal_(mean=0.0, std=0.01)
        # # model.layer1[0].conv2.bias.data.zero_()
        # model.layer1[0].bn2.weight.data.normal_(mean=0.0, std=0.01)
        # model.layer1[0].bn2.bias.data.zero_()
        # model.layer2[0].conv1.weight.data.normal_(mean=0.0, std=0.01)
        # # model.layer2[0].conv1.bias.data.zero_()
        # model.layer2[0].bn1.weight.data.normal_(mean=0.0, std=0.01)
        # model.layer2[0].bn1.bias.data.zero_()
        # model.layer2[0].conv2.weight.data.normal_(mean=0.0, std=0.01)
        # # model.layer2[0].conv2.bias.data.zero_()
        # model.layer2[0].bn2.weight.data.normal_(mean=0.0, std=0.01)
        # model.layer2[0].bn2.bias.data.zero_()
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
        linear_keyword="fc"
        
        # 加载预训练模型
        if args.pretrained:
            if os.path.isfile(args.pretrained): 
                print("=> loading checkpoint '{}'".format(args.pretrained))
                checkpoint = torch.load(args.pretrained, map_location="cpu")
                # rename caco pre-trained keys
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # print(k)
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
                        # remove prefix
                        # print("问斩：", k)
                        state_dict[k[len("encoder_q."):]] = state_dict[k]
                        del state_dict[k]
                args.start_epoch = 0
                # 报错，加上.cuda这句后不报错
                # model = nn.DataParallel(model).cuda()
                msg = model.load_state_dict(state_dict, strict=False)
                # print("set(msg.missing_keys)", set(msg.missing_keys))
                assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
                print("=> loaded pre-trained model '{}'".format(args.pretrained))
            else:
                print("=> no checkpoint found at '{}'".format(args.pretrained))  

        # 仅优化线性分类头
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        # assert len(parameters) == 2  # fc.weight, fc.bias
        # 定义学习率、损失函数、优化器
        # infer learning rate before changing batch size
        # init_lr = args.lr * args.batch_size / 256  # 学习率并非自定义的值，要在此做计算
        init_lr = args.lr
        print("init_lr: ", init_lr)
        # 定义损失函数(criterion)和优化器，创建存储文件夹
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        # criterion = dcl.DCL(temperature=0.5)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        log.info("=> 使用 LARS 优化器")
        optimizer = torch.optim.SGD(parameters, init_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        # 一种修改优化器的方式：注释这两行，并注释adjust_learning_rate行
        from ops.LARS import SGD_LARC
        optimizer = SGD_LARC(optimizer, trust_coefficient=0.001, clip=False, eps=1e-8)
        
        
        args.pretrained = os.path.abspath(args.pretrained)
        save_dir = os.path.split(args.pretrained)[0]
        if args.rank==0:
            mkdir(save_dir)
        save_dir = os.path.join(save_dir, "linear_lars")
        if args.rank==0:
            mkdir(save_dir)
        save_dir = os.path.join(save_dir, "bs_%d" % args.batch_size)
        if args.rank==0:
            mkdir(save_dir)
        save_dir = os.path.join(save_dir, "lr_%.3f" % args.lr)
        if args.rank==0:
            mkdir(save_dir)
        save_dir = os.path.join(save_dir, "wd_" + str(args.weight_decay))
        if args.rank==0:
            mkdir(save_dir)

        '''
        # 从checkpoint恢复模型
        if args.resume is None:
            print("###############None")
            args.resume = os.path.join(save_dir,"checkpoint.pth.tar")
        if args.resume is not None and os.path.isfile(args.resume):
            print("#################Not None")
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            best_acc1 =0
        '''
        cudnn.benchmark = True

        # 数据集加载
        if args.dataset=='WHU-Hi-HanChuan':
            if LOAD_DATA:
                print("WHU-Hi-HanChuan数据集暂时只支持固定样本个数采样")
            elif SAMPLE_NUMS:
                print("挑选{}个训练样本".format(SAMPLE_NUMS))
                if(SAMPLE_NUMS == 25 or SAMPLE_NUMS == 50 or SAMPLE_NUMS == 100 or SAMPLE_NUMS == 150 or SAMPLE_NUMS == 200 or SAMPLE_NUMS == 250 or SAMPLE_NUMS == 300):
                    log.info("采样方式：固定样本个数")
                    # ../dataset/WHU-Hi-HanChuan/WHU_Hi_HanChuan.mat 
                    # ../dataset/WHU-Hi-HanChuan/WHU_Hi_HanChuan_gt.mat
                    train_gt_file = '../dataset/WHU-Hi-HanChuan/Training samples and test samples/Train'+ str(SAMPLE_NUMS) + '.mat'
                    test_gt_file = '../dataset/WHU-Hi-HanChuan/Training samples and test samples/Test'+ str(SAMPLE_NUMS) + '.mat'
                    train_gt = loadmat(train_gt_file)['Train'+str(SAMPLE_NUMS)]
                    test_gt = loadmat(test_gt_file)['Test'+str(SAMPLE_NUMS)]
                    log.info("挑选{}个训练样本，总计{}个可用样本".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
                    log.info("挑选{}个测试样本，总计{}个可用样本".format(np.count_nonzero(test_gt), np.count_nonzero(gt)))
                else:
                    print("WHU-Hi-HanChuan采样个数只支持25,50,100,150,200,250,300")
            elif RAW=='yes':
                log.info("采样方式：原图全采样")
                train_gt_file = '../dataset/WHU-Hi-HanChuan/WHU_Hi_HanChuan_gt.mat'
                test_gt_file = '../dataset/WHU-Hi-HanChuan/Training samples and test samples/Test250.mat'
                train_gt = loadmat(train_gt_file)['WHU_Hi_HanChuan_gt']
                test_gt = loadmat(test_gt_file)['Test250']
                log.info("挑选{}个训练样本，总计{}个可用样本".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
                log.info("挑选{}个测试样本，总计{}个可用样本".format(np.count_nonzero(test_gt), np.count_nonzero(gt)))
            else:
                print("没有进入任何条件")
        else:
            if LOAD_DATA:
                log.info("采样方式：固定样本比例")
                train_gt_file = '../dataset/' + DATASET + '/' + LOAD_DATA + '/train_gt.npy'
                test_gt_file  = '../dataset/' + DATASET + '/' + LOAD_DATA + '/test_gt.npy'
                train_gt = np.load(train_gt_file, 'r')
                test_gt = np.load(test_gt_file, 'r')
                log.info("挑选{}个训练样本，总计{}个可用样本".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
                log.info("挑选{}个测试样本，总计{}个可用样本".format(np.count_nonzero(test_gt), np.count_nonzero(gt)))
            else:
                log.info("采样方式：固定样本个数")
                train_gt, test_gt = sample_gt(gt, TRAINING_PERCENTAGE, mode='fixed', sample_nums=SAMPLE_NUMS)
                log.info("挑选{}个训练样本，总计{}个可用样本".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
                log.info("挑选{}个测试样本，总计{}个可用样本".format(np.count_nonzero(test_gt), np.count_nonzero(gt)))
        # 取消了验证集
        # val_gt, _ = sample_gt(gt, train_size=hyperparams['validation_percentage'], mode=SAMPLING_MODE)
        # print("挑选{}个验证样本，总计{}个可用样本".format(np.count_nonzero(val_gt), np.count_nonzero(gt))) 
        # print('验证集比例：',hyperparams['validation_percentage'])
        
        # 用于display_goundtruth处初始化visdom画图的环境
        if SAMPLING_MODE == 'fixed':
            vis = visdom.Visdom(
                env=DATASET + ' ' + hyperparams['arch'] + ' ' + 'PATCH_SIZE' + str(
                    hyperparams['patch_size']) + ',' + 'EPOCH' + str(hyperparams['epoch']))
        else:
            vis = visdom.Visdom(env=DATASET + ' ' + hyperparams['arch'] + ' ' + 'PATCH_SIZE' + str(
                hyperparams['patch_size']) + ' ' + 'EPOCH' + str(hyperparams['epochs']))
        if not vis.check_connection:
            print("Visdom is not connected. Did you run 'python -m visdom.server' ?")
        
        # 打印训练集、测试集的每个类别样本个数
        mask = np.unique(train_gt)
        tmp = []
        for v in mask:
            tmp.append(np.sum(train_gt==v))
        log.info("数据集类别：{}".format(mask))
        log.info("训练集大小：{}".format(tmp))
        mask = np.unique(test_gt)
        tmp = []
        for v in mask:
            tmp.append(np.sum(test_gt==v))
        # print(mask)
        log.info("测试集大小：{}".format(tmp))
         
        # display_goundtruth(gt=gt, vis=vis, caption = "Training {} samples selected".format(np.count_nonzero(gt)))
        if CLASS_BALANCING:
            weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
            hyperparams['weights'] = torch.from_numpy(weights).float().cuda()
        
        # 构造数据集加载器
        # train_dataset = Hyper2X(img, train_gt, **hyperparams)
        train_dataset = HyperX(img, train_gt, **hyperparams)
        log.info('训练集数据的形状：{}'.format(train_dataset.data.shape))
        log.info('训练集标签的形状：{}'.format(train_dataset.label.shape))           
        # val_dataset = HyperX(img, val_gt, **hyperparams)
        # 这行代码貌似后边没用上,val就视为了test？
        test_dataset = HyperX(img, test_gt, **hyperparams)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                    batch_size=hyperparams['batch_size'],
                                    shuffle=True,
                                    drop_last=False)
        log.info("Train dataloader:{}".format(len(train_loader))) # 9
        # val_loader = torch.utils.data.DataLoader(val_dataset,
        #                             batch_size=hyperparams['batch_size'],
        #                             drop_last=False)
        # log.info("Validation dataloader:{}".format(len(val_loader)))  # 29，参数键值对的个数？不是
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                    batch_size=hyperparams['batch_size'])
        print("数据集加载完毕，开始训练")    
                
        # 打印参数字典
        for k, v in hyperparams.items():
            log.info("{}:{}".format(k, v))
        log.info("Network :")
        with torch.no_grad():
            for input, _ in train_loader:
                break
            summary(model.to(hyperparams['device']), input.size()[1:])
        print(model)
        # 训练阶段
        for epoch in range(args.epochs):
            # 调整学习率方法的必要性验证？考虑是先初始一个大的步长，然后逐渐减小学习率，达到一个快速收敛的目的
            learning_rate = adjust_learning_rate(optimizer, init_lr, epoch, args) 
            # log.info('第{}个epoch的学习率:{}'.format(epoch, learning_rate))
            # 训练一个epoch
            feature_bank, feature_labels = train(train_loader, model, criterion, optimizer, epoch, args, N_CLASSES)
            
            
            # 对每若干轮embedding进行t-SNE可视化
            if((epoch) %99 == 0):
                print("当前对第" + str(epoch) + "个epoch进行可视化！")
                X = feature_bank[1:,:]
                Y = feature_labels[1:]
                print("feature_bank.shape", X.shape)  
                print("feature_labels.shape", Y.shape)  
                digits_final = TSNE(perplexity=30).fit_transform(X) 
                plot(digits_final, Y, epoch, args)  
            
                # print("打印memory bank的相关性矩阵！")
                # print("memory_bank.shape", Memory_Bank.W.shape, Memory_Bank.W)
                # corr = np.corrcoef(Memory_Bank.W)
                # print("corrcoef of memory bank", corr.shape, corr)
                # sns.set(font_scale=1.25)#字符大小设定
                # hm=sns.heatmap(corr, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
                # plt.savefig('./t-SNE/' + str(datetime.date.today()) +args.dataset +'_epoch'+ str(epoch) +'_tsne.png', dpi=120)
            
            '''
            # 添加一个对固定采样200个点数据集的t-SNE可视化方法，仅做观测用
            if(epoch == 199):
                train_gt200_file = '../dataset/' + DATASET + '/train_gt200.npy'
                train_gt200 = np.load(train_gt200_file, 'r')
                mask200 = np.unique(train_gt200)
                tmp200 = []
                for v in mask200:
                    tmp200.append(np.sum(train_gt200==v))
                print("观测数据集类别：{}".format(mask200))
                print("观测训练集大小：{}".format(tmp200))
                train_dataset200 = HyperX(img, train_gt200, **hyperparams)
                train_loader200 = torch.utils.data.DataLoader(train_dataset200,
                                            batch_size=hyperparams['batch_size'],
                                            shuffle=True,
                                            drop_last=False)    
                feature200, label200 = train(train_loader200, model, criterion, optimizer, epoch, args, N_CLASSES)
                print("当前对观测数据集进行可视化！")
                X200 = feature200[1:,:]
                Y200 = label200[1:]
                # print("X200.shape", X200.shape)  # torch.Size([2251, 128])
                # print("Y200.shape", Y200.shape)  # torch.Size([2251])
                digits_final200 = TSNE(perplexity=30).fit_transform(X200) 
                plot(digits_final200, Y200, epoch, args)  
            '''
        # 验证阶段，可选                    
        # if args.evaluate:
        #     validate(val_loader, model, criterion, args)
        
        # 测试阶段（古早测试法）
        prediction = test(model, img, hyperparams)
        print(prediction)
        print(prediction.shape)  # (1476, 256)
        results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=hyperparams['n_classes'])
        
        color_gt = display_goundtruth(gt=prediction, vis=vis, caption="Testing ground truth(full)" + "RUN{}".format(i+1))
        if args.load_data:
            plt.imsave("./result/" + str(datetime.date.today()) + "{} lr{} data{} patch{} {} finetune{} RUN{}Testing gt(full).png".format(hyperparams['dataset'],hyperparams['lr'],hyperparams['load_data'],hyperparams['patch_size'],hyperparams['desc'],hyperparams['fine_tune'],i+1), color_gt)
            mask = np.zeros(gt.shape, dtype='bool')
            for l in IGNORED_LABELS:
                mask[gt == l] = True
            prediction[mask] = 0
            color_gt_raw = display_goundtruth(gt=prediction, vis=vis, caption="Testing ground truth(semi)"+"RUN{}".format(i))
            plt.imsave("./result/" + str(datetime.date.today()) + "{} lr{} data{} patch{} {} finetune{} RUN{}Testing gt(semi).png".format(hyperparams['dataset'],hyperparams['lr'],hyperparams['load_data'],hyperparams['patch_size'],hyperparams['desc'],hyperparams['fine_tune'],i+1), color_gt_raw)
        elif args.sample_nums:
            plt.imsave("./result/" + str(datetime.date.today()) + "{} lr{} sample{} patch{} {} finetune{} RUN{}Testing gt.png".format(hyperparams['dataset'],hyperparams['lr'],hyperparams['sample_nums'],hyperparams['patch_size'],hyperparams['desc'],hyperparams['fine_tune'],i+1), color_gt)
            mask = np.zeros(gt.shape, dtype='bool')
            for l in IGNORED_LABELS:
                mask[gt == l] = True
            prediction[mask] = 0
            color_gt_raw = display_goundtruth(gt=prediction, vis=vis, caption="Testing ground truth(semi)"+"RUN{}".format(i))
            plt.imsave("./result/" + str(datetime.date.today()) + "{} lr{} sample{} patch{} {} finetune{} RUN{}Testing gt.png".format(hyperparams['dataset'],hyperparams['lr'],hyperparams['sample_nums'],hyperparams['patch_size'],hyperparams['desc'],hyperparams['fine_tune'],i+1), color_gt_raw)
        elif args.raw:
            plt.imsave("./result/" + str(datetime.date.today()) + "{} lr{} patch{} {} finetune{} RUN{}raw gt(full).png".format(hyperparams['dataset'],hyperparams['lr'],hyperparams['patch_size'],hyperparams['desc'],hyperparams['fine_tune'],i+1), color_gt)
            mask = np.zeros(gt.shape, dtype='bool')
            for l in IGNORED_LABELS:
                mask[gt == l] = True
            prediction[mask] = 0
            color_gt_raw = display_goundtruth(gt=prediction, vis=vis, caption="Testing ground truth(semi)"+"RUN{}".format(i))
            plt.imsave("./result/" + str(datetime.date.today()) + "{} lr{} patch{} {} finetune{} RUN{}raw gt(semi).png".format(hyperparams['dataset'],hyperparams['lr'],hyperparams['patch_size'],hyperparams['desc'],hyperparams['fine_tune'],i+1), color_gt_raw)
        acc_dataset[i,0] = results['Accuracy']  # 把每次RUN的总体准确率保存
        A[i] = results['F1 scores'][1:]  # 把每次RUN的每类准确率保存
        K[i,0] = results['Kappa']  # 把每次RUN的Kappa准确率保存
        
        log.info('----------Training result----------')
        log.info("\nConfusion matrix:\n{}".format(results['Confusion matrix']))
        log.info("\nAccuracy:\n{:.4f}".format(results['Accuracy']))
        log.info("\nF1 scores:\n{}".format(np.around(results['F1 scores'], 4)))
        log.info("\nKappa:\n{:.4f}".format(results['Kappa']))
        print("Acc_dataset {}".format(acc_dataset))

    
    
    # 计算多轮的平均准确率
    OA_std = np.std(acc_dataset)
    OAMean = np.mean(acc_dataset)
    AA_std = np.std(A,1)
    AAMean = np.mean(A,1)
    Kappa_std = np.std(K)
    KappaMean = np.mean(K)

    AA = list(map('{:.2f}%'.format, AAMean))
    # print("AAAAAAAAAAAAA",AA)
    # print(AAMean, AA_std)
    p = []
    log.info("{}数据集的结果如下".format(DATASET))
    for item,std in zip(AAMean,AA_std):
        p.append(str(round(item*100,2))+"+-"+str(round(std,2)))
    log.info(np.array(p))
    log.info("AAMean {:.2f} +-{:.2f}".format(np.mean(AAMean)*100,np.mean(AA_std)))
    log.info("{}".format(acc_dataset))
    log.info("OAMean {:.2f} +-{:.2f}".format(OAMean,OA_std))
    log.info("{}".format(K))
    log.info("KappaMean {:.2f} +-{:.2f}".format(KappaMean,Kappa_std))
    # 主函数main_worker收尾
    
   

# caco自带的训练法
def train(train_loader, model, criterion, optimizer, epoch, args, N_CLASSES):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    model.eval()
    feature = torch.rand(1,128).cuda() 
    label = torch.rand(1) .cuda()
    end = time.time()
    for i, (images, target ) in enumerate(train_loader):
        # images.shape torch.Size([32, 103, 11, 11])
        # target.shape torch.Size([32])
        
        # 此处的image需要取前半通道吗?似乎不用，预训练阶段取前半通道是用于训练特征空间用，与分类无关。要的
        # images = images[:,:100,:,:]

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        # print(model)
        output, flat_emb = model(images)
        # output = F.normalize(output, dim=1)
        feature = torch.cat([feature, flat_emb], dim=0)
        
        label = torch.cat([label, target], dim=0)
        loss = criterion(output, target)  # torch.Size([32, 17]),torch.Size([32])

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # print("打印loss个数，images的batch",loss.item(), images.size(0))
        losses.update(loss.item(), images.size(0))  # 32
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)
    return feature, label

# caco自带的验证法
def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    mAP = AverageMeter("mAP", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5, mAP],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    # 这里是验证部分的读loader，结构和训练部分类似，应该是训练一个epoch读一个epoch
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # print(images.size())
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            output = output.to(device)
            target = target.to(device)
            # output = concat_all_gather(output)
            # target = concat_all_gather(target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            loss = criterion(output, target)  # torch.Size([32, 17]),torch.Size([32])
            losses.update(loss.item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} mAP {mAP.avg:.3f} '
              .format(top1=top1, top5=top5, mAP=mAP))

    return top1.avg

# 定义t-SNE绘图函数
def plot(x, colors, epoch, args):
    # flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#DE10DE",
    #         "#FFFF00", "#30EE00", "#ffc64b", "#725e82", "#FF0000", "#44cef6",
    #         "#4b5cc4", "#8d4bbb", "#1bd1a5", "#ff0097", "#000000"]  # 0-16共17类的标签，0类不显示
    flatui = ["#9b59b6", "#7A80A4", "#0a5757", "#1DA96C", "#9FD06C", "#05B8E1",
            "#7F655E", "#FDA190", "#4A4D68", "#D1E0E9", "#C4C1C5", "#F2D266",
            "#B15546", "#CE7452", "#A59284", "#DFD2A3", "#F9831A"]  # 用于WHU-Hi-HanChuan数据集的可视化调色盘
    palette = np.array(sns.color_palette(flatui))
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    # print("colors", colors)
    # sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int8)])
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.cpu().numpy().astype(np.int8)])
    txts = []
    for i in range(17):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([pe.Stroke(linewidth=5, foreground="w"), pe.Normal()])
        txts.append(txt)
    if args.load_data:
        plt.savefig('./t-SNE/'+ str(datetime.date.today()) + '_' + args.dataset  + args.load_data + '_' + args.desc + '_' + args.fine_tune + ' finetune_' + 'epoch' + str(epoch) + '_tSNE.png', dpi=120)
    elif args.sample_nums:
        plt.savefig('./t-SNE/'+ str(datetime.date.today()) + '_' + args.dataset  + str(args.sample_nums) + '_' + args.desc + '_' + args.fine_tune + ' finetune_' + 'epoch' + str(epoch) + '_tSNE.png', dpi=120)
    return f, ax, txts


# 古早型高光谱测试函数
def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    batch_size, device = hyperparams['batch_size'], hyperparams['device']
    n_classes = hyperparams['n_classes']

    # probs = np.zeros(img.shape[:2] + (n_classes,))
    probs = np.zeros(img.shape[:2])
    img = np.pad(img, ((patch_size // 2, patch_size // 2), (patch_size // 2, patch_size // 2), (0, 0)), 'reflect')
    
    iterations = count_sliding_window(img, step=hyperparams['test_stride'], window_size=(patch_size, patch_size))
    
    for batch in tqdm(grouper(batch_size, sliding_window(img, step=1, window_size=(patch_size, patch_size))),
                      total=(iterations//batch_size),
                      desc="Inference on the image"
                      ):
        with torch.no_grad():
            data = [b[0] for b in batch]

            data = np.copy(data)

            data = data.transpose(0, 3, 1, 2)

            data = torch.from_numpy(data)

            # data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]

            data = data.to(device)
            data = data.type(torch.cuda.FloatTensor)
            # print(data.shape)
            output, _ = net(data)
            # print(output.shape)
            _, output = torch.max(output, dim=1)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu')
            if center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x, y] += out
                else:
                    probs[x:x + w, y:y + h] += out
    return probs

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


# 动态调整学习率
def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    # print("######################init_lr", init_lr)
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    # print("######################cur_lr", cur_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return param_group['lr']


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pred = pred.to(device)
        target = target.to(device)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()


