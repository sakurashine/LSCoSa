
import argparse
import builtins
import math
import os
import random
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
import model.ResNet_linear as models
import numpy as np
from ops.os_operation import mkdir
from data_processing.datasets import get_dataset, HyperX
from data_processing.utils import  get_device, sample_gt, count_sliding_window, compute_imf_weights, metrics, logger, display_dataset, display_goundtruth, sliding_window, grouper
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import visdom
import datetime
from torchsummary import summary
import logging
from scipy.io import loadmat


parser = argparse.ArgumentParser(description='PyTorch HSI Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',  
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',  
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=32, type=int,  
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total ')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',  
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,  
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,  
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--rank', default=0, type=int,  
                    help='node rank for distributed training')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,  
                    help='GPU id to use.')

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to pretrained checkpoint')
parser.add_argument("--dataset", type=str, default="IndianPines", help="which dataset is used to linear prob")
parser.add_argument('--folder', default="../dataset/IndianPines", type=str, metavar='DIR',
                        help='path to dataset')
parser.add_argument("--load_data", type=str, default=None,  
                           help="Samples use of training")
parser.add_argument('--training_percentage', type=float, default=0.01,  
                           help="Percentage of samples to use for training")
parser.add_argument('--sampling_mode', type=str, default='random',  
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
                        help='exponent value in the norm formulation ')           

best_acc1 = 0

def main():
    args = parser.parse_args()  
    if args.seed is not None:  
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:  
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    RUN = args.run
    DATASET = args.dataset
    file_date = datetime.datetime.now().strftime('%Y-%m-%d')
    log_date = datetime.datetime.now().strftime('%Y-%m-%d:%H:%M')
    filepath = 'test_log'
    if not os.path.isdir(filepath):
	    os.mkdir(filepath)
    log = logger('./test_log/logs-' + file_date + DATASET +'.txt')
    logging.getLogger('matplotlib.font_manager').disabled = True  
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
    N_CLASSES = len(LABEL_VALUES)
    N_BANDS = img.shape[-1]
    FINE_TUNE = args.fine_tune
    RAW = args.raw
    hyperparams.update(
        {'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'center_pixel': True, 'device': CUDA_DEVICE, 'fine_tune': FINE_TUNE})
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
    log.info("dataset{}loaded.".format(DATASET))
    log.info("label names:{}".format(LABEL_VALUES))
    log.info("label numbers:{}".format(N_CLASSES))
    log.info("bands:{}".format(N_BANDS))
    
    # OA list
    acc_dataset = np.zeros([RUN, 1])
    # AA list
    A = np.zeros([RUN, N_CLASSES-1])
    # Kappa list
    K = np.zeros([RUN, 1])
    
    for i in range(RUN):
        log.info("==========================================================================================")
        log.info("======================================RUN:{}===============================================".format(i))
        log.info("==========================================================================================")
        model = models.resnet18(num_classes=N_CLASSES, num_bands=N_BANDS, fine_tune = FINE_TUNE)  

        # froze layers except FC
        for name, param in model.named_parameters():  
            if args.fine_tune == 'no':
                ft = False
            else:
                ft = True
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = ft
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()
        linear_keyword="fc"
        
        # load pretrained model
        if args.pretrained:
            if os.path.isfile(args.pretrained): 
                print("=> loading checkpoint '{}'".format(args.pretrained))
                checkpoint = torch.load(args.pretrained, map_location="cpu")
                state_dict = checkpoint['state_dict']
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
                        # remove prefix
                        state_dict[k[len("encoder_q."):]] = state_dict[k]
                        del state_dict[k]
                args.start_epoch = 0
                # model = nn.DataParallel(model).cuda()
                msg = model.load_state_dict(state_dict, strict=False)
                # print("set(msg.missing_keys)", set(msg.missing_keys))
                assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}
                print("=> loaded pre-trained model '{}'".format(args.pretrained))
            else:
                print("=> no checkpoint found at '{}'".format(args.pretrained))  

        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        init_lr = args.lr
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        log.info("=> use LARS optimizer.")
        optimizer = torch.optim.SGD(parameters, init_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
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

        
        if args.resume is None:
            args.resume = os.path.join(save_dir,"checkpoint.pth.tar")
        if args.resume is not None and os.path.isfile(args.resume):
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
        cudnn.benchmark = True

        # load datasets
        if args.dataset=='WHU-Hi-HanChuan':
            if LOAD_DATA:
                print("support SAMPLE_NUMS only.")
            elif SAMPLE_NUMS:
                print("{}training samples selected.".format(SAMPLE_NUMS))
                if(SAMPLE_NUMS == 25 or SAMPLE_NUMS == 50 or SAMPLE_NUMS == 100 or SAMPLE_NUMS == 150 or SAMPLE_NUMS == 200 or SAMPLE_NUMS == 250 or SAMPLE_NUMS == 300):
                    log.info("sample mode: fixed")
                    train_gt_file = '../dataset/WHU-Hi-HanChuan/Training samples and test samples/Train'+ str(SAMPLE_NUMS) + '.mat'
                    test_gt_file = '../dataset/WHU-Hi-HanChuan/Training samples and test samples/Test'+ str(SAMPLE_NUMS) + '.mat'
                    train_gt = loadmat(train_gt_file)['Train'+str(SAMPLE_NUMS)]
                    test_gt = loadmat(test_gt_file)['Test'+str(SAMPLE_NUMS)]
                    log.info("{}training samples selected, with {}samples totally".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
                    log.info("{}testing samples selected, with {}samples totally".format(np.count_nonzero(test_gt), np.count_nonzero(gt)))
                else:
                    print("supported numbers for WHU-Hi-HanChuan are 25,50,100,150,200,250,300 only.")
            elif RAW=='yes':
                log.info("sample mode: raw")
                train_gt_file = '../dataset/WHU-Hi-HanChuan/WHU_Hi_HanChuan_gt.mat'
                test_gt_file = '../dataset/WHU-Hi-HanChuan/Training samples and test samples/Test250.mat'
                train_gt = loadmat(train_gt_file)['WHU_Hi_HanChuan_gt']
                test_gt = loadmat(test_gt_file)['Test250']
                log.info("{}training samples selected, with {}samples totally".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
                log.info("{}testing samples selected, with {}samples totally".format(np.count_nonzero(test_gt), np.count_nonzero(gt)))
            else:
                print("hyperparameters needed.")
        else:
            if LOAD_DATA:
                train_gt_file = '../dataset/' + DATASET + '/' + LOAD_DATA + '/train_gt.npy'
                test_gt_file  = '../dataset/' + DATASET + '/' + LOAD_DATA + '/test_gt.npy'
                train_gt = np.load(train_gt_file, 'r')
                test_gt = np.load(test_gt_file, 'r')
                log.info("{}training samples selected, with {}samples totally".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
                log.info("{}testing samples selected, with {}samples totally".format(np.count_nonzero(test_gt), np.count_nonzero(gt)))
            else:
                train_gt, test_gt = sample_gt(gt, TRAINING_PERCENTAGE, mode='fixed', sample_nums=SAMPLE_NUMS)
                log.info("{}training samples selected, with {}samples totally".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
                log.info("{}testing samples selected, with {}samples totally".format(np.count_nonzero(test_gt), np.count_nonzero(gt)))
        
        
        mask = np.unique(train_gt)
        tmp = []
        for v in mask:
            tmp.append(np.sum(train_gt==v))
        log.info("classes: {}".format(mask))
        log.info("training set: {}".format(tmp))
        mask = np.unique(test_gt)
        tmp = []
        for v in mask:
            tmp.append(np.sum(test_gt==v))
        log.info("testing set: {}".format(tmp))
         
        if CLASS_BALANCING:
            weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
            hyperparams['weights'] = torch.from_numpy(weights).float().cuda()
        
        train_dataset = HyperX(img, train_gt, **hyperparams)
        test_dataset = HyperX(img, test_gt, **hyperparams)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                    batch_size=hyperparams['batch_size'],
                                    shuffle=True,
                                    drop_last=False)
        log.info("Train dataloader:{}".format(len(train_loader))) # 9
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                    batch_size=hyperparams['batch_size'])
        print("start training FC layer.")    
                
        for k, v in hyperparams.items():
            log.info("{}:{}".format(k, v))
        model.cuda()

        for epoch in range(args.epochs):
            learning_rate = adjust_learning_rate(optimizer, init_lr, epoch, args) 
            feature_bank, feature_labels = train(train_loader, model, criterion, optimizer, epoch, args, N_CLASSES)
            
        prediction = test(model, img, hyperparams)
        results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=hyperparams['n_classes'])
        acc_dataset[i,0] = results['Accuracy']  
        A[i] = results['F1 scores'][1:]  
        K[i,0] = results['Kappa']  
        
        log.info('----------Training result----------')
        log.info("\nConfusion matrix:\n{}".format(results['Confusion matrix']))
        log.info("\nAccuracy:\n{:.4f}".format(results['Accuracy']))
        log.info("\nF1 scores:\n{}".format(np.around(results['F1 scores'], 4)))
        log.info("\nKappa:\n{:.4f}".format(results['Kappa']))
        print("Acc_dataset {}".format(acc_dataset))

    
    
    # mean accuray of multi runs
    OA_std = np.std(acc_dataset)
    OAMean = np.mean(acc_dataset)
    AA_std = np.std(A,1)
    AAMean = np.mean(A,1)
    Kappa_std = np.std(K)
    KappaMean = np.mean(K)

    AA = list(map('{:.2f}%'.format, AAMean))
    p = []
    log.info("{}results:".format(DATASET))
    for item,std in zip(AAMean,AA_std):
        p.append(str(round(item*100,2))+"+-"+str(round(std,2)))
    log.info(np.array(p))
    log.info("AAMean {:.2f} +-{:.2f}".format(np.mean(AAMean)*100,np.mean(AA_std)))
    log.info("{}".format(acc_dataset))
    log.info("OAMean {:.2f} +-{:.2f}".format(OAMean,OA_std))
    log.info("{}".format(K))
    log.info("KappaMean {:.2f} +-{:.2f}".format(KappaMean,Kappa_std))
    
   

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
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output, flat_emb = model(images)
        # output = F.normalize(output, dim=1)
        feature = torch.cat([feature, flat_emb], dim=0)
        label = torch.cat([label, target], dim=0)
        loss = criterion(output, target)  
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
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
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            output = output.to(device)
            target = target.to(device)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            loss = criterion(output, target)  
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


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
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


