
import builtins
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import math

import model.ResNet as models
from model.CaCo import CaCo, CaCo_PN
from ops.os_operation import mkdir, mkdir_rank
from training.train_utils import adjust_learning_rate2,save_checkpoint
from data_processing.loader import TwoCropsTransform, TwoCropsTransform2,GaussianBlur,Solarize
from ops.knn_monitor import knn_monitor
from data_processing.datasets import get_dataset,Hyper2X,HyperX
from data_processing.utils import sample_gt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings("ignore", category=Warning)


# create folders to save models
def init_log_path(args,batch_size):
    save_path = os.path.join(os.getcwd(), args.log_path)
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, args.dataset)
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "Type_"+str(args.type))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "lr_" + str(args.lr) + "_" + str(args.lr_final))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "memlr_"+str(args.memory_lr) +"_"+ str(args.memory_lr_final))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "t_" + str(args.moco_t) + "_memt" + str(args.mem_t))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "wd_" + str(args.weight_decay) + "_memwd" + str(args.mem_wd)) 
    mkdir_rank(save_path,args.rank)
    if args.moco_m_decay:
        save_path = os.path.join(save_path, "mocomdecay_" + str(args.moco_m))
    else:
        save_path = os.path.join(save_path, "mocom_" + str(args.moco_m))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "memgradm_" + str(args.mem_momentum))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "hidden" + str(args.mlp_dim)+"_out"+str(args.moco_dim))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "batch_" + str(batch_size))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "epoch_" + str(args.epochs))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "warm_" + str(args.warmup_epochs))
    mkdir_rank(save_path,args.rank)
    save_path = os.path.join(save_path, "time_" + str(args.time))
    mkdir_rank(save_path,args.rank)
    return save_path


def main_worker(gpu, ngpus_per_node, args):
    params = vars(args)
    args.gpu = gpu
    init_lr = args.lr
    total_batch_size = args.batch_size
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    
    # add random seed to control same initial
    seed = args.seed 
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    print("=> creating model '{}'".format(args.arch))

    # load dataset
    if args.dataset=='IndianPines' or args.dataset=='PaviaU' or \
        args.dataset=='WHU-Hi-HanChuan' or args.dataset=='Salinas' or \
        args.dataset=='Botswana' or args.dataset=='KSC' or \
        args.dataset=='Houston' or args.dataset=='HyRANK':
        print("load" + args.dataset + "dataset!")
        DATASET = args.dataset
        FOLDER = args.data_folder
        LOAD_DATA = args.load_data
        SAMPLING_MODE = args.sampling_mode
        CLASS_BALANCING = args.class_balancing
        hyperparams = vars(args)
        hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)
        print("dataset,folder: ",DATASET, FOLDER)
        print('hyperparams dict: ',hyperparams)
        img, gt, LABEL_VALUES, IGNORED_LABELS = get_dataset(DATASET, FOLDER)
        N_CLASSES = len(LABEL_VALUES)
        N_BANDS = img.shape[-1]
        hyperparams.update(
            {'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS})
        if LOAD_DATA:
            print("training percentage: ",LOAD_DATA)
            train_gt_file = '../dataset/' + DATASET + '/' + LOAD_DATA + '/train_gt.npy'
            test_gt_file  = '../dataset/' + DATASET + '/' + LOAD_DATA + '/test_gt.npy'
            print("ground truth address: ",train_gt_file)
            train_gt = np.load(train_gt_file, 'r')
            test_gt = np.load(test_gt_file, 'r')
        else:
            print("all the samples are used for pre-training!")
            train_gt = gt

        mask = np.unique(train_gt)
        tmp = []
        for v in mask:
            tmp.append(np.sum(train_gt==v))
        print("classes: {}".format(mask))
        print("sample numbers of each class: {}".format(tmp))
        print("{}training samples selected, with {}samples totally.".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))

        val_gt, _ = sample_gt(gt, train_size=hyperparams['validation_percentage'], mode=SAMPLING_MODE)
        test_gt, _ = sample_gt(gt, train_size=hyperparams['validation_percentage'], mode=SAMPLING_MODE)


        #   weights:[ 0.         14.          0.3255814   0.56        2.          1.
        #             0.63636364 14.          1.          14.         0.48275862  0.19178082
        #             0.77777778 2.33333333   0.36842105  1.16666667  4.66666667 ]
        # if CLASS_BALANCING:
        #     weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
            #    hyperparams.update({'weights': torch.from_numpy(weights)})
            # hyperparams['weights'] = torch.from_numpy(weights).float().cuda()
        
        train_dataset = Hyper2X(img, train_gt, **hyperparams)
        val_dataset = Hyper2X(img, val_gt, **hyperparams)
        test_dataset = HyperX(img, test_gt, **hyperparams)

        print("dataset " + args.dataset + " loaded!")
    else:
        print("We don't support this dataset currently")
        exit()


    # create Memory Bank
    Memory_Bank = CaCo_PN(args.cluster,args.moco_dim)
    # create model
    model = CaCo(models.__dict__[args.arch], args, args.moco_dim, args.moco_m, N_BANDS)  
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  

    from model.optimizer import  LARS
    optimizer = LARS(model.parameters(), init_lr,
                         weight_decay=args.weight_decay,
                         momentum=args.momentum)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        Memory_Bank=Memory_Bank.cuda(args.gpu)

    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    save_path = init_log_path(args,total_batch_size)
    print("save_path: ", save_path)
    if not args.resume:
        args.resume = os.path.join(save_path,"checkpoint_best.pth.tar")
        print("searching resume files ",args.resume)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            Memory_Bank.load_state_dict(checkpoint['Memory_Bank'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.knn_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.knn_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler, drop_last=False)


    model.eval()
    print("model: ", model)
    print("GPU consume before training: ", torch.cuda.memory_allocated()/1024/1024)
    
    # initial memory bank
    if args.ad_init and not os.path.isfile(args.resume):
        from training.init_memory import init_memory
        init_memory(train_loader, model, Memory_Bank, criterion,
                optimizer, 0, args)
        print("initial memory bank complete!")
    
    knn_path = os.path.join(save_path,"knn.log")
    train_log_path = os.path.join(save_path,"train.log")
    best_Acc=0
    
    MemaxLoss = []
    Alpha = []
    Label = []

    # begin training 
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        #adjust_learning_rate(optimizer, epoch, args)
        adjust_learning_rate2(optimizer, epoch, args, init_lr)    
        #if args.type<10:
        if args.moco_m_decay:
            moco_momentum = adjust_moco_momentum(epoch, args)
        else:
            moco_momentum = args.moco_m
        from training.train_caco import train_caco
        # print("current epoch: ", epoch)
        acc1, memax, alpha, Labels = train_caco(train_loader, model, Memory_Bank, criterion,
                                optimizer, epoch, args, train_log_path,moco_momentum)  
        MemaxLoss.append(memax)
        Alpha.append(alpha)
        Label.append(Labels)

        # KNN predict
        if epoch%args.knn_freq==0 or epoch<=20:
            print("gpu consuming before cleaning:", torch.cuda.memory_allocated()/1024/1024)
            torch.cuda.empty_cache()
            print("gpu consuming after cleaning:", torch.cuda.memory_allocated()/1024/1024)

            try:
                knn_test_acc=knn_monitor(model.encoder_q, val_loader, test_loader,
                        global_k=min(args.knn_neighbor,len(val_loader.dataset)))
                print({'*KNN monitor Accuracy': knn_test_acc})
                if args.rank ==0:
                    with open(knn_path,'a+') as file:
                        file.write('%d epoch KNN monitor Accuracy %f\n'%(epoch,knn_test_acc))
            except:
                print("small error raised in knn calcu")
                knn_test_acc=0
            torch.cuda.empty_cache()
            epoch_limit=200
            if knn_test_acc<=1.0 and epoch>=epoch_limit:
                exit()
        
        is_best=best_Acc>acc1
        best_Acc=max(best_Acc,acc1)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank == 0):
            save_dict={
            'epoch': epoch + 1,
            'arch': args.arch,
            'best_acc':best_Acc,
            'knn_acc': knn_test_acc,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'Memory_Bank':Memory_Bank.state_dict(),
            }

            if epoch%10==9:
                tmp_save_path = os.path.join(save_path, 'checkpoint_{:04d}.pth.tar'.format(epoch))
                save_checkpoint(save_dict, is_best=False, filename=tmp_save_path)
            tmp_save_path = os.path.join(save_path, 'checkpoint_best.pth.tar')

            save_checkpoint(save_dict, is_best=is_best, filename=tmp_save_path)
           
        
          
def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    return 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)

