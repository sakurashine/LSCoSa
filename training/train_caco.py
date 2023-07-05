import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from training.train_utils import AverageMeter, ProgressMeter, accuracy
import copy

def train_caco(train_loader, model, Memory_Bank, criterion,
          optimizer, epoch, args, train_log_path,moco_momentum):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    mem_losses = AverageMeter('MemLoss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, mem_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    
    # update memory bank lr
    if epoch<args.warmup_epochs:
        cur_memory_lr =  args.memory_lr* (epoch+1) / args.warmup_epochs 
    elif args.memory_lr != args.memory_lr_final:
        cur_memory_lr = args.memory_lr_final + 0.5 * \
                   (1. + math.cos(math.pi * (epoch-args.warmup_epochs) / (args.epochs-args.warmup_epochs))) \
                   * (args.memory_lr- args.memory_lr_final)
    else:
        cur_memory_lr = args.memory_lr
    cur_adco_t =args.mem_t
    end = time.time()
    
    Labels = []
    for i, (data0, _, data1, _) in enumerate(train_loader):
        images = [data0, data1]
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
        
        batch_size = images[0].size(0)

        _, _, _, Dyrloss, alpha, labels1 = update_sym_network(model, images, args, Memory_Bank, losses, top1, top5,
        optimizer, criterion, mem_losses,moco_momentum,cur_memory_lr,cur_adco_t)
        Labels.append(labels1)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and args.rank==0:
            progress.display(i)
            if args.rank == 0:
                progress.write(train_log_path, i)
                   
    return top1.avg, Dyrloss, alpha, Labels


# update query encoder
def update_sym_network(model, images, args, Memory_Bank, 
                   losses, top1, top5, optimizer, criterion, mem_losses,
                   moco_momentum,memory_lr,cur_adco_t):
    model.zero_grad()
    q_pred, k_pred, q, k = model(im_q=images[0], im_k=images[1],run_type=0,moco_momentum=moco_momentum)
    d_norm1, d1, logits1 = Memory_Bank(q_pred)
    d_norm2, d2, logits2 = Memory_Bank(k_pred)
    
    # logits: Nx(1+K)
    with torch.no_grad():
        logits_keep1 = logits1.clone()
        logits_keep2 = logits2.clone()
    
    logits1 /= args.moco_t 
    logits2 /= args.moco_t 

    # search the indexs and labels of positives
    with torch.no_grad():
        d_norm21, d21, check_logits1 = Memory_Bank(k)
        logits_fix1 = copy.deepcopy(check_logits1)
        check_logits1 = check_logits1.detach()
        filter_index1 = torch.argmax(check_logits1, dim=1)
        labels1 = copy.deepcopy(filter_index1)
        
        for i in range(check_logits1.size(0)):
            check_logits1[i][filter_index1[i]] = 0
        filter_index12 = torch.argmax(check_logits1, dim=1)
        labels12 = copy.deepcopy(filter_index12)

        for i in range(check_logits1.size(0)):
            check_logits1[i][filter_index12[i]] = 0
        filter_index13 = torch.argmax(check_logits1, dim=1)
        labels13 = copy.deepcopy(filter_index13)

        for i in range(check_logits1.size(0)):
            check_logits1[i][filter_index13[i]] = 0
        filter_index14 = torch.argmax(check_logits1, dim=1)
        labels14 = copy.deepcopy(filter_index14)

        for i in range(check_logits1.size(0)):
            check_logits1[i][filter_index14[i]] = 0
        filter_index15 = torch.argmax(check_logits1, dim=1)
        labels15 = copy.deepcopy(filter_index15)

        for i in range(check_logits1.size(0)):
            check_logits1[i][filter_index15[i]] = 0
        filter_index16 = torch.argmax(check_logits1, dim=1)
        labels16 = copy.deepcopy(filter_index16)

        for i in range(check_logits1.size(0)):
            check_logits1[i][filter_index16[i]] = 0
        filter_index17 = torch.argmax(check_logits1, dim=1)
        labels17 = copy.deepcopy(filter_index17)

        for i in range(check_logits1.size(0)):
            check_logits1[i][filter_index17[i]] = 0
        filter_index18 = torch.argmax(check_logits1, dim=1)
        labels18 = copy.deepcopy(filter_index18)

        for i in range(check_logits1.size(0)):
            check_logits1[i][filter_index18[i]] = 0
        filter_index19 = torch.argmax(check_logits1, dim=1)
        labels19 = copy.deepcopy(filter_index19)

        for i in range(check_logits1.size(0)):
            check_logits1[i][filter_index19[i]] = 0
        filter_index110 = torch.argmax(check_logits1, dim=1)
        labels110 = copy.deepcopy(filter_index110)

        check_logits1 = logits_fix1
        d_norm22, d22, check_logits2 = Memory_Bank(q)
        check_logits2 = check_logits2.detach()
        logits_fix2 = check_logits2
        filter_index2 = torch.argmax(check_logits2, dim=1)
        for i in range(check_logits2.size(0)):
            check_logits2[i][filter_index2[i]] = 0
        filter_index22 = torch.argmax(check_logits2, dim=1)
        for i in range(check_logits2.size(0)):
            check_logits2[i][filter_index22[i]] = 0
        filter_index23 = torch.argmax(check_logits2, dim=1)
        for i in range(check_logits2.size(0)):
            check_logits2[i][filter_index23[i]] = 0
        filter_index24 = torch.argmax(check_logits2, dim=1)
        check_logits2 = logits_fix2
        
        labels2 = filter_index2
        labels22 = filter_index22
        labels23 = filter_index23
        labels24 = filter_index24
        
    
    # ┌─────────────────────────────────────────┐
    # |            caco loss with SwAV 
    # └─────────────────────────────────────────┘
    # caco_loss_SwAV = (criterion(logits1, labels1)+criterion(logits2, labels2))

    # ┌─────────────────────────────────────────┐
    # |                caco loss 
    # └─────────────────────────────────────────┘
    caco_loss = criterion(logits1, labels1)

    # ┌─────────────────────────────────────────┐
    # |            KL regularizer loss 
    # └─────────────────────────────────────────┘ 
    rho = args.rho
    rho_hat = torch.mean(torch.nn.functional.normalize(logits1, p=args.norm), dim=0)
    KL = rho_hat * torch.log(rho_hat / rho) + (1-rho_hat) * torch.log((1-rho_hat)/(1-rho))  # reverse KL
    # KL = rho * torch.log(rho / rho_hat) + (1-rho) * torch.log((1-rho)/(1-rho_hat))  # forward KL
    KL_loss =  torch.sum(KL) 
    H_rho = torch.sum(-rho_hat * torch.log(rho_hat))
    KL_alpha = 1 / (1 + np.exp( H_rho))  
 
    # ┌─────────────────────────────────────────┐
    # |           multiple positives 
    # └─────────────────────────────────────────┘
    Dy_x1 = logits1
    sim = torch.nn.functional.normalize(logits1)
    Dy_fourhot = np.zeros((args.batch_size, args.cluster))  # float
    for i in range(labels1.size(0)):
        if args.loss=="Dy2hot+1.6KL" or args.loss=="Dy2hot+0.4KL" or args.loss=="Dy2hot":
            Dy_fourhot[i] = np.zeros((1, args.cluster))
            Dy_fourhot[i].flat[labels1[i]] = 1
            Dy_fourhot[i].flat[labels12[i]] = KL_alpha
        elif args.loss=="Dy3hot+2KL" or args.loss=="Dy3hot+0.4KL" or args.loss=="Dy3hot+0.8KL" or \
                    args.loss=="Dy3hot+1.2KL" or args.loss=="Dy3hot+1.6KL" or args.loss=="Dy3hot":
            Dy_fourhot[i] = np.zeros((1, args.cluster))
            Dy_fourhot[i].flat[labels1[i]] = 1
            Dy_fourhot[i].flat[labels12[i]] = KL_alpha
            Dy_fourhot[i].flat[labels13[i]] = KL_alpha
        elif args.loss=="Dy4hot+2KL" or args.loss=="Dy4hot+0.4KL" or args.loss=="Dy4hot+0.8KL" or \
            args.loss=="Dy4hot+1.2KL" or args.loss=="Dy4hot+1.6KL" or args.loss=="Dy4hot":
            Dy_fourhot[i] = np.zeros((1, args.cluster))
            Dy_fourhot[i].flat[labels1[i]] = 1
            Dy_fourhot[i].flat[labels12[i]] = KL_alpha
            Dy_fourhot[i].flat[labels13[i]] = KL_alpha
            Dy_fourhot[i].flat[labels14[i]] = KL_alpha
        elif args.loss=="Dy5hot+1.6KL" or args.loss=="Dy5hot+0.4KL" or args.loss=="Dy5hot":
            Dy_fourhot[i] = np.zeros((1, args.cluster))
            Dy_fourhot[i].flat[labels1[i]] = 1
            Dy_fourhot[i].flat[labels12[i]] = KL_alpha
            Dy_fourhot[i].flat[labels13[i]] = KL_alpha
            Dy_fourhot[i].flat[labels14[i]] = KL_alpha
            Dy_fourhot[i].flat[labels15[i]] = KL_alpha
        elif args.loss=="Dy6hot+1.6KL" or args.loss=="Dy6hot+0.4KL" or args.loss=="Dy6hot":
            Dy_fourhot[i] = np.zeros((1, args.cluster))
            Dy_fourhot[i].flat[labels1[i]] = 1
            Dy_fourhot[i].flat[labels12[i]] = KL_alpha
            Dy_fourhot[i].flat[labels13[i]] = KL_alpha
            Dy_fourhot[i].flat[labels14[i]] = KL_alpha
            Dy_fourhot[i].flat[labels15[i]] = KL_alpha
            Dy_fourhot[i].flat[labels16[i]] = KL_alpha
        elif args.loss=="Dy7hot+1.6KL" or args.loss=="Dy7hot+0.4KL" or args.loss=="Dy7hot":
            Dy_fourhot[i] = np.zeros((1, args.cluster))
            Dy_fourhot[i].flat[labels1[i]] = 1
            Dy_fourhot[i].flat[labels12[i]] = KL_alpha
            Dy_fourhot[i].flat[labels13[i]] = KL_alpha
            Dy_fourhot[i].flat[labels14[i]] = KL_alpha
            Dy_fourhot[i].flat[labels15[i]] = KL_alpha
            Dy_fourhot[i].flat[labels16[i]] = KL_alpha
            Dy_fourhot[i].flat[labels17[i]] = KL_alpha
        elif args.loss=="Dy8hot+1.6KL" or args.loss=="Dy8hot+0.4KL" or args.loss=="Dy8hot":
            Dy_fourhot[i] = np.zeros((1, args.cluster))
            Dy_fourhot[i].flat[labels1[i]] = 1
            Dy_fourhot[i].flat[labels12[i]] = KL_alpha
            Dy_fourhot[i].flat[labels13[i]] = KL_alpha
            Dy_fourhot[i].flat[labels14[i]] = KL_alpha
            Dy_fourhot[i].flat[labels15[i]] = KL_alpha
            Dy_fourhot[i].flat[labels16[i]] = KL_alpha
            Dy_fourhot[i].flat[labels17[i]] = KL_alpha
            Dy_fourhot[i].flat[labels18[i]] = KL_alpha
        elif args.loss=="Dy9hot+1.6KL" or args.loss=="Dy9hot+0.4KL" or args.loss=="Dy9hot":
            Dy_fourhot[i] = np.zeros((1, args.cluster))
            Dy_fourhot[i].flat[labels1[i]] = 1
            Dy_fourhot[i].flat[labels12[i]] = KL_alpha
            Dy_fourhot[i].flat[labels13[i]] = KL_alpha
            Dy_fourhot[i].flat[labels14[i]] = KL_alpha
            Dy_fourhot[i].flat[labels15[i]] = KL_alpha
            Dy_fourhot[i].flat[labels16[i]] = KL_alpha
            Dy_fourhot[i].flat[labels17[i]] = KL_alpha
            Dy_fourhot[i].flat[labels18[i]] = KL_alpha
            Dy_fourhot[i].flat[labels19[i]] = KL_alpha
        elif args.loss=="Dy10hot+1.6KL" or args.loss=="Dy10hot+0.4KL" or args.loss=="Dy10hot":
            Dy_fourhot[i] = np.zeros((1, args.cluster))
            Dy_fourhot[i].flat[labels1[i]] = 1
            Dy_fourhot[i].flat[labels12[i]] = KL_alpha
            Dy_fourhot[i].flat[labels13[i]] = KL_alpha
            Dy_fourhot[i].flat[labels14[i]] = KL_alpha
            Dy_fourhot[i].flat[labels15[i]] = KL_alpha
            Dy_fourhot[i].flat[labels16[i]] = KL_alpha
            Dy_fourhot[i].flat[labels17[i]] = KL_alpha
            Dy_fourhot[i].flat[labels18[i]] = KL_alpha
            Dy_fourhot[i].flat[labels19[i]] = KL_alpha
            Dy_fourhot[i].flat[labels110[i]] = KL_alpha
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Dy_y1 = torch.tensor(Dy_fourhot).to(device)
    norm_y = torch.nn.functional.normalize(Dy_y1) 
    x_hot_loss = x_hot_CrossEntropy()
    Dyxhot = x_hot_loss(Dy_x1,norm_y) 


    # ┌─────────────────────────────────────────┐
    # |                total loss 
    # └─────────────────────────────────────────┘
    if args.loss=="caco":
        loss = caco_loss
    elif args.loss=='caco+KL':
        loss = caco_loss + KL_loss
    elif args.loss=="Dy4hot+KL":
        loss = Dyxhot + KL_loss
    elif args.loss=="caco+0.4KL":
        loss = caco_loss + KL_loss * 0.4
    elif args.loss=="caco+0.8KL":
        loss = caco_loss + KL_loss * 0.8
    elif args.loss=="caco+1.2KL":
        loss = caco_loss + KL_loss * 1.2
    elif args.loss=="caco+1.6KL":
        loss = caco_loss + KL_loss * 1.6
    elif args.loss=="caco+2KL":
        loss = caco_loss + KL_loss * 2
    elif args.loss=="Dy4hot+0.4KL":
        loss = Dyxhot + KL_loss * 0.4
    elif args.loss=="Dy4hot+0.8KL":
        loss = Dyxhot + KL_loss * 0.8
    elif args.loss=="Dy4hot+1.2KL":
        loss = Dyxhot + KL_loss * 1.2
    elif args.loss=="Dy4hot+1.6KL":
        loss = Dyxhot + KL_loss * 1.6
    elif args.loss=="Dy4hot+2KL":
        loss = Dyxhot + KL_loss * 2
    elif args.loss=="Dy2hot" or args.loss=="Dy3hot" or args.loss=="Dy4hot" or \
        args.loss=="Dy5hot" or args.loss=="Dy6hot" or args.loss=="Dy7hot" or \
            args.loss=="Dy8hot" or args.loss=="Dy9hot" or args.loss=="Dy10hot":
        loss = Dyxhot 
    else:
        print("Please set the loss hyperparameter!")
    
    # acc1/acc5 are (K+1)-way contrast classifier accuracy
    # measure accuracy and record loss
    acc1, acc5 = accuracy(logits1, labels1, topk=(1, 5))
    losses.update(loss.item(), images[0].size(0))
    top1.update(acc1.item(), images[0].size(0))
    top5.update(acc5.item(), images[0].size(0))
    acc1, acc5 = accuracy(logits2, labels2, topk=(1, 5))
    losses.update(loss.item(), images[0].size(0))
    top1.update(acc1.item(), images[0].size(0))
    top5.update(acc5.item(), images[0].size(0))
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # update memory bank
    with torch.no_grad():
        # logits: Nx(1+K)
        logits1 = logits_keep1/cur_adco_t#/args.mem_t
        # negative logits: NxK
        # logits: Nx(1+K)
        logits2 = logits_keep2/cur_adco_t#/args.mem_t
        
        p_qd1 = nn.functional.softmax(logits1, dim=1)
        p_qd1[torch.arange(logits1.shape[0]), filter_index1] = 1 - p_qd1[torch.arange(logits1.shape[0]), filter_index1]

        g1 = torch.einsum('cn,nk->ck', [q_pred.T, p_qd1]) / logits1.shape[0] - torch.mul(
            torch.mean(torch.mul(p_qd1, logits_keep1), dim=0), d_norm1)
        
        p_qd2 = nn.functional.softmax(logits2, dim=1)
        p_qd2[torch.arange(logits1.shape[0]), filter_index2] = 1 - p_qd2[torch.arange(logits2.shape[0]), filter_index2]

        g2 = torch.einsum('cn,nk->ck', [k_pred.T, p_qd2]) / logits2.shape[0] - torch.mul(
            torch.mean(torch.mul(p_qd2, logits_keep2), dim=0), d_norm2)
        g = -torch.div(g1, torch.norm(d1, dim=0))  - torch.div(g2,torch.norm(d2, dim=0))#/ args.mem_t  # c*k
        g /=cur_adco_t
        
        Memory_Bank.v.data = args.mem_momentum * Memory_Bank.v.data + g #+ args.mem_wd * Memory_Bank.W.data
        Memory_Bank.W.data = Memory_Bank.W.data - memory_lr * Memory_Bank.v.data
    
    with torch.no_grad():
        logits = torch.softmax(logits1, dim=1)
        posi_prob = logits[torch.arange(logits.shape[0]), filter_index1]
        posi_prob = torch.mean(posi_prob)
        mem_losses.update(posi_prob.item(), logits.size(0))
    
    return logits2, logits1, q_pred, KL_loss.item(), KL_alpha, labels1.tolist()


class x_hot_CrossEntropy(torch.nn.Module):
    def __init__(self):
        super(x_hot_CrossEntropy,self).__init__()
    
    def forward(self, x ,y):
        P_i = torch.nn.functional.softmax(x, dim=1)
        # y = torch.nn.functional.softmax(y, dim=1)
        # print(y[0])
        loss = y*torch.log(P_i + 0.0000001)
        loss = -torch.mean(torch.sum(loss,dim=1),dim = 0)
        return loss
    