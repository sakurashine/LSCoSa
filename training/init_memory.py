import warnings
import torch
import torch.nn 
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from training.train_utils import AverageMeter, ProgressMeter, accuracy
import warnings
warnings.filterwarnings("ignore")
warnings.warn("deprecated", DeprecationWarning)

def init_memory(train_loader, model,Memory_Bank, criterion,
                                optimizer,epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [losses, top1, top5],
        prefix="Init Epoch: [{}]".format(epoch))
    # switch to train mode
    model.train()
    for i, (data0, _, data1, _) in enumerate(train_loader):
        images = [data0, data1]
        if args.gpu is not None:
            for k in range(len(images)):  
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
        
        # compute output   
        q, _, _, k  = model(im_q=images[0], im_k=images[1])
        d_norm, d, l_neg = Memory_Bank(q)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= 0.2#using the default param in MoCo temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(args.gpu)
        loss = criterion(logits, labels)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1.item(), images[0].size(0))
        top5.update(acc5.item(), images[0].size(0))

        if i % args.print_freq == 0 and args.rank==0:
            progress.display(i)

        output = k
        batch_size = output.size(0)
        start_point = i * batch_size
        end_point = min((i + 1) * batch_size, args.cluster)
        Memory_Bank.W.data[:, start_point:end_point] = output[:end_point - start_point].T
        
        if (i+1) * batch_size >= args.cluster:
            break
    for param_q, param_k in zip(model.encoder_q.parameters(),
                                model.encoder_k.parameters()):
        param_k.data.copy_(param_q.data)  # initialize
