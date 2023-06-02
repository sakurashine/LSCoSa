import torch
import torch.nn as nn
import numpy as np

class CaCo(nn.Module):
   
    def __init__(self, base_encoder,args, dim=128, m=0.999, N_BANDS=1):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(CaCo, self).__init__()
        self.args=args
        self.m = m
        # 创建encoder
        # num_classes 是fc层的输出纬度
        self.encoder_q = base_encoder(num_classes=dim, num_bands=N_BANDS)  # base_encoder是自定义的arch，其实就是res18
        self.encoder_k = base_encoder(num_classes=dim, num_bands=N_BANDS)
        dim_mlp = self.encoder_q.fc.weight.shape[1]
        # q、k支路的mlp层，可以考虑增加predictor层
        self.encoder_q.fc = self._build_mlp(3,dim_mlp,args.mlp_dim,dim,last_bn=False)  # build_mlp方法在下方自定义
        self.encoder_k.fc = self._build_mlp(3, dim_mlp, args.mlp_dim, dim,last_bn=False)
        self.mu = nn.Linear(args.mlp_dim, args.mlp_dim)  # 是不是应该改成self.encoder_q.mu?
        self.log_sigma = nn.Linear(args.mlp_dim, args.mlp_dim)
        
        #self.encoder_q.fc = self._build_mlp(2,dim_mlp,args.mlp_dim,dim,last_bn=True)
        #self.encoder_k.fc = self._build_mlp(2, dim_mlp, args.mlp_dim, dim, last_bn=True)
        #self.predictor = self._build_mlp(2,dim,args.mlp_dim,dim,last_bn=False)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # 初始化
            param_k.requires_grad = False  # 停止梯度更新

        self.K=args.cluster

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _momentum_update_key_encoder_param(self,moco_momentum):  # 更新key encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * moco_momentum + param_q.data * (1. - moco_momentum)

    
    # 1打印im_q.shape torch.Size([32, 52, 11, 11])
    # 2打印im_q.shape torch.Size([32, 64])
    def forward_withoutpred_sym(self,im_q,im_k,moco_momentum):
        # print("**********",self.encoder_q,im_q.shape)
        q = self.encoder_q(im_q, use_feature=False)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        q_pred = q
        k_pred = self.encoder_q(im_k, use_feature=False)  # queries: NxC
        k_pred = nn.functional.normalize(k_pred, dim=1)
        with torch.no_grad():  # no gradient to keys
                # if update_key_encoder:
            self._momentum_update_key_encoder_param(moco_momentum)# update the key encoder
            q = self.encoder_k(im_q, use_feature=False)  # keys: NxC
            # （弃用）对q添加VAE的变分,经测试方差取0.1，0.025，0.01时分类效果均不佳，不如在update_sym_network处加
            # q_mu=self.mu(q)
            # q_log_sigma=self.log_sigma(q)
            # q_std_z = torch.from_numpy(np.random.normal(0, 0.01, size=q_log_sigma.size())).float()
            # q=q_mu + q_log_sigma * torch.tensor(q_std_z, requires_grad=False).cuda()
            
            q = nn.functional.normalize(q, dim=1)
            q = q.detach()
            
            k = self.encoder_k(im_k, use_feature=False)  # keys: NxC
            # （弃用）对k添加VAE的变分,经测试方差取0.1，0.025，0.01时分类效果均不佳，不如在update_sym_network处加
            # k_mu=self.mu(k)
            # k_log_sigma=self.log_sigma(k)
            # k_std_z = torch.from_numpy(np.random.normal(0, 0.01, size=k_log_sigma.size())).float()
            # k=k_mu + k_log_sigma * torch.tensor(k_std_z, requires_grad=False).cuda()
            
            k = nn.functional.normalize(k, dim=1)
            k = k.detach()
        # print("打印q、k及其shape",q_pred.shape, k_pred.shape, q.shape, k.shape)
        return q_pred, k_pred, q, k
    
    # 未使用
    def forward_withoutpred_multicrop(self,im_q_list,im_k,moco_momentum):
        q_list = []
        for im_q in [im_k]+im_q_list:
            q = self.encoder_q(im_q, use_feature=False)  # queries: NxC
            q = nn.functional.normalize(q, dim=1)
            q_list.append(q)
        key_list = []
        with torch.no_grad():  # no gradient to keys
                # if update_key_encoder:
            self._momentum_update_key_encoder_param(moco_momentum)# update the key encoder
            for key_image in [im_k]+im_q_list[:1]:
                q = self.encoder_k(key_image, use_feature=False)  # keys: NxC
                q = nn.functional.normalize(q, dim=1)
                q = q.detach()
                key_list.append(q)
        return q_list,key_list

    def forward(self, im_q, im_k,run_type=0,moco_momentum=0.999):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            type: 0:sym; 1:multicrop
        Output:
            logits, targets

        """
        if run_type==0:
            return self.forward_withoutpred_sym(im_q,im_k,moco_momentum)
            
        elif run_type==1:
            return self.forward_withoutpred_multicrop(im_q, im_k, moco_momentum)
        
    def reparameterize(self, mu, log_var):
    
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std


# 模仿SGD update改写的方法
class CaCo_PN(nn.Module):
    # [memory bank的size， embedding的维度] [256,128]
    def __init__(self,bank_size,dim):
        super(CaCo_PN, self).__init__()
        # W是memory bank中的embedding，v是保持momentum的embedding
        self.register_buffer("W", torch.randn(dim, bank_size))
        self.register_buffer("v", torch.zeros(dim, bank_size))
    def forward(self,q):
        # query表征的维度是n*c，n是batch大小，c是特征维度
        # 字典中key的数量记为k，c指的是每个特征的维度，所以它的维度是c*k
        memory_bank = self.W
        memory_bank = nn.functional.normalize(memory_bank, dim=0)
        logit=torch.einsum('nc,ck->nk', [q, memory_bank])
        return memory_bank, self.W, logit
    def update(self, m, lr, weight_decay, g):
        print("打印CaCo_PN.update",m, lr, weight_decay, g)
        # g是针对positive和negative combined gradient
        g = g + weight_decay * self.W
        self.v = m * self.v + g
        self.W = self.W - lr * self.v
    def print_weight(self):
        print(torch.sum(self.W).item())

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
