# LSCoSa (TGRS2023)
LSCoSa[https://doi.org/10.1109/TGRS.2023.3331888] is a contrastive-learning based self-supervised learning method for HSIC followed in [CaCo: Both Positive and Negative Samples are Directly Learnable via Cooperative-adversarial Contrastive Learning](https://arxiv.org/abs/2203.14370). 


<p align="center">
  <img src="https://github.com/sakurashine/LSCoSa/blob/main/LSCoSa_Motivation.png" >
</p>

## Introduction

When dealing with hard samples in HSIC, instance-level alignment with excessive uniformity may descend into trivial clusters, especially when confronted with inter-class similarity and intra-class diversity in hyperspectral images. To solve this problem, we regard prototypical contrastive learning as tracing the potential probability density distribution. Then, we propose a novel pre-training method, Learnable Sparse Contrastive Sampling (LSCoSa), for discriminative representation learning. Firstly, we maintain a learnable dictionary of positive and negative samples, which is learned cooperatively and adversarially with contrastive loss, to perform effective discriminative supervision. Secondly, we exert a KL divergence regularizer on the average activation probability of the prototypes in the dictionary, preventing fake density clusters from activation and achieving sparse positive sampling. Furthermore, we propose multiple positives learning, in which the top-k potential positives are retrieved and dynamically weighted for contrastive supervision, to avoid trivial clusters and cover satisfying semantic variations. We conduct comprehensive experiments on three HSI benchmark datasets, where LSCoSa shows significant advantages over other HSIC methods, especially for linear classifier evaluation with few-shot samples.

<p align="center">
  <img src="https://github.com/sakurashine/LSCoSa/blob/main/LSCoSa_Framework.png" width="300">
</p>

## Usage

### 1. Unsupervised Pre-Training
We can run on a single machine of single GPU with the following command:
```
bash run_train.sh
```

### 2. Linear Classification
With a pre-trained model, we can easily evaluate its performance on IndianPines with:
```
bash run_test.sh
```

## Citation
lf you use LSCoSa code in your research, we would appreciate a citation to the original paper:
```
@ARTICLE{10314565,
  author={Liang, Miaomiao and Dong, Jian and Yu, Lingjuan and Yu, Xiangchun and Meng, Zhe and Jiao, Licheng},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Self-Supervised Learning With Learnable Sparse Contrastive Sampling for Hyperspectral Image Classification}, 
  year={2023},
  volume={61},
  number={},
  pages={1-13},
  doi={10.1109/TGRS.2023.3331888}}
```
If you have any questions or suggestions, welcome to contact me by email: dongj39@mail2.sysu.edu.cn
