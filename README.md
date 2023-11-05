# Self-Supervised Learning with Learnable Sparse Contrastive Sampling for Hyperspectral Image Classification
LSCoSa is a contrastive-learning based self-supervised learning method for HSIC followed in [CaCo: Both Positive and Negative Samples are Directly Learnable via Cooperative-adversarial Contrastive Learning](https://arxiv.org/abs/2203.14370). 

Contact: Jian Dong (Mr.DongJianjian@gmail.com)

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
