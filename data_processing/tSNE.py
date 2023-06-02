from sklearn.manifold import TSNE
from time import time
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class T_sne_visual():
    def __init__(self, model, dataset, dataloader):
        self.model = model
        self.dataset = dataset
        self.dataloader = dataloader
        self.class_list=dataset.classes
    def visual_dataset(self):
        imgs = []
        labels = []
        for img, label, _, _ in self.dataset:
            imgs.append(np.array(img).transpose((2, 1, 0)).reshape(-1))
            tag = self.class_list[label]
            labels.append(tag)
        self.t_sne(np.array(imgs), labels,title=f'Dataset visualize result\n')

    def visual_feature_map(self, layer):
        self.model.eval()
        with torch.no_grad():
            self.feature_map_list = []
            labels = []
            getattr(self.model, layer).register_forward_hook(self.forward_hook)
            for img, label in self.dataloader:
                img=img.cuda()
                self.model(img)
                for i in label.tolist():
                    tag=self.class_list[i]
                    labels.append(tag)
            self.feature_map_list = torch.cat(self.feature_map_list,dim=0)
            self.feature_map_list=torch.flatten(self.feature_map_list,start_dim=1)
            self.t_sne(np.array(self.feature_map_list.cpu()), np.array(labels),title=f'{layer} resnet feature map\n')

    def forward_hook(self, model, input, output):
        self.feature_map_list.append(output)

    def set_plt(self, start_time, end_time,title):
        plt.title(f'{title} time consume:{end_time - start_time:.3f} s')
        plt.legend(title='')
        plt.ylabel('')
        plt.xlabel('')
        plt.xticks([])
        plt.yticks([])

    def t_sne(self, data, label,title):
        # t-sne处理
        print('starting T-SNE process')
        start_time = time()
        data = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(data)
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        df = pd.DataFrame(data, columns=['x', 'y'])  # 转换成df表
        df.insert(loc=1, column='label', value=label)
        end_time = time()
        print('Finished')

        # 绘图
        sns.scatterplot(x='x', y='y', hue='label', s=3, palette="Set2", data=df)
        self.set_plt(start_time, end_time, title)
        plt.savefig('1.jpg', dpi=400)
        plt.show()