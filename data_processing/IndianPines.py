import random

import torch
import numpy as np
from scipy.io import loadmat
from sklearn import preprocessing

# 定义DATASETS_CONFIG字典是为了保存数据集的图片和标签文件的文件名，用于在56、57行处获取文件的具体路径
DATASETS_CONFIG = { # dict{'key':'value'}
    'PaviaU': { 
        'img': 'PaviaU.mat',   # 图片文件
        'gt': 'PaviaU_gt.mat'  # 标签文件，gt就是ground truth(labels)
    },
    'KSC': {
        'img': 'KSC.mat',
        'gt': 'KSC_gt.mat'
    },
    'IndianPines': {
        'img': 'indian_pines.mat',
        'gt': 'indian_pines_gt.mat'
    },
    'Salinas': {
        'img': 'salinas.mat',
        'gt': 'salinas_gt.mat'
    },
    'Houston': {
        'img': 'Houston.mat',
        'gt': 'Houston_gt.mat'
    },
    'Botswana': {
        'img': 'Botswana.mat',
        'gt': 'Botswana_gt.mat'
    },
    'WHU-Hi-HanChuan':{
        'img': 'WHU_Hi_HanChuan.mat',
        'gt': 'WHU_Hi_HanChuan_gt.mat'
    },
    'HyRANK':{
        'img': 'HyRANK.mat',
        'gt': 'HyRANK_GT.mat'
    }
}

# 四个参数：日志，数据集名，数据集所在目录，数据集文件名
def get_dataset(dataset_name, target_folder, datasets=DATASETS_CONFIG):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """

    #如果数据集不在项目范围内，报错
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    #获取数据集路径
    img_file = target_folder + '/' + datasets[dataset_name].get('img')
    gt_file = target_folder + '/' + datasets[dataset_name].get('gt')
    print('img_file, gt_file',img_file, gt_file)
    
    #因为每种数据集的标签各不相同，比如印第安松林有16个类别，故label_values保存为17个值的数组，Undefined表示不属于任何类（黑点）
    if dataset_name == 'Houston':
        # 加载图片
        img = loadmat(img_file)['Houston']
        # print(img)
        gt = loadmat(gt_file)['Houston_gt']

        label_values = ["Undefined", "Healthy grass", "Stressed grass", " Synthetic grass",
                        "Trees", "Soil", "Water", "Residential", "Commercial", "Road",
                        "Highway", "Railway", "Parking Lot 1", "Parking Lot 2",
                        "Tennis Court", "Running Track"]

        ignored_labels = [0]

    elif dataset_name == 'PaviaU':
        # Load the image
        img = loadmat(img_file)['paviaU']

        gt = loadmat(gt_file)['Data_gt']

        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']

        ignored_labels = [0]

    elif dataset_name == 'IndianPines':
        # Load the image
        print("已加载IndianPines数据集！")
        img = loadmat(img_file)
        img = img['HSI_original']

        gt = loadmat(gt_file)['Data_gt']
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]

        ignored_labels = [0]

    elif dataset_name == 'Botswana':
        # Load the image
        img = loadmat(img_file)['Botswana']

        gt = loadmat(gt_file)['Botswana_gt']
        label_values = ["Undefined", "Water", "Hippo grass",
                        "Floodplain grasses 1", "Floodplain grasses 2",
                        "Reeds", "Riparian", "Firescar", "Island interior",
                        "Acacia woodlands", "Acacia shrublands",
                        "Acacia grasslands", "Short mopane", "Mixed mopane",
                        "Exposed soils"]

        ignored_labels = [0]

    elif dataset_name == 'KSC':
        # Load the image
        img = loadmat(img_file)['KSC']

        gt = loadmat(gt_file)['KSC_gt']
        label_values = ["Undefined", "Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]

        ignored_labels = [0]

    elif dataset_name == 'Salinas':
        # Load the image
        img = loadmat(img_file)['HSI_original']

        gt = loadmat(gt_file)['Data_gt']
        label_values = ["Undefined", "Brocoli green weeds 1", "Brocoli_green_weeds_2",
                        "Fallow", "Fallow rough plow", "Fallow smooth", "Stubble",
                        "Celery", "Grapes untrained", "Soil vinyard develop",
                        "Corn senesced green weeds", "Lettuce romaine 4wk",
                        "Lettuce romaine 5wk", "Lettuce romaine 6wk", "Lettuce romaine 7wk",
                        "Vinyard untrained", "Vinyard vertical trellis"]

        ignored_labels = [0]

    elif dataset_name == 'WHU-Hi-HanChuan':
        # Load the image
        img = loadmat(img_file)['WHU_Hi_HanChuan']

        gt = loadmat(gt_file)['WHU_Hi_HanChuan_gt']
        label_values = ["Undefined", "Strawberry", "Cowpea",
                        "Soybean", "Sorghum", "Water spinach", "Watermelon",
                        "Greens", "Trees", "Grass",
                        "Red roof", "Gray roof",
                        "Plastic", "Bare soil", "Road",
                        "Bright object", "Water"]

        ignored_labels = [0]

    elif dataset_name == 'HyRANK':
        # Load the image
        img = loadmat(img_file)['Dioni']
        gt = loadmat(gt_file)['Dioni_GT']
        label_values = ["Undefined", "Dense urban fabric", "Mineral extraction site",
                        "Non-irrigated arable land", "Fruit trees", "Olive groves", "Broad-leaved forest",
                        "Coniferous forest", "Mixed forest", "Dense sclerophyllous vegetation",
                        "Sparce sclerophyllous vegetation", "Sparsely vegetated areas",
                        "Rocks and sand", "Water", "Coastal water"]

        ignored_labels = [0]
    

    # 检查像元是否为空:一个像素对应1x1x200（波段）个数据，把这200个数据加起来看看等不等于0来判断像素有没有缺失

    # isnan判断是否为零，返回True/False，axis=-1表示选取倒数第一个下标所在维度，此处为200
    nan_mask = np.isnan(img.sum(axis=-1))
    # print(nan_mask)
    # print(np.count_nonzero(nan_mask))
    # count_nonzero统计非零元素的个数
    if np.count_nonzero(nan_mask) > 0:
        logger.info("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN "
              "data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)
    print("数据集形状：",np.shape(img))
    print("ignored_labels:",ignored_labels)

    ignored_labels = list(set(ignored_labels))
    # 做标准化??
    # asarray将数据转化为数组，dtype决定元素的数据类型（float型数据在数组中会表示为右下多一个点，如3200.）
    img = np.asarray(img, dtype='float32')
    
    # X=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]])
    # 则X[2,:]表示取第三行，X[:,2]表示取第三列，X[2:]表示取第三行及之后所有行，X[:2]表示取第三列及之后所有列
    data = img.reshape(np.prod(img.shape[:2]), np.prod(img.shape[2:]))
    
    data = preprocessing.minmax_scale(data, axis=1)
    
    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    
    data = scaler.fit_transform(data)
    
    img = data.reshape(img.shape) 
    
    return img, gt, label_values, ignored_labels

# 传入的参数依次是高光谱图像img, 训练集标签train_gt, 超参数字典hyperparams
class Hyper2X(torch.utils.data.Dataset):
    # 处理高光谱数据的一个工具包
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(Hyper2X, self).__init__()
        self.flip_augmentation = hyperparams['flip_augmentation']
        # self.channel_slice = hyperparams['channel_slice']
        if hyperparams['flip_augmentation']:
            data_copy = data.copy()
            gt_copy = gt.copy()
            for i in range(1):  # range(1)生成一个positive sample
                data = np.hstack((data, data_copy))
                gt = np.hstack((gt, gt_copy))
        self.data = data
        self.label = gt
        self.classes = hyperparams['n_classes']
        self.dataset_name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        # self.center_pixel = hyperparams['center_pixel']
        self.center_pixel = True
        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            # ones_like返回一个用1填充的跟输入形状和类型一致的数组
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
            
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        # 丢弃了那些图像边缘的点（以所取像素为中心点，若该patch大小覆盖了图像边缘之外的部分，该像素丢弃）
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)
        self.count = len(self.labels) // 2

    #class C(object):
    #   @staticmethod
    #   def f(arg1, arg2, ...):
    #       ...
    # 使用@staticmethod声明了静态方法 f，从而可以实现实例化使用 C().f()，当然也可以不实例化调用该方法 C.f()
    
    @staticmethod
    def ud_flip(data, label):
        data = np.flipud(data)
        label = np.flipud(label)
        return data, label

    @staticmethod
    def lr_flip(data, label):
        data = np.fliplr(data)
        label = np.fliplr(label)
        return data, label

    @staticmethod
    def trans_flip(data, label):
        data = data.transpose((1, 0, 2))
        label = label.transpose((1, 0))
        return data, label

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        data0 = self.data[x1:x2, y1:y2]
        label0 = self.label[x1:x2, y1:y2]

        idx = i + 1
        if idx >= len(self.indices):
            idx = i - 1
        w, z = self.indices[idx]
        w1, z1 = w - self.patch_size // 2, z - self.patch_size // 2
        w2, z2 = w1 + self.patch_size, z1 + self.patch_size
        data1 = self.data[x1:x2, y1:y2]
        label1 = self.label[x1:x2, y1:y2]

        # 选择是否需要数据增强
        if self.flip_augmentation:
            if (i > self.count) & (i <= self.count * 2):
                data1, label1 = self.ud_flip(data0, label0)
            elif (i > self.count * 2) & (i <= self.count * 3):
                data1, label1 = self.lr_flip(data0, label0)
            elif i > self.count * 3:
                data1, label1 = self.trans_flip(data0, label0)

        # if self.flip_augmentation:
        #     data0, label0 = self.ud_flip(data0, label0)
        #     data1, label1 = self.lr_flip(data1, label1)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        # 把通道维放到了最前面
        data0 = np.asarray(
            np.copy(data0).transpose(
                (2, 0, 1)), dtype='float32')
        label0 = np.asarray(np.copy(label0), dtype='int64')
        data1 = np.asarray(
            np.copy(data1).transpose(
                (2, 0, 1)), dtype='float32')
        label1 = np.asarray(np.copy(label1), dtype='int64')

        # Load the data into PyTorch tensors
        data0 = torch.from_numpy(data0)
        label0 = torch.from_numpy(label0)
        data1 = torch.from_numpy(data1)
        label1 = torch.from_numpy(label1)
        # Extract the center label if needed
        # 取出中心点像素的标签
        if self.center_pixel and self.patch_size > 1:
            label0 = label0[self.patch_size // 2, self.patch_size // 2]
            label1 = label1[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data0 = data0[:, 0, 0]
            label0 = label0[0, 0]
            data1 = data1[:, 0, 0]
            label1 = label1[0, 0]

        '''
        # Add a fourth dimension for 3D CNN
        if self.patch_size > 1:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data0 = data0.unsqueeze(0)  # uncommon if need.
            data1 = data1.unsqueeze(0)  # uncommon if need.
        '''
        # torch.Size([103, 11, 11]), torch.Size([]), torch.Size([103, 11, 11]), torch.Size([])

        # 把增强操作放到这里
        # 选择是否需要翻转增强
        
        # 原0.2
        data0_std = torch.from_numpy(np.random.normal(1, 0.5, size=data0.size())).float()
        data0 = data0 * data0_std
        data1_std = torch.from_numpy(np.random.normal(1, 0.5, size=data0.size())).float()
        data1 = data1 * data1_std

        # data0_std = torch.from_numpy(np.random.normal(1, 0.025, size=data0.size())).float()
        # data0 = data0 * data0_std
        # data1_std = torch.from_numpy(np.random.normal(1, 0.075, size=data0.size())).float()
        # data1 = data1 * data1_std
        return data0, label0, data1, label1

# 传入的参数依次是高光谱图像img, 训练集标签train_gt, 超参数字典hyperparams
class HyperX(torch.utils.data.Dataset):
    # 处理高光谱数据的一个工具包
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.data = data
        self.label = gt
        self.dataset_name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']
        self.classes = hyperparams['n_classes']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        # self.center_pixel = hyperparams['center_pixel']
        self.center_pixel = True
        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            # ones_like返回一个用1填充的跟输入形状和类型一致的数组
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
            
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        # 丢弃了那些图像边缘的点（以所取像素为中心点，若该patch大小覆盖了图像边缘之外的部分，该像素丢弃）
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    #class C(object):
    #   @staticmethod
    #   def f(arg1, arg2, ...):
    #       ...
    # 使用@staticmethod声明了静态方法 f，从而可以实现实例化使用 C().f()，当然也可以不实例化调用该方法 C.f()
    @staticmethod
    def get_data(data, x, y, patch_size, data_3D=False):
        x1, y1 = x - patch_size // 2, y - patch_size // 2
        x2, y2 = x1 + patch_size, y1 + patch_size
        data = data[x1:x2, y1:y2]
        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        # Add a fourth dimension for 3D CNN
        if data_3D:
            # Make 4D data ((Batch x) Planes x Channels x Width x Height)
            data = data.unsqueeze(0)  # uncommon if need.
        #print('开始打印1len(train_data)',len(data)) # 200
        return data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        data = self.get_data(self.data, x, y, self.patch_size, data_3D=False)
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        label = self.label[x1:x2, y1:y2]
        label = np.asarray(np.copy(label), dtype='int64')
        label = torch.from_numpy(label)
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        #print('开始打印2len(train_data)',len(data)) # 200
        return data, label