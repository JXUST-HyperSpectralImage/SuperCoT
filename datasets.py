# --------------------------------------------------------
# Original Code from BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Modified for implementation of Masked Image Modeling with Denoising Contrast(https://arxiv.org/abs/2205.09616)
# By Kun Yi
# --------------------------------------------------------'
import torch

from torchvision import datasets, transforms
from utils import open_file
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from transforms import RandomResizedCropAndInterpolationWithTwoPic
from timm.data import create_transform
import cv2
from masking_generator import MaskingGenerator, RandomMaskingGenerator, MaskGenerator
from dataset_folder import ImageFolder
from imagefromlist import ImageFromList
from PIL import ImageFilter, ImageOps, Image
import random
import numpy as np
from sklearn import preprocessing
from scipy.io import loadmat

from sklearn.preprocessing import scale, minmax_scale, normalize
from skimage.segmentation import slic, mark_boundaries, find_boundaries
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class DataAugmentationForConMIM(object):
    def __init__(self, args):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        self.common_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=args.input_size, second_size=args.second_input_size,
                interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
            ),
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        self.patch_transform_hard = transforms.Compose([
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(0.1),
            Solarization(0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        if args.mask_type == "block":
            self.masked_position_generator = MaskingGenerator(
                args.window_size, num_masking_patches=args.num_mask_patches,
                max_num_patches=args.max_mask_patches_per_block,
                min_num_patches=args.min_mask_patches_per_block,
            )
        elif args.mask_type == 'random_mps32':
            self.masked_position_generator = MaskGenerator(mask_ratio=args.mask_ratio)      
        self.mask_type = args.mask_type

    def __call__(self, image):
        for_patches, for_visual_tokens = self.common_transform(image)
        return \
            self.patch_transform(for_patches), self.patch_transform_hard(for_patches), self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForConMIM,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  patch_transform_hard = %s,\n" % str(self.patch_transform_hard)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr





def build_conmim_pretraining_dataset(args):
    transform = DataAugmentationForConMIM(args)
    print("Data Aug = %s" % str(transform))
    if args.data_set == 'IMNET':
        # Read from List
        return ImageFromList(args.data_path + '/train', args.data_path + '/train_map.txt', transform=transform)

    else:
        return datasets.CIFAR10(root=args.data_path, download=True,transform=transform) 


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    try:
        print("Transform = ")
        if isinstance(transform, tuple):
            for trans in transform:
                print(" - - - - - - - - - - ")
                for t in trans.transforms:
                    print(t)
        else:
            for t in transform.transforms:
                print(t)
        print("---------------------------")
    except:
        pass
    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        if is_train:
            dataset = ImageFromList(args.data_path + '/train', args.data_path + '/train_map.txt', transform=transform)
        else:
            dataset = ImageFromList(args.data_path + '/val', args.data_path + '/val_map.txt', transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

# Hyperspectral
DATASETS_CONFIG = {
    'PaviaU': {
        'img': 'PaviaU.mat',
        'gt': 'PaviaU_gt.mat'
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
    'HyRANK':{
        'img': 'HyRANK.mat',
        'gt': 'HyRANK_GT.mat'
    },
    'Xuzhou': {
        'img': 'xuzhou.mat',
        'gt': 'xuzhou_gt.mat'
    }
}



def get_dataset(dataset_name, target_folder="../dataset/", datasets=DATASETS_CONFIG):
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    img_file = target_folder + dataset_name + '/' + datasets[dataset_name].get('img')
    gt_file = target_folder + dataset_name + '/' + datasets[dataset_name].get('gt')
    if dataset_name == 'Houston':
        # Load the image
        img = loadmat(img_file)['Houston']
        gt = loadmat(gt_file)['Houston_gt']
        label_values = ["Undefined", "Healthy grass", "Stressed grass", " Synthetic grass",
                        "Trees", "Soil", "Water", "Residential", "Commercial", "Road",
                        "Highway", "Railway", "Parking Lot 1", "Parking Lot 2",
                        "Tennis Court", "Running Track"]
        ignored_labels = [0]

    elif dataset_name == 'PaviaU':
        # Load the image
        img = loadmat(img_file)['paviaU'][:, :, :-1] 
        gt = loadmat(gt_file)['Data_gt']
        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
        ignored_labels = [0]

    elif dataset_name == 'IndianPines':
        # Load the image
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

    elif dataset_name == 'Xuzhou':                             
        # Load the image
        img = loadmat(img_file)['xuzhou']
        gt = loadmat(gt_file)['xuzhou_gt']
        label_values = ["Undefined", "BARELAND1", "LAKES", "COALS", "CEMENT", "CROPS-1", "TRESS", "BARELAND2", "CROPS", "RED-TITLE"]  
        ignored_labels = [0]
        
    nan_mask = np.isnan(img.sum(axis=-1))   
    if np.count_nonzero(nan_mask) > 0:
        logger.info("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN "
              "data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))  
    # Normalization
    img = np.asarray(img, dtype='float32')  

    print("得到图像的大小：", img.shape)
    data = img.reshape(np.prod(img.shape[:2]), np.prod(img.shape[2:])) 
    data = preprocessing.minmax_scale(data, axis=1)
    scaler = preprocessing.StandardScaler() 
    scaler.fit(data) 
    data = scaler.fit_transform(data)  
    img = data.reshape(img.shape) 
    return img, gt, label_values, ignored_labels 

def HSI_to_superpixels(img, num_superpixel, is_pca=False, is_show_superpixel=False):
    n_row, n_col, n_band = img.shape
    if is_pca:    
        pca = PCA(n_components=0.95)
        img = pca.fit_transform(scale(img.reshape(-1, n_band))).reshape(n_row, n_col, -1)

    superpixel_label = slic(img, n_segments=num_superpixel, compactness=20, max_iter=10, convert2lab=False,
                            enforce_connectivity=True, min_size_factor=0.3, max_size_factor=2, slic_zero=False)


    superpixel_label = superpixel_label.astype(np.int64)
    return superpixel_label    #返回超像素标签（与输入图像大小一致）


class HyperX(torch.utils.data.Dataset):#添加超像素
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, sp_labels, superpixel_pca=True, is_superpixel=True, path_to_sp=None, **hyperparams):
        super(HyperX, self).__init__()
        self.data = data        
        self.label = gt        
        self.dataset_name = hyperparams['dataset']      
        self.patch_size = hyperparams['patch_size']     
        self.ignored_labels = set(hyperparams['ignored_labels'])        
        self.center_pixel = hyperparams['center_pixel']    
        supervision = hyperparams['supervision']     
        self.n_bands = hyperparams['n_bands']           
        self.mask_ratio = hyperparams['mask_ratio']
        self.sp_labels = sp_labels

        if supervision == 'full':   
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        elif supervision == 'semi':  
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)   
        p = self.patch_size // 2        
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if  
                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x, y] for x, y in self.indices]   
                
        np.random.shuffle(self.indices)




    @staticmethod
    def get_data(data, x, y, patch_size, data_3D=False):
        x1, y1 = x - patch_size // 2, y - patch_size // 2
        x2, y2 = x1 + patch_size, y1 + patch_size
        data = data[x1:x2, y1:y2]

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)    
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')  #hwc ->chw
        data0 = np.asarray(np.copy(data), dtype='float32')  #chw
        data1 = np.asarray(np.copy(data), dtype='float32')  #chw
        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        data0 = torch.from_numpy(data0)
        data1 = torch.from_numpy(data1)
        # Add a fourth dimension for 3D CNN
        if data_3D:
            data = data.unsqueeze(0)  # uncommon if need.
        return data,data0,data1   

    def __len__(self):
        return len(self.indices)
                               
    def __getitem__(self, i):   
        x, y = self.indices[i]      
        data,data0,data1 = self.get_data(self.data, x, y, self.patch_size, data_3D=False)
        
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        label = self.label[x1:x2, y1:y2]        
        label0 = np.asarray(np.copy(label), dtype='int64')
        label0 = torch.from_numpy(label0)   

        super_labels = self.sp_labels[x1:x2, y1:y2]  
        super_labels = torch.from_numpy(super_labels)

        data0_std = torch.from_numpy(np.random.normal(1, 0.2, size=data0.size())).float()
        data0 = data0 * data0_std
        data1_std = torch.from_numpy(np.random.normal(1, 0.5, size=data0.size())).float()
        data1 = data1 * data1_std
        
        patch = int(self.patch_size ** 2)  
        num_mask = int(self.mask_ratio * patch)
        mask = np.hstack([
                np.zeros(patch - num_mask),
                np.ones(num_mask),
            ])
        np.random.shuffle(mask)
        
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        return data0, data1, mask, super_labels
  

class HyperF(torch.utils.data.Dataset):    
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
        super(HyperF, self).__init__()
        self.data = data     
        self.label = gt        
        self.dataset_name = hyperparams['dataset']      
        self.patch_size = hyperparams['patch_size']    
        self.ignored_labels = set(hyperparams['ignored_labels'])    
        self.center_pixel = hyperparams['center_pixel'] 
        supervision = hyperparams['supervision']       

        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)    
        p = self.patch_size // 2      
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos) if
                                 x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x, y] for x, y in self.indices]
                
        np.random.shuffle(self.indices)

    @staticmethod
    def get_data(data, x, y, patch_size, data_3D=False):
        x1, y1 = x - patch_size // 2, y - patch_size // 2
        x2, y2 = x1 + patch_size, y1 + patch_size
        data = data[x1:x2, y1:y2]
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        # Add a fourth dimension for 3D CNN
        if data_3D:
            data = data.unsqueeze(0)  # uncommon if need.
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
        return data, label


