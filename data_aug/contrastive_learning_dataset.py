from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection

import os, torch
from glob import glob
from PIL import Image
import shutil
from skimage import io
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, store_path, folder, transform=None, download=True):
        self.folder = folder
        # self.transform = transforms.Compose(
        #     [
        #         # transforms.ToPILImage(), # ref: https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch
        #         # v2.Grayscale(),
        #         transforms.Resize((96,96)),
        #     ]
        # )
        self.transform = transform
        self.img_fpns = sorted([fn for fn in glob(os.path.join(folder, '*.*')) if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))])

        if download:
            os.makedirs(store_path, exist_ok=True)
            for remote_fpn in self.img_fpns:
                print('Downloading file:')
                print(remote_fpn)
                fn = os.path.basename(remote_fpn)
                local_fpn = os.path.join(store_path, fn)
                if not os.path.exists(local_fpn):
                    shutil.copyfile(remote_fpn, local_fpn)
            self.img_fpns = sorted([fn for fn in glob(os.path.join(store_path, '*.*')) if fn.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))])
    
    def __len__(self):
        return len(self.img_fpns)
    
    def __getitem__(self, index):
        img_fpn = self.img_fpns[index]
        res = transforms.Resize((96,96))(Image.fromarray(io.imread(img_fpn)))
        if self.transform:
            res = self.transform(res)
        return res, img_fpn
    
class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        # make "valid_datasets" lambda functions because you don't want to instantiate
        # all different datasets (which leads to download all images) when the
        # dictionary is defined
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder,
                                                          split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True),
                          'custom': lambda: MyDataset(self.root_folder, r'\\azatshfs.intel.com\AZATAnalysis$\MAOATM\ATTD_Yield\HBI\ANALYSIS\CSAM\output\post_process\D3330670_FK6VN801JKD6_B_margin0.050', transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views))
            }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
