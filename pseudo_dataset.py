import csv
import numpy as np
from random import sample
from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import matplotlib.pyplot as plt
from PIL import Image
import csv
import cv2
from skimage import measure
import numpy as np
import timm
from model import backboneNet_efficient
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import math
from tqdm import tqdm
from torch.autograd import Variable
from sklearn import metrics
import csv
import numpy as np
import os
import pseudo_config as cfg


def Image_loader(path):
    return cv2.imread(path)

def crop_image_from_gray(img, tol=7):
    if img is None:
        print(img)
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img


class PseudoDataset(Dataset):
    def __init__(self, csv_file, file_id, transform=None, loader=Image_loader, mode=None, id_conversion=None):
        super(PseudoDataset, self).__init__()
        imgs = []
        with open(csv_file, newline='') as csvfile:
            rows = csv.DictReader(csvfile)
            for row in rows:
                if id_conversion[row['id_code']] in file_id:
                    if mode != 'pseudo':
                        img_path = './data/train_images/' + row['id_code'] + '.jpeg'
                    else:
                        img_path = './data/test_images/' + row['id_code'] + '.png'
                    label = row['diagnosis']
                    imgs.append((img_path, int(label)))   
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]

        img = self.loader(fn)
        if img is None:
            print(fn)
        img = crop_image_from_gray(img)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

    def __add__(self, other):
        return ConcatDataset([self, other])