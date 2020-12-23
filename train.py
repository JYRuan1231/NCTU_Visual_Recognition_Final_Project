from model import backboneNet_efficient
import matplotlib.pyplot as plt
from model import backboneNet_efficient
import numpy as np
import csv
import cv2
import os
import copy
import math
import time
from PIL import Image

from random import sample
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm
from sklearn import metrics
from skimage import measure
from model import backboneNet_efficient
import config

def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img


def scale_radius(src, img_size, padding=False):
    x = src[src.shape[0] // 2, ...].sum(axis=1)
    r = (x > x.mean() / 10).sum() // 2
    yx = src.sum(axis=2)
    region_props = measure.regionprops((yx > yx.mean() / 10).astype('uint8'))
    yc, xc = np.round(region_props[0].centroid).astype('int')
    x1 = max(xc - r, 0)
    x2 = min(xc + r, src.shape[1] - 1)
    y1 = max(yc - r, 0)
    y2 = min(yc + r, src.shape[0] - 1)
    dst = src[y1:y2, x1:x2]
    dst = cv2.resize(dst, dsize=None, fx=img_size/(2*r), fy=img_size/(2*r))
    if padding:
        pad_x = (img_size - dst.shape[1]) // 2
        pad_y = (img_size - dst.shape[0]) // 2
        dst = np.pad(dst, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), 'constant')
    return dst


def Image_loader(path):
    return cv2.imread(path)

def get_transforms(Set):
    if Set == 'train':
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomAffine(
                degrees=(-180, 180),
                scale=(1, 1.3),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=(0.2), 
                                   contrast=(0.2),
                                   hue=(0.1),
                                   saturation=(0.2),
                                  ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
        ])
        return transform

    else:
        transform = transforms.Compose([
            transforms.Resize((280, 280)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform
    
def get_labels(Set):
    # config
    data_path = config.data_path
    train_label_name = config.train_label_name
    test_label_name = config.test_label_name
    
    
    if Set == 'train':
        labels = pd.read_csv(data_path + train_label_name)
        labels = np.squeeze(labels.values)

        return labels[:,0],  labels[:,1]

    else:
        names = pd.read_csv(data_path + test_label_name)
        names = np.squeeze(names.values)
        return labels, None


class RetinopathyDataset(Dataset):
    def __init__(self, Set='train'):
        # config
        data_path = config.data_path
        
        super(RetinopathyDataset, self).__init__()
        assert Set in ['train', 'test']
        self.Set = Set
        self.imgpath = data_path + Set + '_images/'
       
        self.input_transform = get_transforms(Set)
        self.image_names, self.labels = get_labels(Set)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_name = self.imgpath + image_name + '.jpeg'
        
        img = Image.open(image_name).convert('RGB')

        img = self.input_transform(img)

        if self.labels is not None:
            label = self.labels[index]
            return img, label

        else:
            return img
        
    def __add__(self, other):
        return ConcatDataset([self, other])
    def __len__(self):
        return len(self.image_names)

# class MyDataset(Dataset):
#     def __init__(self, csv_file, file_id, transform = None, loader=Image_loader):
#         imgs = []
#         with open(csv_file, newline='') as csvfile:
#             rows = csv.DictReader(csvfile)
#             for row in rows:
#                 if id_conversion[row['id_code']] in file_id:
#                     img_path = './data/train_images/'+row['id_code']+'.jpeg'
#                     label = row['diagnosis']
#                     imgs.append((img_path,int(label))) 
#         super(MyDataset, self).__init__()
#         self.imgs = imgs
#         self.transform = transform
#         self.loader = loader

#     def __getitem__(self, index):
#         fn, label = self.imgs[index]
#         img = self.loader(fn)
#         img = crop_image_from_gray(img)  
#         img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) 
#         img = self.transform(img)
#         return img,label

#     def __len__(self):
#         return len(self.imgs)

def quadratic_weighted_kappa(y_pred, y_true):
    return metrics.cohen_kappa_score(y_pred, y_true, weights='quadratic')



class BinaryFocalLoss(nn.Module):
    def __init__(self, device):
        super(BinaryFocalLoss, self).__init__()
        
        self.alpha = torch.Tensor([1, 3]).to(device)
        self.gamma = 2
        self.device = device
        self.epsilon = 1e-9
        
    def forward(self, prob, target):
        pt = torch.where(target == 1, prob, 1 - prob)
        alpha = torch.where(target == 1, self.alpha[1], self.alpha[0])
        
        log_pt = torch.log(pt + self.epsilon)
        focal = (1 - pt).pow(self.gamma)
        
        focal_loss = -1 * alpha * focal * log_pt
        
        return focal_loss.mean()


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i

    return num

def Average(lst): 
  
    return  sum(lst) / len(lst)



class FocalLoss(nn.Module):
 
    def __init__(self, num_class=5, alpha=.25, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average
 
        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')
 
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
 
    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
 
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)
 
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)
 
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)
 
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()
 
        gamma = self.gamma
 
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
 
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
 

    
def train_model(model, criterion, optimizer, scheduler,train_loader, valid_loader, num_epochs=25):
    # config
    
    accumulation_steps = config.accumulation_steps
    
    # initialization
    ordinal_labels = torch.FloatTensor([[0, 0, 0, 0],
                                       [1, 0, 0, 0],
                                       [1, 1, 0, 0],
                                       [1, 1, 1, 0],
                                       [1, 1, 1, 1]]).to(device)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_loss, best_valid_loss = 10000.0, 10000.0
    best_train_acc, best_valid_acc = 0.0, 0.0
    best_train_score, best_valid_score = 0.0, 0.0
    best_train_rg_acc, best_valid_rg_acc = 0.0, 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                model_loader = train_loader
                dataset_size = len(train_loader)
            elif phase == 'val':
                model.eval()   # Set model to evaluate mode
                model_loader = valid_loader
                dataset_size = len(valid_loader)

            train_loss, valid_loss = 0.0, 0.0
            train_cls_acc, valid_cls_acc = 0, 0
            train_score, valid_score = 0.0, 0.0
            train_rg_acc, valid_rg_acc = 0.0, 0.0
            y1_score, y2_score = [], []
            # Iterate over data.
            if phase == 'train':
                for iter, (inputs, targets) in tqdm(enumerate(model_loader), total=dataset_size):
                    inputs, targets = inputs.to(device), targets.to(device)

                    # forward
                    outputs = model(inputs)
                     
                    rg_outputs = outputs[0]
                    rg_outputs = torch.sigmoid(rg_outputs) * 4.5 
                    
                    cls_outputs = outputs[1]
                    cls_softmax = torch.nn.Softmax(dim=1)
                    cls_outputs = cls_softmax(cls_outputs)
                                                                           
                    ord_outputs = outputs[2]
                    ord_outputs = torch.sigmoid(ord_outputs)
                    
                    preds = torch.max(cls_outputs, 1)
                    
                    # loss 
                    rg_loss = criterion['regression'](rg_outputs.view(-1), targets.float())
                    cls_loss = criterion['classification'](cls_outputs, targets)        
                    ord_loss = criterion['ordinal'](ord_outputs, ordinal_labels[targets])

                    # gradient accumulation
                    loss = rg_loss + cls_loss + ord_loss
                    loss = loss / accumulation_steps
                    loss.backward()
                    
                    # Update weights when times of accumulation meets accumulation_steps
                    if (iter+1) % accumulation_steps == 0:
                        optimizer.step()      # Update weights
                        optimizer.zero_grad() # zero the parameter gradients
                        
                    # Post-processing
                    outputs = rg_outputs.unsqueeze(1)    
                    thrs = [0.5, 1.5, 2.5, 3.5]
                    outputs[outputs < thrs[0]] = 0
                    outputs[(outputs >= thrs[0]) & (outputs < thrs[1])] = 1
                    outputs[(outputs >= thrs[1]) & (outputs < thrs[2])] = 2
                    outputs[(outputs >= thrs[2]) & (outputs < thrs[3])] = 3
                    outputs[outputs >= thrs[3]] = 4   
                
                    # Record predictions and ground truth labels 
                    y1_score = y1_score + outputs[:,0].squeeze(1).tolist()
                    y2_score = y2_score + targets.tolist()
                   
                    # Statistics
                    train_loss += loss.item() * accumulation_steps * inputs.size(0)
                                       
                    train_rg_acc += torch.sum(outputs[:,0].squeeze(1) == targets.data)
                    
                    train_cls_acc += torch.sum(preds[1] == targets.data)
                    
                    if (iter+1) % 500 ==0:
                            print('{} iter:{} cls_loss: {:.4f} rg_loss {:.4f} ord_loss {:.4f}' 
                                  'cls_Acc: {:.4f} rg_Acc: {:.4f} Score {:.4f}'.format(phase, iter, cls_loss,rg_loss,ord_loss,                                 
                                                                                              train_cls_acc.double() / ((iter+1)*batch_size),
                                                                                              train_rg_acc.float() / ((iter+1)*batch_size),          
                                                                                              quadratic_weighted_kappa(y1_score, y2_score),
                                       
                                                                                             ))                  

                # Update optimizer
                scheduler.step()
                
                # log: training performance
                train_score = quadratic_weighted_kappa(y1_score, y2_score)

                epoch_loss = train_loss / dataset_size
                epoch_acc = train_cls_acc.double() / train_size
                epoch_rg_acc = train_rg_acc.double() / train_size
                epoch_score = train_score
                print('{} Loss: {:.4f}  cls_acc: {:.4f} rg_acc: {:.4f} Score: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_rg_acc, epoch_score))
                
                if(best_train_acc < epoch_acc):
                    best_train_acc = epoch_acc
                    
                if(best_train_rg_acc < epoch_rg_acc):
                    best_train_rg_acc = epoch_rg_acc    
                    
                if(best_train_loss > epoch_loss):
                    best_train_loss = epoch_acc
                    
                if(best_train_score < epoch_score):
                    best_train_score = epoch_score
                    
                if (epoch+1) % 5 == 0 and (epoch+1) >= 20:
                    epoch_model = copy.deepcopy(model.state_dict())
                    torch.save(epoch_model, './models/' + model_name+'_'+str(epoch+1))
                    print("save epoch training weight,complete!")


            # val and deep copy the model
            if phase == 'val':
                with torch.no_grad():
                    for iter, (inputs, targets) in tqdm(enumerate(model_loader), total=dataset_size):
                        inputs, targets = inputs.to(device), targets.to(device)
                        
                        # forward
                        outputs = model(inputs)
                        
                        rg_outputs = outputs[0]
                        rg_outputs = torch.sigmoid(rg_outputs) * 4.5                    
                        
                        cls_outputs = outputs[1]
                        cls_softmax = torch.nn.Softmax(dim=1)
                        cls_outputs = cls_softmax(cls_outputs)
                        
                        ord_outputs = outputs[2]
                        ord_outputs = torch.sigmoid(ord_outputs)
                        
#                         print(rg_outputs.tolist(),cls_outputs.tolist(),ord_outputs.tolist())
                        preds = torch.max(cls_outputs, 1)
                
                        # Loss
                        rg_loss = criterion['regression'](rg_outputs.view(-1), targets.float())
                        cls_loss = criterion['classification'](cls_outputs, targets)
                        ord_loss = criterion['ordinal'](ord_outputs, ordinal_labels[targets])

                        loss = rg_loss + cls_loss + ord_loss

                        # Post-processing
                        thrs = [0.5, 1.5, 2.5, 3.5]
                        
                        outputs = rg_outputs.unsqueeze(1)
                        outputs[outputs < thrs[0]] = 0
                        outputs[(outputs >= thrs[0]) & (outputs < thrs[1])] = 1
                        outputs[(outputs >= thrs[1]) & (outputs < thrs[2])] = 2
                        outputs[(outputs >= thrs[2]) & (outputs < thrs[3])] = 3
                        outputs[outputs >= thrs[3]] = 4   
                        
                        # Record predictions and ground truth labels 
                        y1_score = y1_score + outputs[:,0].squeeze(1).tolist()
                        y2_score = y2_score + targets.tolist()
                        
                        # statistics
                        valid_loss += loss.item() * inputs.size(0)
                            
                        valid_rg_acc += torch.sum(outputs[:,0].squeeze(1) == targets.data)    
                        valid_cls_acc += torch.sum(preds[1] == targets.data)
                        
                # log: validate training performance
                valid_score = quadratic_weighted_kappa(y1_score, y2_score)

                epoch_loss = valid_loss / dataset_size
                epoch_acc = valid_cls_acc.double() / valid_size
                epoch_rg_acc = valid_rg_acc.double() / valid_size
                epoch_score = valid_score
                print('{} Loss: {:.4f}  cls_acc: {:.4f} rg_Acc: {:.4f} Score: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_rg_acc, epoch_score))
                
                if(best_valid_acc < epoch_acc):
                    best_valid_acc = epoch_acc
                    
                if(best_valid_rg_acc < epoch_rg_acc):
                    best_valid_rg_acc = epoch_rg_acc

                if(best_valid_loss > epoch_loss):
                    best_valid_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())                    
                    if epoch+1 > 5:
                        torch.save(best_model_wts, './models/' + model_name+'_'+str(epoch+1))
                        print("save best training weight,complete!")  
                    
                if(best_valid_score < epoch_score):
                    best_valid_score = epoch_score
                
        time_elapsed = time.time() - since
        print('Complete one epoch in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        # training fininshed
        
    print()
    print('Best train acc: {:4f} Best val acc: {:4f}'.format(best_train_acc ,best_valid_acc))
    print('Best train rg acc: {:4f} Best val rg acc: {:4f}'.format(best_train_rg_acc ,best_valid_rg_acc))
    print('Best train loss: {:4f} Best val loss: {:4f}'.format(best_train_loss ,best_valid_loss))
    print('Best train score: {:4f} Best val score: {:4f}'.format(best_train_score ,best_valid_score))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    with torch.cuda.device(1):
        # config
        model_name = config.model_name
        result_path = config.result_path        
        
        # training parameters
        # data
        validation_percentage = config.validation_percentage
        batch_size = config.batch_size
        num_workers = config.num_workers
        optimizer_lr = config.optimizer_lr
        optimizer_momentum = config.optimizer_momentum
        optimizer_weight_decay = config.optimizer_weight_decay
        
        scheduler_exp_t0 = config.scheduler_exp_t0
        scheduler_exp_t_mult = config.scheduler_exp_t_mult
        scheduler_exp_eta_min = config.scheduler_exp_eta_min
        
        num_epochs = config.num_epochs
        
        
        # get gpu
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        
#         # data
#         data_transforms = get_transforms()
        
        train_dataset = RetinopathyDataset('train')
#         train_dataset = MyDataset(csv_file=label_pth, file_id = train_id, transform=data_transforms['train'])

        valid_size = int(len(train_dataset) * validation_percentage)
        train_size = len(train_dataset) - valid_size

        train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, valid_size])
        valid_dataset.input_transform = get_transforms('test')
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader =  DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        
        # model
        model_ft= backboneNet_efficient()
        model_ft = model_ft.to(device)

        # loss functions
        criterion = {
                    'classification': FocalLoss().to(device),
                    'regression': nn.SmoothL1Loss().to(device),
                    'ordinal': BinaryFocalLoss(device),
                    }

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=optimizer_lr, momentum=optimizer_momentum, weight_decay=optimizer_weight_decay)

        # # Decay LR by a factor of 0.1 every 30 epochs
        exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ft, T_0=scheduler_exp_t0, T_mult=scheduler_exp_t_mult, eta_min=scheduler_exp_eta_min)

        
        # train model. The best model returned.
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, train_loader=train_loader, valid_loader=valid_loader, num_epochs=num_epochs)

        # Save model
        torch.save(model_ft.state_dict(), result_path + model_name)
