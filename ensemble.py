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
from pseudo_model import backboneNet_efficient
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
import ensemble_config as cfg
from pseudo_dataset import PseudoDataset

data_transforms = {
    "train": transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomAffine(
                degrees=(-180, 180),
                scale=(1, 1.3),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=(0.2),
                contrast=(0.2),
                hue=(0.1),
                saturation=(0.2),
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((288, 288)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "tta": transforms.Compose(
        [
            transforms.Resize((288, 288)),
            transforms.RandomAffine(
                degrees=(-180, 180),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}


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
        if (
            check_shape == 0
        ):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img


def scale_radius(src, img_size, padding=False):
    x = src[src.shape[0] // 2, ...].sum(axis=1)
    r = (x > x.mean() / 10).sum() // 2
    yx = src.sum(axis=2)
    region_props = measure.regionprops((yx > yx.mean() / 10).astype("uint8"))
    yc, xc = np.round(region_props[0].centroid).astype("int")
    x1 = max(xc - r, 0)
    x2 = min(xc + r, src.shape[1] - 1)
    y1 = max(yc - r, 0)
    y2 = min(yc + r, src.shape[0] - 1)
    dst = src[y1:y2, x1:x2]
    dst = cv2.resize(
        dst, dsize=None, fx=img_size / (2 * r), fy=img_size / (2 * r)
    )
    if padding:
        pad_x = (img_size - dst.shape[1]) // 2
        pad_y = (img_size - dst.shape[0]) // 2
        dst = np.pad(dst, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), "constant")
    return dst


def quadratic_weighted_kappa(y_pred, y_true):
    return metrics.cohen_kappa_score(y_pred, y_true, weights="quadratic")


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


class FocalLoss(nn.Module):
    def __init__(
        self,
        num_class=5,
        alpha=0.25,
        gamma=2,
        balance_index=-1,
        smooth=None,
        size_average=True,
    ):
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
            raise TypeError("Not support alpha type")

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

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
                one_hot_key, self.smooth, 1.0 - self.smooth
            )
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


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    ordinal_labels = torch.FloatTensor(
        [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]
    ).to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_loss = 10000.0
    best_train_acc = 0.0
    best_valid_loss = 10000.0
    best_valid_acc = 0.0
    best_train_score = 0.0
    best_valid_score = 0.0

    best_train_rg_acc = 0.0
    best_valid_rg_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                model_loader = train_loader
                dataset_size = train_size
            elif phase == "val":
                model.eval()  # Set model to evaluate mode
                model_loader = valid_loader
                dataset_size = valid_size

            train_loss = 0.0
            valid_loss = 0.0
            train_corrects = 0
            valid_corrects = 0
            train_score = 0.0
            valid_score = 0.0

            train_rg_acc = 0.0
            valid_rg_acc = 0.0
            y1_score = []
            y2_score = []

            # Iterate over data.
            if phase == "train":
                for iter, (inputs, targets) in tqdm(
                    enumerate(model_loader), total=dataset_size
                ):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    # zero the parameter gradients
                    #                     optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    outputs = model(inputs)

                    rg_outputs = outputs[0]
                    rg_outputs = torch.sigmoid(rg_outputs) * 4.5

                    cls_outputs = outputs[1]
                    cls_softmax = torch.nn.Softmax(dim=1)
                    cls_outputs = cls_softmax(cls_outputs)

                    ord_outputs = outputs[2]
                    ord_outputs = torch.sigmoid(ord_outputs)

                    preds = torch.max(cls_outputs, 1)

                    #                     loss = criterion(outputs, targets)

                    rg_loss = criterion["regression"](
                        rg_outputs.view(-1), targets.float()
                    )
                    cls_loss = criterion["classification"](
                        cls_outputs, targets
                    )
                    ord_loss = criterion["ordinal"](
                        ord_outputs, ordinal_labels[targets]
                    )

                    loss = rg_loss + cls_loss + ord_loss

                    # gradient accumulation
                    loss = loss / accumulation_steps

                    # backward + optimize only if in training phase
                    loss.backward()

                    if (iter + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    outputs = rg_outputs.unsqueeze(1)
                    thrs = [0.5, 1.5, 2.5, 3.5]
                    outputs[outputs < thrs[0]] = 0
                    outputs[(outputs >= thrs[0]) & (outputs < thrs[1])] = 1
                    outputs[(outputs >= thrs[1]) & (outputs < thrs[2])] = 2
                    outputs[(outputs >= thrs[2]) & (outputs < thrs[3])] = 3
                    outputs[outputs >= thrs[3]] = 4

                    y1_score = y1_score + outputs[:, 0].squeeze(1).tolist()
                    y2_score = y2_score + targets.tolist()

                    # statistics
                    train_loss += (
                        loss.item() * accumulation_steps * inputs.size(0)
                    )

                    train_rg_acc += torch.sum(
                        outputs[:, 0].squeeze(1) == targets.data
                    )

                    train_corrects += torch.sum(preds[1] == targets.data)

                    if (iter + 1) % 500 == 0:
                        print(
                            "{} iter:{} cls_loss: {:.4f} rg_loss {:.4f} ord_loss {:.4f} cls_Acc: {:.4f} rg_Acc: {:.4f} Score {:.4f}".format(
                                phase,
                                iter,
                                cls_loss,
                                rg_loss,
                                ord_loss,
                                train_corrects.double()
                                / ((iter + 1) * batch_size),
                                train_rg_acc.float()
                                / ((iter + 1) * batch_size),
                                quadratic_weighted_kappa(y1_score, y2_score),
                            )
                        )

                scheduler.step()

                train_score = quadratic_weighted_kappa(y1_score, y2_score)

                epoch_loss = train_loss / dataset_size
                epoch_acc = train_corrects.double() / n_train
                epoch_score = train_score
                epoch_rg_acc = train_rg_acc.double() / n_train

                print(
                    "{} Loss: {:.4f}  Acc: {:.4f} rg_acc: {:.4f} Score: {:.4f}".format(
                        phase, epoch_loss, epoch_acc, epoch_rg_acc, epoch_score
                    )
                )

                if best_train_acc < epoch_acc:
                    best_train_acc = epoch_acc

                if best_train_rg_acc < epoch_rg_acc:
                    best_train_rg_acc = epoch_rg_acc

                if best_train_loss > epoch_loss:
                    best_train_loss = epoch_acc
                if best_train_score < epoch_score:
                    best_train_score = epoch_score

            # val and deep copy the model
            if phase == "val":
                with torch.no_grad():
                    for iter, (inputs, targets) in tqdm(
                        enumerate(model_loader), total=dataset_size
                    ):
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        # forward
                        # track history if only in train
                        outputs = model(inputs)

                        rg_outputs = outputs[0]
                        rg_outputs = torch.sigmoid(rg_outputs) * 4.5

                        cls_outputs = outputs[1]
                        cls_softmax = torch.nn.Softmax(dim=1)
                        cls_outputs = cls_softmax(cls_outputs)

                        ord_outputs = outputs[2]
                        ord_outputs = torch.sigmoid(ord_outputs)

                        preds = torch.max(cls_outputs, 1)

                        rg_loss = criterion["regression"](
                            rg_outputs.view(-1), targets.float()
                        )
                        cls_loss = criterion["classification"](
                            cls_outputs, targets
                        )
                        ord_loss = criterion["ordinal"](
                            ord_outputs, ordinal_labels[targets]
                        )

                        loss = rg_loss + cls_loss + ord_loss

                        outputs = rg_outputs.unsqueeze(1)

                        thrs = [0.5, 1.5, 2.5, 3.5]
                        outputs[outputs < thrs[0]] = 0
                        outputs[(outputs >= thrs[0]) & (outputs < thrs[1])] = 1
                        outputs[(outputs >= thrs[1]) & (outputs < thrs[2])] = 2
                        outputs[(outputs >= thrs[2]) & (outputs < thrs[3])] = 3
                        outputs[outputs >= thrs[3]] = 4

                        y1_score = y1_score + outputs[:, 0].squeeze(1).tolist()
                        y2_score = y2_score + targets.tolist()

                        # statistics
                        valid_loss += loss.item() * inputs.size(0)

                        valid_rg_acc += torch.sum(
                            outputs[:, 0].squeeze(1) == targets.data
                        )
                        valid_corrects += torch.sum(preds[1] == targets.data)

                valid_score = quadratic_weighted_kappa(y1_score, y2_score)

                epoch_loss = valid_loss / dataset_size
                epoch_rg_acc = valid_rg_acc.double() / n_valid
                epoch_acc = valid_corrects.double() / n_valid
                epoch_score = valid_score

                print(
                    "{} Loss: {:.4f}  Acc: {:.4f} rg_Acc: {:.4f} Score: {:.4f}".format(
                        phase, epoch_loss, epoch_acc, epoch_rg_acc, epoch_score
                    )
                )

                if best_valid_acc < epoch_acc:
                    best_valid_acc = epoch_acc

                if best_valid_rg_acc < epoch_rg_acc:
                    best_valid_rg_acc = epoch_rg_acc

                if best_valid_loss > epoch_loss:
                    best_valid_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print("save best training weight,complete!")

                if best_valid_score < epoch_score:
                    best_valid_score = epoch_score

        time_elapsed = time.time() - since
        print(
            "Complete one epoch in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

        print()

    print(
        "Best train acc: {:4f} Best val acc: {:4f}".format(
            best_train_acc, best_valid_acc
        )
    )
    print(
        "Best train rg acc: {:4f} Best val rg acc: {:4f}".format(
            best_train_rg_acc, best_valid_rg_acc
        )
    )
    print(
        "Best train loss: {:4f} Best val loss: {:4f}".format(
            best_train_loss, best_valid_loss
        )
    )
    print(
        "Best train score: {:4f} Best val score: {:4f}".format(
            best_train_score, best_valid_score
        )
    )

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


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
    return sum(lst) / len(lst)


if __name__ == "__main__":

    csv_name = cfg.submission_csv

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = backboneNet_efficient()
    model_ft2 = backboneNet_efficient()
    model_ft3 = backboneNet_efficient()
    model_ft4 = backboneNet_efficient()

    model_ft.load_state_dict(
        torch.load(cfg.model_folder + cfg.model_1, map_location="cuda:0")
    )
    model_ft = model_ft.to(device)
    model = model_ft

    model_ft2.load_state_dict(
        torch.load(cfg.model_folder + cfg.model_2, map_location="cuda:0")
    )
    model_ft2 = model_ft2.to(device)

    model_ft3.load_state_dict(
        torch.load(cfg.model_folder + cfg.model_3, map_location="cuda:0")
    )
    model_ft3 = model_ft3.to(device)

    thrs = [0.5, 1.5, 2.5, 3.5]

    with open("./output/" + csv_name, "w", newline="") as csvFile:

        field = ["id_code", "diagnosis"]
        writer = csv.DictWriter(csvFile, field)
        writer.writeheader()

        test_path = cfg.test_path
        allFileList = os.listdir(test_path)

        for iter, file in enumerate(allFileList):
            print(iter)
            result = []
            for model in [model_ft, model_ft2, model_ft3]:
                if os.path.isfile(test_path + file):
                    path = test_path + file
                    img = cv2.imread(path)
                    img = crop_image_from_gray(img)
                    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    img = data_transforms["test"](img)
                    img = img.unsqueeze(0)
                    with torch.no_grad():
                        model.eval()
                        img = img.to(device)
                        outputs = model(img)
                        rg_outputs = outputs[0]
                        rg_outputs = torch.sigmoid(rg_outputs) * 4.5

                        cls_outputs = outputs[1]
                        cls_softmax = torch.nn.Softmax(dim=1)
                        cls_outputs = cls_softmax(cls_outputs)
                        cls_preds = torch.max(cls_outputs, 1)
                        cls_predict = cls_preds[1].cpu().numpy()

                        ord_outputs = outputs[2]
                        ord_outputs = torch.sigmoid(ord_outputs)

                        rg_outputs = rg_outputs.unsqueeze(1)
                        rg_outputs[rg_outputs < thrs[0]] = 0
                        rg_outputs[
                            (rg_outputs >= thrs[0]) & (rg_outputs < thrs[1])
                        ] = 1
                        rg_outputs[
                            (rg_outputs >= thrs[1]) & (rg_outputs < thrs[2])
                        ] = 2
                        rg_outputs[
                            (rg_outputs >= thrs[2]) & (rg_outputs < thrs[3])
                        ] = 3
                        rg_outputs[rg_outputs >= thrs[3]] = 4
                        rg_preds = rg_outputs.tolist()
                        rg_predict = rg_preds[0][0]

                        ord_outputs[ord_outputs >= 0.5] = 1
                        ord_outputs[ord_outputs < 0.5] = 0
                        ord_predict = sum(ord_outputs[0].cpu().numpy())

                        predict = [rg_predict[0], cls_predict[0], ord_predict]
                        predict = [most_frequent(predict)]

                        result.extend(predict)
            final_predict = most_frequent(result)
            writer.writerow(
                {
                    "id_code": file.split(".png")[0],
                    "diagnosis": int(final_predict),
                }
            )
