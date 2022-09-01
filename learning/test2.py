# -*- coding = utf-8 -*-
# @Time : 2022/8/20 21:26
# @Author : DL
# @File : test_readImg.py
# @Software : PyCharm

import time
import os
from os import path
import sys
import shutil
from glob import glob
import random
import cv2
import numpy as np
# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
from torchvision.datasets import ImageFolder

# This is for the progress bar.

import torchvision.models
from tqdm.auto import tqdm
import torch.nn.functional as F

# It is important to do data augmentation in training.
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for food recognition.
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((224, 224)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    # transforms.RandomResizedCrop(120),
    transforms.RandomHorizontalFlip(),
    transforms.RandomChoice([transforms.RandomRotation(45), transforms.ColorJitter(0.5, 0.5, 0.5)]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# train_imgpath = 'D:/pythonProject/testImgCopyDetect/train/train'
# test_imgpath = 'D:/pythonProject/testImgCopyDetect/validation/validation'
#
# dst_root = 'D:/pythonProject/testImgCopyDetect/train/train/*.jpg'

batch_size = 32
# full_set = ImageFolder("D:/pythonProject/testImgCopyDetect/train", transform=train_tfm)
# train_full_size = int(0.8 * len(full_set))
# test_size = len(full_set) - train_full_size
# valid_size = int(0.1 * len(full_set))
# train_size = int(0.9 * len(full_set))
# valid_size = len(full_set) - train_size
# # train_full_set, test_set = torch.utils.data.random_split(full_set, [train_full_size, test_size],
# # generator=torch.Generator().manual_seed(520))
# train_set, valid_set = torch.utils.data.random_split(full_set, [train_size, valid_size])

train_set = ImageFolder("D:/pythonProject/testImgCopyDetect/train", transform=train_tfm)
valid_set = ImageFolder("D:/pythonProject/testImgCopyDetect/validation", transform=test_tfm)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# class ImageDataset(Dataset){
#
# }


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True,
                 init_weights=False):  # 这是主分类器  aux_logits是true则启动使用辅助分类器，否则不启动
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:  # 是否启用辅助分类器
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:  # 是否使用初始化权重
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.training and self.aux_logits:  # eval model lose this layer 如果为训练模型则使用辅助分类器，验证模型则关闭辅助分类器
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.training and self.aux_logits:  # eval model lose this layer# eval model lose this layer 如果为训练模型则使用辅助分类器，验证模型则关闭辅助分类器
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        if self.training and self.aux_logits:  # eval model lose this layer# eval model lose this layer 如果为训练模型则使用辅助分类器，验证模型则关闭辅助分类器
            return x, aux2, aux1
        return x

    def _initialize_weights(self):  # 初始化权重的提房，有兴趣可以查查函数看看
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Inception(nn.Module):  # 搭建多分支架构的一部分
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5,
                 pool_proj):  # self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        super(Inception, self).__init__()

        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)  # 保证输出大小等于输入大小
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)  # 保证输出大小等于输入大小
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):  # 辅助分类器结构
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # output[batch, 128, 4, 4]

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = self.averagePool(x)
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 2048
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        # N x 1024
        x = self.fc2(x)
        # N x num_classes
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x



def load_imgname(train_imgpath, test_imgpath):
    test_filenames = os.listdir(test_imgpath)
    train_filenames = os.listdir(train_imgpath)

    for tfname in train_filenames:
        origin_tfname = tfname
        src = os.path.join(os.path.abspath(train_imgpath), tfname)  # 要修改的文件的绝对路径
        if tfname.split('_')[0]=='Tp':
            tfname = '1'+'_'+tfname
        if tfname.split('_')[0]=='Au':
            tfname = '0'+'_'+tfname
        dst = os.path.join(os.path.abspath(train_imgpath), tfname)
        os.rename(src, dst)


def load_img(root):
    img_set = glob(root)
    for img_name in img_set:
        origin_img_name = img_name
        print(img_name.split('_')[0])
        if img_name.split('_')[0]==r'D:/pythonProject/testImgCopyDetect/train/train\Tp':
            img_name.split('_')[0]=r'D:/pythonProject/testImgCopyDetect/train/train\1'
        if img_name.split('_')[0]==r'D:/pythonProject/testImgCopyDetect/train/train\Au':
            img_name.split('_')[0]=r'D:/pythonProject/testImgCopyDetect/train/train\0'
        os.rename(origin_img_name, img_name)
    a = 0
    print(img_set)
    for i in img_set:
        a += 1
    print(a)

if __name__ == '__main__':
    print(train_set)
    # load_imgname(train_imgpath, test_imgpath)

    # "cuda" only when GPUs are available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize a model, and put it on the device specified.
    model = GoogLeNet(num_classes=2, aux_logits=True, init_weights=True).to(device)
    model.device = device

    # For the classification task, we use cross-entropy as the measurement of performance.
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # The number of training epochs.
    n_epochs = 1000

    # Whether to do semi-supervised learning.
    do_semi = False

    # These are used to record information in training.
    train_losses = []
    train_accs = []

    # These are used to record information in validation.
    valid_losses = []
    valid_accs = []

    train_loss_epoch = []
    train_acc_epoch = []
    valid_loss_epoch = []
    valid_acc_epoch = []

    enter_ = 1  # save the model
    enter_2 = 1  # save the model

    for epoch in range(n_epochs):
        # ---------- TODO ----------
        # In each epoch, relabel the unlabeled dataset for semi-supervised learning.
        # Then you can combine the labeled dataset and pseudo-labeled dataset for the training.
        # if do_semi:
        #     # Obtain pseudo-labels for unlabeled data using trained model.
        #     # pseudo_set = get_pseudo_labels(unlabeled_set, model)
        #
        #     # Construct a new dataset and a data loader for training.
        #     # This is used in semi-supervised learning only.
        #     concat_dataset = ConcatDataset([train_set, pseudo_set])
        #     train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()



        # Iterate the training set by batches.
        for batch in tqdm(train_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch

            # Forward the data. (Make sure data and model are on the same device.)
            logits_1 = model(imgs.to(device))

            logit, logit_1, logit_2 = logits_1

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss1 = criterion(logit, labels.to(device))
            loss2 = criterion(logit_1, labels.to(device))
            loss3 = criterion(logit_2, labels.to(device))

            loss = loss1 + loss2 * 0.3 + loss3 * 0.3  # 计算总损失

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logit.argmax(dim = -1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_losses.append(loss.item())
            train_accs.append(acc.item())

        # The average loss and accuracy of the training set is the average of the recorded values.
        train_loss = sum(train_losses) / len(train_losses)
        train_acc = sum(train_accs) / len(train_accs)
        train_loss_epoch.append(train_loss)
        train_acc_epoch.append(train_acc)

        # Print the information.
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()



        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_losses.append(loss.item())
            valid_accs.append(acc.item())

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_losses) / len(valid_losses)
        valid_acc = sum(valid_accs) / len(valid_accs)
        valid_loss_epoch.append(valid_loss)
        valid_acc_epoch.append(valid_acc)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        if valid_acc > 0.70 and enter_ == 1:
            time = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())
            torch.save(model.state_dict(), "GoogleNet" + "_" + time + ".pth")
            enter_ = 0
            print("模型保存于" + time)
        if valid_acc > 0.80 and enter_2 == 1:
            time = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())
            torch.save(model.state_dict(), "GoogleNet" + "_" + time + ".pth")
            enter_2 = 0
            print("模型保存于" + time)

    # Tensor.cpu(train_loss)
    # Tensor.cpu(valid_loss)
    # Tensor.cpu(train_accs)
    # Tensor.cpu(valid_accs)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(n_epochs), train_loss_epoch,
             "ro-", label="Train loss")
    plt.plot(range(n_epochs), valid_loss_epoch,
             "bs-", label="valid loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(range(n_epochs), train_acc_epoch,
             "ro-", label="Train accur")
    plt.plot(range(n_epochs), valid_acc_epoch,
             "bs-", label="valid accur")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()

    time = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())
    torch.save(model.state_dict(), "GoogleNet" + "_" + time + ".pth")

    print("模型保存于" + time)

