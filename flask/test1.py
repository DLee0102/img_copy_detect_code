# -*- coding = utf-8 -*-
# @Time : 2022/8/16 11:38
# @Author : DL
# @File : test1.py
# @Software : PyCharm

from datetime import timedelta
from flask import Flask, request
import torchvision.transforms as transforms
from PIL import Image
import os
import torch
import json
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import base64
import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader



app = Flask(__name__)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x



@app.route('/')
def hello_world():
    return 'Hello World'


# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        team_image = base64.b64decode(request.form.get("image"))  # 队base64进行解码还原。
        with open("static/static/111111.jpg", "wb") as f:
            f.write(team_image)
        image = Image.open("static/static/111111.jpg")
        image = image.convert('RGB')

        model = Classifier().to('cuda')
        model.load_state_dict(torch.load('D:/pythonProject/flask/CNN_2022-08-29_12_26_46.pth'))
        model.eval()

        test_tfm = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        test_set = ImageFolder('D:/pythonProject/flask/static', transform = test_tfm)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        for batch in test_loader:
            img, label = batch
        with torch.no_grad():
            pred = model(img.to('cuda'))
        print(pred)
        classes = pred.argmax(dim=-1).cpu().item()

        result = 0
        if classes == 0:
            result = '图片未经过篡改'
        if classes == 1:
            result = '图片已被篡改'

        return result   # 将预测结果传回给前端

if __name__ == '__main__':
    app.run()