import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.utils.data as Data
import  torchvision
import numpy as np

#Hyper Parameters
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True
BATCH_SIZE = 5

#1.下载数据
train_data = torchvision.datasets.MNIST(root='./mnist',
                                        train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=DOWNLOAD_MNIST)
test_data = torchvision.datasets.MNIST(root='./mnist',
                                        train=False,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=DOWNLOAD_MNIST)
# #2.加载数据
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
#3构建网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x),
        x = self.conv2(x),
        x = x.view(x.size(0), -1)
        out = self.out(x)
        return out
cnn = CNN()
print(cnn)

#4定义损失函数，定义优化器
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
#训练过程
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
