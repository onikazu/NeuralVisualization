#!/usr/bin/env python
# coding: utf-8

# In[4]:


# http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
# http://pytorch.org/docs/master/torchvision/datasets.html#mnist

# import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')


# In[9]:


import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# データセットをダウンロード
mnist_data = MNIST('./', train=True, download=True, transform=transforms.ToTensor())
data_loader = DataLoader(mnist_data,
                         batch_size=4,
                         shuffle=False)


# In[6]:


# 中身を見てみる
data_iter = iter(data_loader)
images, labels = data_iter.next()


# In[10]:


# matplotlibで1つ目のデータを可視化してみる
npimg = images[0].numpy()
npimg = npimg.reshape((28, 28))
# plt.imshow(npimg, cmap='gray')
print('Label:', labels[0])


# In[11]:


# 訓練データとテストデータを用意
train_data = MNIST('~/tmp/mnist', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_data,
                         batch_size=4,
                         shuffle=True)
test_data = MNIST('~/tmp/mnist', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_data,
                         batch_size=4,
                         shuffle=False)


# In[12]:


# ニューラルネットワークを定義
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 50)
        self.l2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.l1(x)
        x = self.l2(x)
        return x
    
net = Net()


# In[13]:


# コスト関数と最適化手法を定義
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)


# In[8]:


for epoch in range(3):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        
        # Variableに変換
        inputs, labels = Variable(inputs), Variable(labels)
        
        # 勾配情報をリセット
        optimizer.zero_grad()
        
        # 順伝播
        outputs = net(inputs)
        
        # ロスの計算
        loss = criterion(outputs, labels)
        
        # 逆伝播
        loss.backward()
        
        # パラメータの更新
        optimizer.step()
        
        running_loss += loss.data[0]
        
        if i % 5000 == 4999:
            print('%d %d loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0
            
print('Finished Training')


# In[9]:


import torch

correct = 0
total = 0
for data in test_loader:
    inputs, labels = data
    outputs = net(Variable(inputs))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    
print('Accuracy %d / %d = %f' % (correct, total, correct / total))


# In[10]:


test_iter = iter(test_loader)
inputs, labels = test_iter.next()
outputs = net(Variable(inputs))
_, predicted = torch.max(outputs.data, 1)

# plt.imshow(inputs[0].numpy().reshape(28, 28), cmap='gray')
print('Label:', predicted[0])


# In[ ]:





# In[ ]:




