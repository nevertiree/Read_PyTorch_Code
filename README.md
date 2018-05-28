# PyTorch源码分析

越简单的东西往往越难讲清楚。使用`PyTorch`完成科研中的深度学习任务是很容易的（相对于`TensorFlow`而言），但是我对于`PyTorch`背后的运行机制缺乏基本的了解。本系列文打算从极其简单的MNIST手写数字识别开始，一步步窥探简单流程背后的PyTorch运行机制。

## 1. Hello World
输出“Hello World”是各个编程语言的入门仪式，完成`MNIST`手写数字图像识别任务就相当于完成了深度学习领域的“`Hello World`”。要实现基于深度学习的图像识别问题，我们至少必须要准备好以下几个模块：
1. **数据导入模块**；（例如PyTorch的`torch.util.data`模块和TensorFlow的`tf.data`模块）
2. **前馈网络模块**；（例如AlexNet等著名网络模型，这是最核心的部分）
3. **损失衡量模块**；（例如`CrossEntropy`等损失函数）
4. **参数优化模块**；（例如`SGD`、`Adam`、`RMSProp`等梯度方法）

我们**用`PyTorch`简单地实现经典的CNN模型**，这也是我们未来深入分析`PyTorch`源码的基础。MNIST手写数字模型由28*28的单通道矩阵表示，原始图片依次通过卷积层、池化层和全连接层，然后被转化一个Softmax的向量。该向量中最大值所在的坐标就代表着该图片的分类标签。`PyTorch`实现的CNN模型的代码流程框架可以如下表示。

```python
import torch
import torch.nn as nn
import torch.optim as optimal

class MNISTNET(nn.Module):
    ... ...

def get_data_loader():
    ... ...

def train(train_loader):
    ... ...

def test(test_loader):
    ... ...

if __name__ == '__main__':

    """ Loading MNIST Data """
    train_loader, test_loader = get_data_loader()  # 数据导入模块
    
    """ MNIST classification Model """
    mnist_net = MNISTNET()  # 前馈网络模块
    
    """ Loss Function """
    criterion = nn.CrossEntropyLoss()  # 损失衡量模块
    
    """ Gradient Descent Method """
    optimizer = optimal.Adam(mnist_net.parameters(), lr=0.001)  # 参数优化模块

    """ Training """
    for _  in range(1, epoch_num+1):
        train(train_loader)
    
    """ Testing """
    evaluate(test_loader)
```

现在我们会先分析**参数优化模块`MNISTNET`**，然后分析**损失衡量模块**和**参数优化模块**，接着分析**训练和测试流程**部分。鉴于数据处理部分内容比较繁琐而且不居于主要地位，因此我们准备在后面再做介绍。

## 2. 神经网络模型的基石 nn.Module
```python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optimal
import torch.nn.functional as functional

class MNISTNET(nn.Module):
    def __init__(self):
        super(MNISTNET, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5, 5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5, 5))
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = functional.max_pool2d(functional.relu(self.conv1(x)), (2, 2))
        x = functional.max_pool2d(functional.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, 16*5*5)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
```

def train(epoch):
    lenet.train()
    losses = []
    for batch_index, (image, label) in enumerate(train_loader):
        optimizer.zero_grad()
        image, label = Variable(image), Variable(label)
        output = lenet(image)
        loss = criterion(output, label)
        loss.backward()

        optimizer.step()
        losses.append(loss.item())
        if batch_index % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_index * len(image),
                len(train_loader.dataset),
                100. * batch_index / len(train_loader),
                loss.item()))
    print("\nTrain set: Average loss: {:.4f}".format(sum(losses) / len(losses)))


def evaluate():
    lenet.eval()
    test_loss = 0
    correct = 0
    for image, label in test_loader:
        image, label = torch.Tensor(image), torch.Tensor(label)
        output = lenet(image)
        test_loss += criterion(output, label).item()
        pred = output.data.max(1)[1]
        correct += pred.eq(label.item()).cpu().sum()

    test_loss /= len(test_loader)
    print("Test data: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))
```
通过继承nn.Module写LeNet模型
