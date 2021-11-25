import matplotlib
from numpy.core.shape_base import block
import model as M
import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda:0')

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10('./data/cifar10',True,transform=transform)
trainloder = data.DataLoader(trainset, batch_size=4, shuffle=True)

testset = torchvision.datasets.CIFAR10('./data/cifar10',False,transform=transform)
testloder = data.DataLoader(testset, batch_size=4, shuffle=False)
classes = ('plane', 'car', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# def imshow(img):
#     img = img / 2 + 0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg,(1,2,0)))
#     plt.show()


# dataiter = iter(trainloder)
# images, labels = dataiter.next()
# print(images)
# imshow(torchvision.utils.make_grid(images))

net  = M.Net()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    for i, datas in enumerate(trainloder,1):
        inputs, labels = datas[0].to(device), datas[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        if i % 2000 == 0:
            print('loss : {}'.format(loss.item()))
print('train finish')

net.eval()
total = 0
acc_num = 0
for datas in testloder:
    inputs, labels = datas[0].to(device), datas[1].to(device)
    outputs = net(inputs)
    pre = torch.max(outputs, 1)[1]
    total += labels.size(0)
    acc_num += (pre == labels).sum().item()

print('acc : {}'.format(acc_num/total))