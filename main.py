# Author : Mudit Rastogi

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import os
from data.dataloader import ChestXrayDataSet
from model.model import DenseNet
import torch.nn as nn

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize(224),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



DATA_DIR = '/home/mudit/project/chestai/chest_traindata/train/'
classes = [ 'Atelectasis',  'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass' 'Nodule', 'Pneumonia',
            'Pneumothorax', 'Consolidation','Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia' ]


N_CLASSES = len(classes)
TRAIN_IMAGE_LIST = '/home/mudit/project/chestai/chest_traindata/new_labels/train_list.txt'
TEST_IMAGE_LIST = '/home/mudit/project/chestai/chest_traindata/new_labels/test_list.txt'


BATCH_SIZE = 1


trainloader = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TRAIN_IMAGE_LIST
                                    )
                                    # transform=transforms.Compose([
                                    #     transforms.Resize(256),
                                    #     transforms.TenCrop(224),
                                    #     # transforms.Lambda
                                    #     # (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                    #     # transforms.Lambda
                                    #     # (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    #     #
                                    #     ]))

net = DenseNet(growthRate=12, depth=10, reduction=0.5, bottleneck=True, nClasses=14)
lr = 0.1
optimizer = optim.SGD(net.parameters(),
                      lr=lr,
                      momentum=0.9,
                      weight_decay=0.0005)

criterion = nn.BCELoss()


for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        inputs,labels = inputs.unsqueeze(0), labels.unsqueeze(0)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
