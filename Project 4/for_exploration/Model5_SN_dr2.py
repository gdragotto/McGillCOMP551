import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
from comet_ml import Experiment
import datetime
now = datetime.datetime.now()
unique = str(now.hour) + str(now.minute)

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import time
import copy


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version='original', num_classes=1000, dropout=0.5):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        #final_conv = nn.Conv2d(384, self.num_classes, kernel_size=1)
        if (version == 'softmax'):
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                final_conv,
                nn.Softmax(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                final_conv,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(float(correct_k.mul_(100 / batch_size)))
    return res

def train_model(experiment, model, dataloaders, criterion, optimizer, earlyStopping):
    print("Starting the training with earlyStopping:" + str(earlyStopping))
    print("Each epoch trains on " + str(len(dataloaders['train'])) + " batches.")
    print("Validating on " + str(len(dataloaders['val'])) + " batches.")
    since = time.time()
    val_acc_history_5x = []
    val_acc_history_1x = []
    best_acc_5x = -1.0
    best_acc_1x = -1.0
    notimp = 0

    epoch=0
    StopCond=True
    while (StopCond):
        experiment.log_current_epoch(epoch)
        print("Epoch "+str(epoch))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                experiment.train()
            else:
                model.eval()
                experiment.validate()

            running_loss = 0.0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            print('\t{} Loss: {:.4f} Acc1x: {:.4f} Acc5x: {:.4f}'.format(phase, epoch_loss, prec1, prec5))
            if phase == 'val':
                if (prec5 == best_acc_5x and prec1 >best_acc_1x) or (prec5 >= best_acc_5x):
                        best_acc_5x = prec5
                        best_acc_1x = prec1
                        bestModel = copy.deepcopy(model.state_dict())
                        print("New incumbent accuracy found. Saving model")
                        fPath = "Model_" + unique + "__x5_" + str(round(prec5,4)) + "_x1_" + str(round(prec1,4)) +".pt"
                        torch.save(bestModel, fPath)
                        experiment.log_asset(file_path=fPath, file_name="Incumbent.pt", overwrite=True)
                        experiment.log_other("Incumbent 1x", prec1)
                        experiment.log_other("Incumbent 5x", prec5)
                        experiment.log_other("Incumbent Mixed", "__x5_" + str(round(prec5,4)) + "_x1_" + str(round(prec1,4)))
                        notimp = 0
                else:
                    notimp += 1
                val_acc_history_5x.append(prec5)
                val_acc_history_1x.append(prec1)
                experiment.log_metric("Accuracy 5x (v)",prec5)
                experiment.log_metric("Accuracy 1x (v)",prec1)
            else:
                experiment.log_metric("Accuracy 5x (t)",prec5)
                experiment.log_metric("Accuracy 1x (t)",prec1)
                experiment.log_other("Loss (t)", epoch_loss)

        if (notimp > earlyStopping and best_acc_5x>79):
            print("Early Stopping triggered.")
            StopCond=False
        experiment.log_epoch_end(epoch)
        epoch+=1

    time_elapsed = time.time() - since
    print('\tTraining completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('\tBest validation accuracy: {:4f}'.format(best_acc_5x))

    torch.save(bestModel, "Model_" + unique+".pt")
    model.load_state_dict(bestModel)
    return model, val_acc_history_5x, val_acc_history_1x


num_classes = 200
batch_size = 512
early_stopping = 50
learningRate=0.04
weight_decay= 0.0002

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    ]),
    'val': transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    ])
}
experiment = Experiment(api_key="<HIDDEN>",project_name="SqueezeNet STD",workspace="gdragotto")
data_dir = "/local_workspace/draggabr/tiny-imagenet-200-pytorch/"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])for x in ['train', 'val']}
dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}
model_ft = SqueezeNet(num_classes=num_classes,dropout=0.75)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model_ft = nn.DataParallel(model_ft)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)
optimizer_ft = optim.SGD(model_ft.parameters(), lr=learningRate, momentum=0.9, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
#model_ft.load_state_dict(torch.load('Model_933.pt'))
experiment.log_parameters({"GPUs": torch.cuda.device_count(), "batch_size": batch_size, "earlyStopping": early_stopping,"lr":learningRate,"decay":weight_decay})
model_ft, hist5x, hist1x = train_model(experiment, model_ft, dataloaders, criterion, optimizer_ft, early_stopping)