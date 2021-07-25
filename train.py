from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import sys

data_dir =  "/Users/apple/Desktop/art/PACS/dataset"
MODEL_PATH="/Users/apple/Desktop/art/PACS/result/vgg16"
CKPT_PATH = MODEL_PATH+"/ckpt"
epochs=100
batch_size=128

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batchsize = target.size(0)

    _,pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul(100.0 / batch_size))
    return res



image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),  data_transforms[x])for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,   shuffle=True, num_workers=1)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getckpt(dirs):
    res=-1
    if(os.path.exists(dirs)):
        f=open(dirs,"r")
        line=f.readline()
        print(line)
        res=int(line.strip())
        f.close()
    return res
def writeckpt(dirs,epoch):
    with    open(dirs,"w+") as f:
        f.write(str(epoch))
   
        f.close()
def train_model(model, criterion, optimizer, scheduler, num_epochs=25,start=0):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start,num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print(outputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
                MODEL_name=MODEL_PATH+".pth"
                print("save model {}".format(MODEL_name))
                torch.save(model.state_dict(), MODEL_name)
                writeckpt(CKPT_PATH,epoch)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



model_ft = models.vgg16(pretrained=False)

for param in model_ft.parameters():
    param.requires_grad = False
#Freeze all layers except the last one（fc）
model_ft.classifier = nn.Sequential(            
     nn.Linear(512 * 7 * 7, 4096),            
     nn.ReLU(True),           
      nn.Dropout(),            
      nn.Linear(4096, 4096),            
      nn.ReLU(True),            
      nn.Dropout(),            
     nn.Linear(4096, len(class_names)),        
     )
start = getckpt(CKPT_PATH)
if(start>=0):
    LOAD_FILE = MODEL_PATH+".pth"
    model_ft.load_state_dict(torch.load(LOAD_FILE))
start=start+1
#num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).

#model_ft.fc = nn.Linear(num_ftrs, len(class_names))
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# notice that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)   
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=epochs,start=start)
model_ft.eval()
top1 = AverageMeter()
top2 = AverageMeter()

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(dataloaders["val"]):
        # measure data loading time

        if(torch.cuda.is_available()):
          inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model_ft(inputs)

        # measure accuracy and record loss
        prec1, prec2 = accuracy(outputs.data, targets.data, topk=(1, 2))
        top1.update(prec1.item(), inputs.size(0))
        top2.update(prec2.item(), inputs.size(0))

print("top1",top1.avg)
print("top2",top2.avg)