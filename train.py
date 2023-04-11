####################################################
##### By: Mahsa Hasheminejad
##### Implementation of DCNNs
####################################################

import os
import kornia
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrt
import torch.utils.tensorboard
import torch.hub
import sys
import numpy as np

from torchvision import models
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from MyDataSetClass import MultiLabelDataset
from logger import Logger
from PIL import Image, ImageFile
# from models import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---- General Settings -----------------
nGPUs = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()  # use GPU

# ---- DataSet Settings -----------------
TRAIN_IMG_PATH = ''
TRAIN_IMG_EXT = '.png'
TRAIN_DATA_CSV = '/home/sorter1/Dataset/2clss_20221219_Cheft/train.csv'

# VAL_IMG_PATH = '/media/12t_hdd/Imagenet/'
VAL_IMG_PATH = ''
VAL_IMG_EXT = '.png'
VAL_DATA_CSV = '/home/sorter1/Dataset/2clss_20221219_Cheft/test.csv'

number_of_classes = 2

# ---- Training Settings -----------------
batch_size = 4
nCores = 1
batchSize = batch_size * nGPUs
numOfWorkers = nCores  # Number of Training CPU Cores
initialLearningRate = 0.000001
momentumValue = 0.4
learningRateDecayFactor = 0.1  # Factor by which the learning rate will be reduced. new_lr = lr * factor
patienceFactor = 3  # Number of epochs with no improvement after which learning rate will be reduced
numberOfEpochs = 50
save_dir = './snapshots/cleancode'
tensorBoardLoggerAddress = os.path.join(save_dir)
address_label = os.path.join(save_dir, 'label.txt')  # to save class labels list
address_labels_distribution = os.path.join(save_dir, 'label_dist.txt')  # to save labels distributions list
lossType = "CE"  ## CE, Focal
optimizerType = 'Adam'  # SGD, RMSProp, Adam
focalLossAlpha = 0.7  # Set if the lossType is Focal
focalLossGamma = 2.5  # Set if the lossType is Focal
weightDecay = 0.0001
imgSize = 224
fineTuneEnable = True
fineTuneLayers = ['layer4']  # Set layers if fine tune is enabled
fineTuneBN = True  # fine tune batch-normalization

hyperParamsDictionary = {'learningRate': initialLearningRate,
                         'batchSize': batch_size,
                         'imgSize': imgSize,
                         'optimizer': optimizerType,
                         'alphaFocalLoss': focalLossAlpha,
                         'gammaFocalLoss': focalLossGamma,
                         'momentum': momentumValue,
                         'fineTuneEnable': fineTuneEnable,
                         'fineTuneLayer': fineTuneLayers,
                         'finetuneBatchNorm': fineTuneBN,
                         'lossType': lossType,
                         'numberOfEpochs': numberOfEpochs,
                         'weightDecay': weightDecay,
                         'dataset': TRAIN_DATA_CSV}

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

with open(os.path.join(save_dir, "hyperParams.txt"), "w+") as file:
    file.write(str(hyperParamsDictionary))

# Set the logger
logger = Logger(tensorBoardLoggerAddress)

### dataset normalization
normalize = transforms.Normalize(
    mean=[0.2975, 0.2434, 0.1932],
    std=[0.3122, 0.2546, 0.1989]
)

trainTransformations = transforms.Compose([transforms.Resize((imgSize, imgSize)), transforms.ToTensor()])
dset_train = MultiLabelDataset(TRAIN_DATA_CSV, TRAIN_IMG_PATH, TRAIN_IMG_EXT, trainTransformations)

valTransformations = transforms.Compose([transforms.Resize((imgSize, imgSize)), transforms.ToTensor()])
dset_val = MultiLabelDataset(VAL_DATA_CSV, VAL_IMG_PATH, VAL_IMG_EXT, valTransformations)

# Loading the data (second part) (DataLoader):
if use_gpu:
    train_loader = DataLoader(dset_train,
                              batch_size=batchSize,
                              shuffle=True,
                              num_workers=numOfWorkers,  # 1 for CUDA
                              pin_memory=True  # CUDA only
                              )

    val_loader = DataLoader(dset_val,
                            batch_size=batchSize,
                            shuffle=True,
                            num_workers=numOfWorkers,  # 1 for CUDA
                            pin_memory=True  # CUDA only
                            )

else:
    train_loader = DataLoader(dset_train,
                              batch_size=batchSize,
                              shuffle=True,
                              num_workers=numOfWorkers  # 1 for CUDA
                              )

    val_loader = DataLoader(dset_val,
                            batch_size=batchSize,
                            shuffle=True,
                            num_workers=numOfWorkers  # 1 for CUDA
                            )


# Define Method for Save Class Labels into txt file:
def saveClassLabelsIntoTextFile(address_label):
    with open(address_label, "w") as output:
        output.write("\n".join(dset_train.listClassLabels))


# Define Method for Save Labels Distributions into txt file:
def saveLabelsDistributionIntoTextFile(address_labels_distribution):
    fileDistributions = open(address_labels_distribution, "w")
    for item in dset_train.listLabelsDistributions:
        fileDistributions.write("%d\n" % item)
    fileDistributions.close()


# Saving synset & labels distributions:
saveClassLabelsIntoTextFile(address_label)
saveLabelsDistributionIntoTextFile(address_labels_distribution)

#################### Model
model = models.resnet50(pretrained=True)
number_feats = model.fc.in_features
model.fc = nn.Linear(in_features=number_feats, out_features=number_of_classes)
print(model)


#### FineTune networks
if fineTuneEnable:
    # #### Fine-Tune ImageNet models
    for name, child in model.named_children():
        if name in fineTuneLayers:
            print(name + ' is unfrozen')
            # print(child + ' child name')
            for param in child.parameters():
                param.requires_grad = True
        else:
            print(name + ' is frozen')
            for param in child.parameters():
                param.requires_grad = False

if fineTuneBN:
    #### to Fine-Tune all batch norm layers in the model
    for name, param in model.named_parameters():
        if "bn" in name:
            print("Unfrozen batch norms: ")
            print(name)
            param.requires_grad = True


#####################################################################################################################

if use_gpu:
    model = torch.nn.DataParallel(model).cuda()

# Defining the Training Requirements:
if optimizerType == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=initialLearningRate, weight_decay=weightDecay)
elif optimizerType == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=initialLearningRate, momentum=momentumValue, weight_decay=weightDecay)
elif optimizerType == "RMSprop":
    optimizer = torch.optim.RMSprop(model.parameters(), lr=initialLearningRate, alpha=0.99, eps=1e-08,
                                    weight_decay=weightDecay,
                                    momentum=0, centered=False)
elif optimizerType == "Custom":
    ### to define learning rate for each layer seperately
    optimizer = optim.Adam([
        {'params': model.module.classifier.parameters(), 'lr': 0.001},
        {'params': model.module.layer5.parameters(), 'lr': 0.001},
    ], lr=initialLearningRate)

# Defining the Learning Rate Scheduler:
scheduler = lrt.ReduceLROnPlateau(optimizer, 'max', factor=learningRateDecayFactor, patience=patienceFactor,
                                  verbose=True)

# Defining & Preparing the Weighted Loss Function:
if lossType == "CE":
    print('loss function is Cross Entropy')
    criterion = nn.CrossEntropyLoss(weight=None)
if lossType == "Focal":
    print('loss function is Focal')
    kwargs = {"alpha": focalLossAlpha, "gamma": focalLossGamma, "reduction": 'mean'}  # alpha 7.0
    criterion = kornia.losses.FocalLoss(**kwargs)

if use_gpu:
    criterion = criterion.cuda()


# Define method for getting the learning rate from the optimizer:
def getLearningRate(optimizer):
    lr = 0.0
    counter = 0  # for single learning rate optimizers
    for param_group in optimizer.param_groups:
        if counter > 0:
            break
        lr = param_group['lr']
        counter += 1
    return lr


# Defining the training function:
def train(epoch, train_loader, model, criterion, optimizer, step):
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)  # On GPU
        data, target = Variable(data), Variable(target)
        _, targetMax = torch.max(target.data, 1)
        optimizer.zero_grad()
        output = model(data)
        targetMax = Variable(targetMax)  # Note
        loss = criterion(output, targetMax)  # Note
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

        # ============ TensorBoard logging ============#
        # Log the scalar values
        info = {
            'train_loss': loss.item()
        }
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step)

        step += 1
    model.train()
    return step


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

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


# Defining the validation function:
def validate(epoch, val_loader, model, criterion):
    model.eval()
    total_val_loss = 0
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if use_gpu:
                input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)

            input_var = Variable(input)
            target = Variable(target)

            # compute output
            output = model(input_var)

            # measure accuracy and record loss
            _, targetMax = torch.max(target.data, 1)
            prec1, prec5 = accuracy(output.data, targetMax, topk=(1, 2))  # topk=(1, 3)
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute loss
            targetMax = Variable(targetMax)  # Note
            total_val_loss += criterion(output, targetMax).item()

    avg_loss = total_val_loss / len(val_loader)

    # ============ TensorBoard logging ============#
    #   Log the scalar values
    info = {
        'val_loss': avg_loss,
        'Top@1_Accuracy': top1.avg.cpu(),
        'Learning_Rate': getLearningRate(optimizer)
    }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)
    return top1.avg, avg_loss


# Defining the snapshot method (for saving the best model & its training state):
def snapshot(dir_path, run_name, state):
    snapshot_file = os.path.join(dir_path,
                                 run_name + '-model_best.pth')
    torch.save(state, snapshot_file)


### Define a method which delete previous snapshot file form HDD (for save memory)
def deletePreviousBestModel(dir_path, prevEpochNumber):
    if prevEpochNumber > 0:
        prev_snapshot_file = os.path.join(dir_path,
                                          str(prevEpochNumber) + '-model_best.pth')
        os.remove(prev_snapshot_file)


### Define a method for save the last snapshot & last model
def saveLastSnapshot(dir_path, run_name, state):
    snapshot_file = os.path.join(dir_path,
                                 run_name + '-model_last.pth')
    torch.save(state, snapshot_file)


# *** Training our model:
best_accuracy = 0.0
previousBestEpoch = 0
trainTensorBoardCounter = 1

for epoch in range(1, (numberOfEpochs + 1)):
    trainTensorBoardCounter = train(epoch, train_loader, model, criterion, optimizer, trainTensorBoardCounter)
    if epoch % 10 == 0:
        score, val_loss = validate(epoch, val_loader, model, criterion)
        saveLastSnapshot(save_dir, str(epoch), {
            'epoch': epoch,
            'state_dict': model,
            'best_score': score,
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss
        })
    else:
        score, val_loss = validate(epoch, val_loader, model, criterion)
    print('{} {:d}\t{} {:.6f}\t{} {:.6f}'.format('Val Epoch:', epoch, 'Loss:', val_loss, 'Accuracy:', score))

    # Learning Rate Scheduling:
    scheduler.step(score)

    # Save the best model:
    is_best = score >= best_accuracy
    # if is_best:
    best_accuracy = max(score, best_accuracy)
    snapshot(save_dir, str(epoch), {
        'epoch': epoch,
        'state_dict': model,
        'best_score': score,
        'optimizer': optimizer.state_dict(),
        'val_loss': val_loss
    })
    previousBestEpoch = epoch

#### save last snapshot & last model
saveLastSnapshot(save_dir, str(numberOfEpochs), {
    # 'epoch': numberOfEpochs,
    # 'state_dict': model.state_dict(), # ???
    'state_dict': model
    # 'best_score': score,
    # 'optimizer': optimizer.state_dict(),
    # 'val_loss': val_loss
})
