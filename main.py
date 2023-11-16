import os
import kornia
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrt
import torch.utils.tensorboard
import torch.hub
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms

from multi_label_dataset import MultiLabelDataset
from logger import Logger
from trainer import Trainer

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# General Settings
GPUs = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()  # Set all variables to use GPU

# DataSet Settings
TRAIN_IMG_PATH = ''
TRAIN_IMG_EXT = '.png'
TRAIN_DATA_CSV = '/Users/mahsa/Mine/Dataset/train.csv' ##'./Dataset/2clss_train/train.csv'

VAL_IMG_PATH = ''
VAL_IMG_EXT = '.png'
VAL_DATA_CSV = '/Users/mahsa/Mine/Dataset/val.csv' ##'./Dataset/2clss_val/val.csv'

number_of_classes = 2

# Training Settings
num_cores = 1
final_batch_size = 2
num_workers = num_cores
init_learning_rate = 0.000001
momentum = 0.4
lr_decay_factor = 0.1
patience_factor = 3
total_epochs = 50
save_dir = './snapshots/'
tensorboard_path = os.path.join(save_dir)
label_path = os.path.join(save_dir, 'label.txt')
label_distribution_path = os.path.join(save_dir, 'label_dist.txt')
loss_function = "CE"  # CE, Focal
optimizer_type = 'Adam'  # SGD, RMSProp, Adam
focalLoss_alpha = 0.7  # Set if the loss_function is Focal
focalLoss_gamma = 2.5  # Set if the loss_function is Focal
weight_decay = 0.0001
image_size = 224
fineTune_enable = True
fineTune_layers = ['layer4']  # Set layers name if fine tune is enabled
fineTune_batchNorm = True  # Fine tune batch-normalization

hyper_params_dictionary = {
    'learningRate': init_learning_rate,
    'final_batch_size': final_batch_size,
    'image_size': image_size,
    'optimizer': optimizer_type,
    'alphaFocalLoss': focalLoss_alpha,
    'gammaFocalLoss': focalLoss_gamma,
    'momentum': momentum,
    'fineTune_enable': fineTune_enable,
    'fineTuneLayer': fineTune_layers,
    'finetuneBatchNorm': fineTune_batchNorm,
    'loss_function': loss_function,
    'total_epochs': total_epochs,
    'weight_decay': weight_decay,
    'dataset': TRAIN_DATA_CSV
}

# Create snapshots directory if it doesn't exist
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

# Write hyperparameters to file
with open(os.path.join(save_dir, "hyperParams.txt"), "w+") as file:
    file.write(str(hyper_params_dictionary))

# Set up the logger
logger = Logger(tensorboard_path)

# Dataset normalization
normalize = transforms.Normalize(
    mean=[0.2975, 0.2434, 0.1932],
    std=[0.3122, 0.2546, 0.1989]
)

# Training data transformations
train_transformations = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
dset_train = MultiLabelDataset(TRAIN_DATA_CSV, TRAIN_IMG_PATH, TRAIN_IMG_EXT, train_transformations)

# Validation data transformations
val_transformations = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
dset_val = MultiLabelDataset(VAL_DATA_CSV, VAL_IMG_PATH, VAL_IMG_EXT, val_transformations)

# Loading the data:
if use_gpu:
    train_loader = DataLoader(dset_train, batch_size=final_batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(dset_val, batch_size=final_batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=True)

else:
    train_loader = DataLoader(dset_train, batch_size=final_batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(dset_val, batch_size=final_batch_size, shuffle=True, num_workers=num_workers)


# Define method to save class labels in a txt file
def save_class_labels_textfile(label_path):
    with open(label_path, "w") as output:
        output.write("\n".join(dset_train.listClassLabels))


# Define method to save  labels distributions in a txt file:
def save_labels_distribution_textfile(label_distribution_path):
    fileDistributions = open(label_distribution_path, "w")
    for item in dset_train.listLabelsDistributions:
        fileDistributions.write("%d\n" % item)
    fileDistributions.close()


save_class_labels_textfile(label_path)
save_labels_distribution_textfile(label_distribution_path)

# Model definition
model = models.resnet50(pretrained=True)
number_feats = model.fc.in_features
model.fc = nn.Linear(in_features=number_feats, out_features=number_of_classes)
print(model)


# FineTune networks
if fineTune_enable:
    # Fine-Tune ImageNet models
    for name, child in model.named_children():
        if name in fineTune_layers:
            print(name + ' is unfrozen')
            for param in child.parameters():
                param.requires_grad = True
        else:
            print(name + ' is frozen')
            for param in child.parameters():
                param.requires_grad = False

if fineTune_batchNorm:
    # to Fine-Tune all batch norm layers in the model
    for name, param in model.named_parameters():
        if "bn" in name:
            print("Unfrozen batch norms: ")
            print(name)
            param.requires_grad = True

#####################################################################################################################
if use_gpu:
    model = torch.nn.DataParallel(model).cuda()

# Defining the training requirements:
if optimizer_type == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=init_learning_rate, weight_decay=weight_decay)
elif optimizer_type == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=init_learning_rate, momentum=momentum, weight_decay=weight_decay)
elif optimizer_type == "RMSprop":
    optimizer = torch.optim.RMSprop(model.parameters(), lr=init_learning_rate, alpha=0.99, eps=1e-08,
                                    weight_decay=weight_decay,
                                    momentum=0, centered=False)
elif optimizer_type == "Custom":
    # To define learning rate for each layer separately
    optimizer = optim.Adam([
        {'params': model.module.classifier.parameters(), 'lr': 0.001},
        {'params': model.module.layer5.parameters(), 'lr': 0.001},
    ], lr=init_learning_rate)

# Define the learning rate scheduler:
scheduler = lrt.ReduceLROnPlateau(optimizer, 'max', factor=lr_decay_factor, patience=patience_factor,
                                  verbose=True)

# Define & prepare the weighted loss function:
if loss_function == "CE":
    print('loss function is Cross Entropy')
    criterion = nn.CrossEntropyLoss(weight=None)
if loss_function == "Focal":
    print('loss function is Focal')
    kwargs = {"alpha": focalLoss_alpha, "gamma": focalLoss_gamma, "reduction": 'mean'}  # alpha 7.0
    criterion = kornia.losses.FocalLoss(**kwargs)

if use_gpu:
    criterion = criterion.cuda()


# Define method for getting the learning rate from the optimizer:
def get_learning_rate(optimizer):
    lr = 0.0
    counter = 0  # for single learning rate optimizers
    for param_group in optimizer.param_groups:
        if counter > 0:
            break
        lr = param_group['lr']
        counter += 1
    return lr


# Training the model:
best_accuracy = 0.0
previous_best_epoch = 0
trainTensorBoardCounter = 1
trainer_obj = Trainer(model, train_loader, val_loader, optimizer, criterion, use_gpu, trainTensorBoardCounter, logger)

for epoch in range(1, (total_epochs + 1)):
    trainTensorBoardCounter = trainer_obj.train(epoch)
    if epoch % 10 == 0:
        score, val_loss = trainer_obj.validate(epoch, get_learning_rate(optimizer))
        trainer_obj.save_last_snapshot(save_dir, str(epoch), {
            'epoch': epoch,
            'state_dict': model,
            'best_score': score,
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss
        })
    else:
        score, val_loss = trainer_obj.validate(epoch, get_learning_rate(optimizer))
    print('{} {:d}\t{} {:.6f}\t{} {:.6f}'.format('Val Epoch:', epoch, 'Loss:', val_loss, 'Accuracy:', score))

    # Learning rate scheduling:
    scheduler.step(score)

    # Save the model:
    # is_best = score >= best_accuracy
    # best_accuracy = max(score, best_accuracy)
    trainer_obj.save_snapshot(save_dir, str(epoch), {
        'epoch': epoch,
        'state_dict': model,
        'best_score': score,
        'optimizer': optimizer.state_dict(),
        'val_loss': val_loss
    })
    previous_best_epoch = epoch

# save last model
trainer_obj.save_last_snapshot(save_dir, str(total_epochs), {
    # 'epoch': total_epochs,
    # 'state_dict': model.state_dict(),
    'state_dict': model
    # 'best_score': score,
    # 'optimizer': optimizer.state_dict(),
    # 'val_loss': val_loss
})
