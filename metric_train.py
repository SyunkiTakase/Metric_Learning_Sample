import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
from time import time
from tqdm import tqdm
from functools import partial

import trainer
from metric_loss import ContrastiveLoss, TripletLoss

import timm
from timm.models import create_model

device = torch.device("cuda")

train_losses = []
train_accs = []
test_losses = []
test_accs = []
save_path = './model/'

s = time()

def main(args):
    
    num_epoch = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    img_size = args.img_size
    dataset_name = args.dataset
    margin = args.margin
    use_amp = args.amp
    method = args.method
    use_hard_triplets = args.hard_triplets

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean, std),
    ])
    
    if dataset_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10("./data", train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=test_transform, download=False)

    elif dataset_name == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100("./data", train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR100("./data", train=False, transform=test_transform, download=False)

    class_names = train_dataset.classes        
    print(class_names)
    print('Class:', len(class_names))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    model = create_model("resnet18", pretrained=False, num_classes=0) 
    # model = create_model("resnet34", pretrained=False, num_classes=0) 
    # model = create_model("resnet50", pretrained=False, num_classes=0) 
    # model = create_model("resnet101", pretrained=False, num_classes=0)
    # model = create_model("resnet152", pretrained=False, num_classes=0)  
    model.fc = torch.nn.Identity()
    model.to('cuda')
    
    classifier = nn.Linear(512, len(class_names))
    classifier.to('cuda')

    criterion = torch.nn.CrossEntropyLoss()

    if method == 'contrastive':
        metric = ContrastiveLoss(margin=margin) # 損失関数
    elif method == 'triplet':
        metric = TripletLoss(margin=margin, hard_triplets=use_hard_triplets) # 損失関数

    optimizer = torch.optim.Adam([{'params':model.parameters()}, {'params':classifier.parameters()}], lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(num_epoch):
        ce_loss, metric_loss, train_loss, train_count = trainer.train(device, train_loader, model, classifier, optimizer, scaler, use_amp, epoch, criterion, metric)
        test_loss, test_count = trainer.test(device, test_loader, model, classifier, criterion)

        ce_loss = (ce_loss / len(train_loader))
        metric_loss = (metric_loss / len(train_loader))
        train_loss = (train_loss / len(train_loader))
        train_acc = (train_count / len(train_loader.dataset))
        test_loss = (test_loss / len(test_loader))
        test_acc = (test_count / len(test_loader.dataset))

        print(f"epoch: {epoch+1},\
                train loss: {train_loss},\
                ce loss: {ce_loss},\
                metric loss: {metric_loss},\
                train accuracy: {train_acc}\
                test loss: {test_loss},\
                test accuracy: {test_acc}")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    e = time()
    print('Elapsed time is ', e-s)

if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--dataset", type=str, choices=['cifar10', 'cifar100'], default="cifar10")
    parser.add_argument("--margin", type=int, default=10)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument("--method", type=str, choices=['Siamese', 'Triplet'], default="Siamese")
    parser.add_argument("--hard_triplets", action='store_true')
    args=parser.parse_args()
    main(args)
