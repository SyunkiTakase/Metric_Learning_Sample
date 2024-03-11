import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from functools import partial
from time import time
from tqdm import tqdm
import numpy as np
from PIL import Image
from metric_loss import contrastive_loss, triplet_loss, batch_hard_triplet_loss, batch_all_triplet_loss

def train(device, train_loader, model, classifier, criterion, optimizer, scaler, use_amp, epoch, method):
    model.train()
    
    sum_ce_loss = 0.0
    sum_metric_loss = 0.0
    sum_loss = 0.0
    count = 0
    margin = 0.2

    for img,label in tqdm(train_loader):
        img = img.to(device, non_blocking=True).float()
        label = label.to(device, non_blocking=True).long()

        with torch.cuda.amp.autocast(enabled=use_amp):
            features = model(img)
            logit = classifier(features)

            if method =='Siamese':
                metric_loss = contrastive_loss(features, label, margin)
            elif method =='Triplet':
                metric_loss = triplet_loss(features, label, margin)
            elif method =='Hard':
                metric_loss = batch_hard_triplet_loss(features, label, margin)
            elif method =='All':
                metric_loss = batch_all_triplet_loss(features, label, margin)
            
            ce_loss = criterion(logit, label)
            loss = ce_loss + metric_loss
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        sum_ce_loss += ce_loss.item()
        sum_metric_loss += metric_loss.item()
        sum_loss += loss.item()
        count += torch.sum(logit.argmax(dim=1) == label).item()
        
    return sum_ce_loss, sum_metric_loss, sum_loss, count

def test(device, test_loader, model):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    sum_loss = 0.0
    count = 0

    with torch.no_grad():
        for img, label in tqdm(test_loader):
            img = img.to(device, non_blocking=True).float()
            label = label.to(device, non_blocking=True).long()
            
            logit = model(img)
            loss = criterion(logit, label)
            
            sum_loss += loss.item()
            count += torch.sum(logit.argmax(dim=1) == label).item()

    return sum_loss, count