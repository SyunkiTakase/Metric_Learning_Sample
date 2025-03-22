import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
from time import time
from tqdm import tqdm

from functools import partial

def train(device, train_loader, model, classifier, optimizer, scaler, use_amp, epoch, criterion, metric):
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

            ce_loss = criterion(logit, label)
            metric_loss = metric(features, label, margin)            
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

def test(device, test_loader, model, classifier, criterion):
    model.eval()
    sum_loss = 0.0
    count = 0

    with torch.no_grad():
        for img, label in tqdm(test_loader):
            img = img.to(device, non_blocking=True).float()
            label = label.to(device, non_blocking=True).long()
            
            features = model(img)
            logit = classifier(features)
            
            ce_loss = criterion(logit, label)
            
            sum_loss += ce_loss.item()
            count += torch.sum(logit.argmax(dim=1) == label).item()

    return sum_loss, count