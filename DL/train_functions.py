import os
import gc
import time

import numpy as np

import tqdm

import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import DataLoader, Dataset

#class realised 3 methods from torch.utilis.data.Dataset
class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        #if dataframe or Series -> np array .values
        if hasattr(X, 'values'):
            X = X.values
        if y is not None and hasattr(y, 'values'):
            y = y.values
            
        #get tensors from np arrays
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y=torch.tensor(y, dtype=torch.float32) if y is not None else None
        
    def __len__(self): #back count of ds samples
        return  len(self.X)
    
    def __getitem__(self, idx): #get sample by index
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train() #active dropout
    total_loss = 0.0
    correct = 0
    total = 0
    
    for X_batch, Y_batch in dataloader:
        
        X_batch = X_batch.to(device) # iterate over batches and move to device
        Y_batch = Y_batch.to(device).unsqueeze(1) #dim to (batch, 1)
        
        optimizer.zero_grad() # drop gradients from previous step
        outputs = model(X_batch) #forward pass
        loss = criterion(outputs, Y_batch)
        loss.backward() #backward pass
        optimizer.step() #update weights
        
        total_loss += loss.item() * X_batch.size(0)
        preds = (outputs > 0.0).float()
        correct += (preds == Y_batch).sum().item()
        total += Y_batch.size(0)
    return total_loss / total, correct / total #return average loss and accuracy for the epoch

def eval_one_epoch(model, dataloader, criterion, device):
    model.eval() #model to evaluation mode (disable dropout)
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad(): # disable gradient calculating for evaluation
        for X_batch, Y_batch in dataloader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device).unsqueeze(1) #dim to (batch, 1)
            
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
            preds = (outputs > 0.0).float()
            correct += (preds == Y_batch).sum().item()
            total += Y_batch.size(0)
    return total_loss / total, correct / total