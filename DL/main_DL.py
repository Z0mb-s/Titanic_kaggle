import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import config
from data_DL import get_data_splits, df_ts
from train_functions import TabularDataset, train_one_epoch, eval_one_epoch

#layers of model
class TitanicModel(nn.Module):
    def __init__(self, input_dim, hidden1=config.model.hidden_dim_1, hidden2=config.model.hidden_dim_2, dropout=config.model.dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden2, 1),
        )
        
    def forward(self, x):
        return self.net(x)
    
#check device
def main():
    device_name = 'cuda' if torch.cuda.is_available() and config.training.device == 'cuda' else 'cpu'
    device = torch.device(device_name)
    print(f"Using device: {device}")
    
    #get data splits
    X_train, X_val, Y_train, Y_val, X_test = get_data_splits(seed=config.general.seed)
    
    #create datasets and dataloaders
    train_dataset = TabularDataset(X_train, Y_train)
    val_dataset = TabularDataset(X_val, Y_val)
    test_dataset = TabularDataset(X_test)
    
    train_loader = DataLoader( # dataloader splits by batch and shuffle
        train_dataset,
        batch_size=config.dataloader_params.batch_size,
        shuffle=config.dataloader_params.shuffle,
        num_workers=config.dataloader_params.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.dataloader_params.batch_size,
        shuffle=False,
        num_workers=config.dataloader_params.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.dataloader_params.batch_size,
        shuffle=False,
    )
    
    #optimizer and initialization of model
    input_dim = X_train.shape[1]
    model = TitanicModel(
            input_dim=input_dim,
            hidden1=config.model.hidden_dim_1,
            hidden2=config.model.hidden_dim_2,
            dropout=config.model.dropout
        ).to(device)
    
    criterion = nn.BCEWithLogitsLoss() #function of loss for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr) # optimization algorithm
    
    #create checkpoints directory
    os.makedirs(config.paths.checkpoints_dir, exist_ok=True)
    best_ckpt_path = os.path.join(config.paths.checkpoints_dir, f"{config.general.experement_name}_best.pth")
    
    best_val_acc = 0.0
    for epoch in range(1, config.training.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"Epoch {epoch}: New best model saved with val_acc: {best_val_acc:.4f}")
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{config.training.num_epochs} | " 
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"\nЛучшая точность на валидации: {best_val_acc:.4f}")
    
    #load best model for testing
    model.load_state_dict(torch.load(best_ckpt_path))
    model.eval()
    
    test_predictions = []
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = (outputs > 0.0).float().cpu().numpy()
            test_predictions.extend(preds.flatten())

    #save predictions to submission file
    test_preds = np.array(test_predictions, dtype=int)
    submission = df_ts[['PassengerId']].copy()
    submission['Survived'] = test_preds
    submission.to_csv(config.paths.submission_csv, index=False)
    print(f"file saved in {config.paths.submission_csv}")


if __name__ == '__main__':
    main()