import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score

from src.networks.dual_branch import FragmentSleepNet
from src.dataset import PhysNetDualDataset 


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Training on device: {device.type.upper()}")

    # Load Data
    print("[*] Loading datasets into RAM...")
    train_ds = PhysNetDualDataset(Path("data/processed/train.npz"))
    val_ds = PhysNetDualDataset(Path("data/processed/test.npz"))
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    n_pos = train_ds.labels.sum().item()
    n_neg = len(train_ds) - n_pos
    pos_weight_val = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"[*] Dataset Imbalance -> Neg: {int(n_neg)} | Pos: {int(n_pos)}")
    print(f"[*] Applying Loss pos_weight: {pos_weight_val:.2f}")

    model = FragmentSleepNet().to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)

    EPOCHS = 20
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]", leave=False, dynamic_ncols=True)
        
        for signals, psds, labels in pbar:
            signals, psds, labels = signals.to(device), psds.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(signals, psds)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for signals, psds, labels in val_loader:
                signals, psds, labels = signals.to(device), psds.to(device), labels.to(device)
                
                logits = model(signals, psds)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = (logits >= -1.0).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader)

        val_acc = accuracy_score(all_targets, all_preds)
        val_recall = recall_score(all_targets, all_preds, zero_division=0)
        val_f1 = f1_score(all_targets, all_preds, zero_division=0)

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc*100:.1f}% | "
              f"Val Recall: {val_recall*100:.1f}% | "
              f"Val F1: {val_f1:.3f}")

if __name__ == "__main__":
    main()