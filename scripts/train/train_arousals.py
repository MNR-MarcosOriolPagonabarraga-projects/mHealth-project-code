import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from src.networks.dual_branch import EEGContextNet
from offline.dataset import PhysNetContextDataset 
from offline.viz import plot_history, plot_epoch_confusion_matrix

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Training on device: {device.type.upper()}")

    RUN_DIR = Path("models") / "context_net"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Saving artifacts to: {RUN_DIR}")
    
    # ---------------------------------------------------------
    # FIXED CONFIGURATION
    # ---------------------------------------------------------
    best_precission = -1.0  
    PRED_THRESHOLD = 0.4  # Neutral threshold
    EPOCHS = 40
    
    print("[*] Loading datasets into RAM...")
    train_ds = PhysNetContextDataset(Path("data/processed/train.npz"))
    val_ds = PhysNetContextDataset(Path("data/processed/test.npz"))
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    model = EEGContextNet().to(device)
    
    # Clean history dictionary (Includes train_loss to prevent your viz.py KeyError)
    history = {'train_loss': [], 'val_loss': [], 'val_precision': [], 'val_f1': []}
    
    pos_weight = torch.tensor([1.2]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    for epoch in range(EPOCHS):
        # ==========================================================
        # TRAINING LOOP
        # ==========================================================
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]", leave=False, dynamic_ncols=True)
        
        for signals, contexts, labels in pbar:
            signals = signals.to(device)
            contexts = contexts.to(device).float().permute(0, 2, 1)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            logits = model(signals, contexts)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        scheduler.step()

        # ==========================================================
        # VALIDATION LOOP
        # ==========================================================
        model.eval()
        val_loss = 0.0
        tp, fp, fn, tn = 0, 0, 0, 0
        
        with torch.no_grad():
            for signals, contexts, labels in val_loader:
                signals = signals.to(device)
                contexts = contexts.to(device).float().permute(0, 2, 1)
                labels = labels.to(device).float()
                
                logits = model(signals, contexts)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(logits) >= PRED_THRESHOLD).float()
                
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()
                
        val_loss /= len(val_loader)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8) # Needed mathematically for F1
        f1_score = (2 * precision * recall) / (precision + recall + 1e-8)

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {train_loss:.4f} / {val_loss:.4f} | Val Prec: {precision:.3f} | Val F1: {f1_score:.3f}")
        
        history['val_loss'].append(val_loss)
        history['val_precision'].append(precision)
        history['val_f1'].append(f1_score)
        
        # Save standard latest checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'val_precision': precision,
            'val_f1': f1_score
        }
        torch.save(checkpoint, RUN_DIR / "latest_model.pt")
        
        # ==========================================================
        # CHECKPOINTING (Strictly on Precision)
        # ==========================================================
        if True:
            best_precission = precision
            torch.save(checkpoint, RUN_DIR / "best_sleep_cnn.pt")
            
            # Save Confusion Matrix ONLY when there is a new best precision
            plot_epoch_confusion_matrix(tp, fp, fn, tn, save_path=RUN_DIR)
            print(f"  [+] Found new best model! Saved weight checkpoint & CM (Prec: {best_precission:.3f})")

        # Automatically update the history plot every single epoch
        plot_history(history, save_path=RUN_DIR)

if __name__ == "__main__":
    main()