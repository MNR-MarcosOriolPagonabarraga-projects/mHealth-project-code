import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from src.networks.dual_branch import EEGContextNet
from src.dataset import PhysNetContextDataset 
from src.viz import plot_history, plot_epoch_confusion_matrix

def apply_time_masking(signals, max_mask_pct=0.15):
    """
    Applies random temporal cutout to 1D signals to prevent overfitting.
    Expects shape: [batch, channels, time_steps]
    """
    masked_signals = signals.clone()
    batch_size, channels, time_steps = masked_signals.shape
    
    for i in range(batch_size):
        mask_t = int(time_steps * max_mask_pct)
        if mask_t > 0:
            # Pick a random starting point for the mask
            t0 = torch.randint(0, time_steps - mask_t, (1,)).item()
            # Zero out the time window across all channels
            masked_signals[i, :, t0:t0+mask_t] = 0.0
            
    return masked_signals

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Training on device: {device.type.upper()}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = Path("models") / "context_net" / timestamp
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Saving artifacts to: {RUN_DIR}")
    
    # Track F2 score instead of accuracy for highly imbalanced data
    best_f2 = -1.0  

    print("[*] Loading datasets into RAM...")
    train_ds = PhysNetContextDataset(Path("data/processed/train.npz"))
    val_ds = PhysNetContextDataset(Path("data/processed/test.npz"))
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    model = EEGContextNet().to(device)
    history = {'train_loss': [], 'val_loss': [], 'val_f2': [], 'val_precision': [], 'val_recall': []}
    
    criterion = nn.BCEWithLogitsLoss().to(device)
    
    EPOCHS = 50
    PRINT_EVERY = 1 
    
    # Slightly higher starting LR for optimal convergence, decayed over 50 epochs
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]", leave=False, dynamic_ncols=True)
        
        for signals, contexts, labels in pbar:
            signals, contexts, labels = signals.to(device), contexts.to(device).float().permute(0, 2, 1), labels.to(device).float()

            optimizer.zero_grad()
            
            logits = model(signals, contexts)
            
            # Standard BCE loss calculation
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        scheduler.step()

        if (epoch + 1) % PRINT_EVERY == 0 or (epoch + 1) == EPOCHS:
            model.eval()
            val_loss = 0.0
            
            # Confusion matrix trackers
            tp, fp, fn, tn = 0, 0, 0, 0
            
            with torch.no_grad():
                for signals, contexts, labels in val_loader:
                    signals, contexts, labels = signals.to(device), contexts.to(device).float().permute(0, 2, 1), labels.to(device).float()
                    
                    logits = model(signals, contexts)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    
                    # Compute predictions
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    
                    # Update confusion matrix
                    tp += ((preds == 1) & (labels == 1)).sum().item()
                    fp += ((preds == 1) & (labels == 0)).sum().item()
                    fn += ((preds == 0) & (labels == 1)).sum().item()
                    tn += ((preds == 0) & (labels == 0)).sum().item()
                    
            val_loss /= len(val_loader)
            
            # Calculate clinical metrics
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            # F2 Score formulation (beta=2, prioritizes recall)
            f2_score = (5 * precision * recall) / (4 * precision + recall + 1e-8)

            print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
                  f"Loss: {train_loss:.4f} / {val_loss:.4f} | "
                  f"Val Rec: {recall:.3f} | Val Prec: {precision:.3f} | Val F2: {f2_score:.3f}")
            
            history['val_loss'].append(val_loss)
            history['val_precision'].append(precision)
            history['val_recall'].append(recall)
            history['val_f2'].append(f2_score)
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f2': f2_score,
                'val_loss': val_loss
            }
            
            torch.save(checkpoint, RUN_DIR / "latest_model.pt")
            
            # Save strictly based on F2 score performance
            if f2_score > best_f2:
                best_f2 = f2_score
                best_model_path = RUN_DIR / "best_sleep_cnn.pt"
                torch.save(checkpoint, best_model_path)
                plot_epoch_confusion_matrix(tp, fp, fn, tn, save_path=RUN_DIR)
                print(f"  [+] Found new best model! Saved weight checkpoint (F2: {best_f2:.3f})")

        plot_history(history, save_path=RUN_DIR)

if __name__ == "__main__":
    main()