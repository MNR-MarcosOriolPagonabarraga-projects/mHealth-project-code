import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from src.networks.stft_cnn import StftArousalNet
from src.dataset import PhysNetStftDataset 
from src.viz import plot_history

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Training on device: {device.type.upper()}")

    # Dedicated timestamped directories for your 2D runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = Path("models") / "stft_net" / timestamp
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Saving artifacts to: {RUN_DIR}")
    
    best_acc = -1.0  # Changed from best_f1 to track accuracy

    print("[*] Loading STFT datasets into RAM...")
    train_ds = PhysNetStftDataset(Path("data/processed/train.npz"))
    val_ds = PhysNetStftDataset(Path("data/processed/test.npz"))
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    # Lightweight 2D Embedded Model
    model = StftArousalNet().to(device)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    num_pos = (train_ds.labels == 1).sum()
    num_neg = (train_ds.labels == 0).sum()
    ratio = torch.tensor([num_neg / num_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=ratio).to(device)
    
    EPOCHS = 50
    PRINT_EVERY = 1 
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]", leave=False, dynamic_ncols=True)
        
        for signals, labels in pbar:
            signals, labels = signals.to(device), labels.to(device).float()

            optimizer.zero_grad()
            logits = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Calculate batch accuracy safely (view_as prevents hidden broadcasting bugs)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            batch_correct = (preds.view_as(labels) == labels).sum().item()
            batch_acc = batch_correct / labels.size(0)
            
            train_loss += loss.item()
            train_correct += batch_correct
            train_total += labels.size(0)
            
            # Display accuracy in the tqdm bar instead of loss
            pbar.set_postfix({'acc': f"{batch_acc*100:.1f}%"})
            
        train_loss /= len(train_loader)
        train_epoch_acc = train_correct / train_total
        history['train_acc'].append(train_epoch_acc)
        scheduler.step()

        if (epoch + 1) % PRINT_EVERY == 0 or (epoch + 1) == EPOCHS:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for signals, labels in val_loader:
                    signals, labels = signals.to(device), labels.to(device).float()
                    
                    logits = model(signals)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    
                    # Clean accuracy calculation
                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    val_correct += (preds.view_as(labels) == labels).sum().item()
                    val_total += labels.size(0)
                    
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total if val_total > 0 else 0.0

            # Only printing Loss and Accuracy
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc*100:.1f}%")
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }
            
            torch.save(checkpoint, RUN_DIR / "latest_model.pt")
            
            # Save based on best accuracy now
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_path = RUN_DIR / "best_sleep_cnn.pt"
                torch.save(checkpoint, best_model_path)
                print(f"  [+] Found new best model! Saved weight checkpoint to {best_model_path}")

    plot_history(history, save_path=RUN_DIR)


if __name__ == "__main__":
    main()