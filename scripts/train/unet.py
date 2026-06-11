import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from src.networks.micro_unet import MicroSleepArousalUNet, BCEDiceLoss
from offline.dataset import PhysNetSignalDataset 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Training on device: {device.type.upper()}")

    # --- Setup Dedicated Timestamped Directories ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = Path("models") / "unet" / timestamp
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Checkpoints and logs will be saved to: {RUN_DIR}")
    
    best_f1 = -1.0  

    # Load Data
    print("[*] Loading datasets into RAM...")
    train_ds = PhysNetSignalDataset(Path("data/processed/train.npz"))
    val_ds = PhysNetSignalDataset(Path("data/processed/test.npz"))
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Instantiate the corrected network architecture
    model = MicroSleepArousalUNet(bottleneck=512).to(device)
    criterion = BCEDiceLoss(pos_weight_scalar=1.0).to(device)
    
    EPOCHS = 60
    PRINT_EVERY = 5
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]", leave=False, dynamic_ncols=True)
        
        for signals, labels in pbar:
            signals, labels = signals.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        scheduler.step()

        # Run Evaluation
        if (epoch + 1) % PRINT_EVERY == 0 or (epoch + 1) == EPOCHS:
            model.eval()
            val_loss = 0.0
            
            tp = torch.tensor(0, dtype=torch.int64, device=device)
            fp = torch.tensor(0, dtype=torch.int64, device=device)
            tn = torch.tensor(0, dtype=torch.int64, device=device)
            fn = torch.tensor(0, dtype=torch.int64, device=device)
            
            with torch.no_grad():
                for signals, labels in val_loader:
                    signals, labels = signals.to(device), labels.to(device)
                    
                    logits = model(signals)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()
                    
                    # Compute predictions via a safe 0.5 probability threshold
                    preds = (torch.sigmoid(logits) >= 0.5).long()
                    targets = labels.long()
                    
                    # Vectorized metrics calculated natively on the GPU
                    tp += (preds & targets).sum()
                    fp += (preds & ~targets).sum()
                    tn += (~preds & ~targets).sum()
                    fn += (~preds & targets).sum()
                    
            val_loss /= len(val_loader)

            # Extract counts safely out of the GPU context
            TP, FP, TN, FN = tp.item(), fp.item(), tn.item(), fn.item()
            total_eval_points = TP + FP + TN + FN

            # Scalar structural calculations
            val_acc = (TP + TN) / total_eval_points if total_eval_points > 0 else 0.0
            val_recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            val_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            val_f1 = (2 * val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0.0

            print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc*100:.1f}% | "
                  f"Val Recall: {val_recall*100:.1f}% | "
                  f"Val F1: {val_f1:.3f}")
            
            # Save structures
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss
            }
            
            # Regular checkpoint backup into timestamp folder
            torch.save(checkpoint, RUN_DIR / "latest_model.pt")
            
            # Track and save best performing configurations
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_path = RUN_DIR / "best_sleep_unet.pt"
                torch.save(checkpoint, best_model_path)
                print(f"  [+] Found new best model! Saved weight checkpoint to {best_model_path}")
                
        else:
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val: Skipped metrics")


if __name__ == "__main__":
    main()