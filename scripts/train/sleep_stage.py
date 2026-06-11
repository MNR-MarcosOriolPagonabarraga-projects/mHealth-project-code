import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.networks.sleep_stage import LowPowerConvNet
from offline.viz import plot_history

# Global Training Configs
BATCH_SIZE = 256
EPOCHS = 40
LEARNING_RATE = 7e-4
STAGE_NAMES = ['Wake', 'Light Sleep', 'Deep Sleep', 'REM']

def plot_epoch_confusion_matrix(all_targets, all_preds, stage_names, save_path: Path):
    """
    Generates and saves a clean Seaborn heatmap of a multi-class confusion matrix.
    Works dynamically for any number of classes.
    """
    # 1. Compute the N x N confusion matrix dynamically
    cm = confusion_matrix(all_targets, all_preds)
    num_classes = len(stage_names)
    
    # 2. Generate text annotations (Raw Count + Percentage relative to the True Class row)
    # Using row-sums (axis=1) shows the accuracy per individual sleep stage
    row_sums = cm.sum(axis=1, keepdims=True) + 1e-8
    cm_percentages = (cm / row_sums) * 100
    
    labels = np.empty_like(cm, dtype=object)
    for r in range(num_classes):
        for c in range(num_classes):
            labels[r, c] = f"{cm[r, c]:,d}\n({cm_percentages[r, c]:.1f}%)"
            
    # 3. Render the Plot
    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
    
    sns.heatmap(
        cm, 
        annot=labels, 
        fmt="", 
        cmap="Blues", 
        cbar=True, 
        ax=ax,
        xticklabels=stage_names,
        yticklabels=stage_names,
        annot_kws={"size": 9, "weight": "bold"}
    )
    
    # Custom adjustments for readability
    ax.set_title("Sleep Stage Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Predicted Stage", fontsize=11, labelpad=10)
    ax.set_ylabel("True Ground-Truth Stage", fontsize=11, labelpad=10)
    
    # Rotate tick labels so they don't overlap
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save directly to your active run directory
    save_file = save_path / "best_model_confusion_matrix.png"
    plt.savefig(save_file, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Saved breakthrough confusion matrix to: {save_file.name}")

def apply_time_masking(windows, max_mask_pct=0.15):
    """
    Applies random temporal cutout to (Batch, Time, Features) tensors to prevent overfitting.
    """
    masked_windows = windows.clone()
    batch_size, time_steps, n_features = masked_windows.shape
    
    for i in range(batch_size):
        mask_t = int(time_steps * max_mask_pct)
        if mask_t > 0:
            # Pick a random starting point for the temporal mask
            t0 = torch.randint(0, time_steps - mask_t, (1,)).item()
            # Zero out the block across all frequency features
            masked_windows[i, t0:t0+mask_t, :] = 0.0
            
    return masked_windows

def load_data(split_name):
    print(f"[*] Loading {split_name} dataset into RAM...")
    # Matches the specific file names specified in your notebook sample
    file_path = Path("data/processed") / f"{split_name}.npz"
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find processed dataset at {file_path}")
        
    data = np.load(file_path)
    
    # Safe key extraction matching your processed naming mapping
    X = data['spectral_band_windows']
    y_onehot = data['sleep_stages']
    y = np.argmax(y_onehot, axis=1)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    return X_tensor, y_tensor

def main():
    # Setup execution device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Training on device: {device.type.upper()}")

    # Setup unique output directory matching your reference pipeline
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = Path("models") / "sleep_stage_mlp" / timestamp
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Saving artifacts to: {RUN_DIR}")
    
    # Initialize metric tracking
    best_acc = -1.0  
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'val_acc': [], 
        'val_macro_f1': []
    }

    # 1. Pipeline Datasets & Loaders
    X_train, y_train = load_data("sleep_stage_train")
    X_test, y_test = load_data("sleep_stage_test")
    computed_weights = torch.tensor([1.1, 0.7, 1.5, 1.5]).to(device)
    
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        drop_last=True, 
        num_workers=2, 
        pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(X_test, y_test), 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )

    # 2. Network Components
    model = LowPowerConvNet().to(device)
    criterion = nn.CrossEntropyLoss(weight=computed_weights).to(device)
    
    # Optimization scheduling
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # 3. Main Training Execution Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]", leave=False, dynamic_ncols=True)
        
        for batch_X, batch_y in pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Apply TinyML on-the-fly regularizing time mask augmentation
            batch_X = apply_time_masking(batch_X, max_mask_pct=0.15)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            
            # Prevent exploding gradients through standard clipping threshold
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        scheduler.step()

        # 4. Evaluation Sequence (Every Epoch)
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()
                
                _, preds = torch.max(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                
        val_loss /= len(val_loader)
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate balanced Multi-class classification metrics
        epoch_acc = accuracy_score(all_targets, all_preds)
        epoch_f1 = f1_score(all_targets, all_preds, average='macro')

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} / {val_loss:.4f} | "
              f"Val Acc: {epoch_acc:.4f} | Val Macro F1: {epoch_f1:.4f}")
        
        # Keep metrics dictionary updated for history logging
        history['val_loss'].append(val_loss)
        history['val_acc'].append(epoch_acc)
        history['val_macro_f1'].append(epoch_f1)
        
        # Establish checkpoint serialization payload
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': epoch_acc,
            'val_loss': val_loss
        }
        
        # Keep latest model up-to-date
        torch.save(checkpoint, RUN_DIR / "latest_model.pt")
        
        # Track and save best model strictly looking at validation accuracy
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(checkpoint, RUN_DIR / "best_sleep_mlp.pt")
            print(f"  [+] Found new best model! Saved weight checkpoint (Acc: {best_acc:.4f})")
            plot_epoch_confusion_matrix(all_targets, all_preds, STAGE_NAMES, save_path=RUN_DIR)

        plot_history(history, save_path=RUN_DIR)

if __name__ == "__main__":
    main()