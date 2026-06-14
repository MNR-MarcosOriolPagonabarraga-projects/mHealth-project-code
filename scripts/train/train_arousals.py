import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from src.networks.arousal_net import ArousalNet, WeightedFocalWithLogitsLoss
from src.offline.dataset import ArousalsDataset 
from src.config import TrainArousalsConfig
from src.offline.trainer import ModelTrainer

def main():
    cfg = TrainArousalsConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    out_model_path = Path(cfg.out_path)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)

    # Set loaders
    train_ds = ArousalsDataset(Path(cfg.train_path), is_train=True)
    val_ds = ArousalsDataset(Path(cfg.test_path), is_train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # Define Model and training workers
    model = ArousalNet().to(device)
    criterion = WeightedFocalWithLogitsLoss(pos_weight=cfg.train_balance, gamma=2.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min)

    # Train the model
    trainer = ModelTrainer(
        model=model, criterion=criterion, optimizer=optimizer, 
        scheduler=scheduler, device=device, config=cfg, task_type="binary"
    )
    trainer.fit(train_loader, val_loader, out_model_path, class_names=cfg.class_names)

if __name__ == "__main__":
    main()