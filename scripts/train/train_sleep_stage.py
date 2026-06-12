import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from src.networks.sleep_stage_net import LowPowerConvNet
from src.offline.dataset import SleepStageDataset 
from src.offline.config import TrainSleepStagesConfig
from src.offline.trainer import ModelTrainer

def main():
    cfg = TrainSleepStagesConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    RUN_DIR = Path(cfg.out_path)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Data
    train_ds = SleepStageDataset(Path(cfg.train_path))
    val_ds = SleepStageDataset(Path(cfg.test_path))
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)

    # 2. Components
    model = LowPowerConvNet().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([cfg.train_balance]).squeeze()).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min)

    # 3. Execute
    trainer = ModelTrainer(
        model=model, criterion=criterion, optimizer=optimizer, 
        scheduler=scheduler, device=device, config=cfg, task_type="multiclass"
    )
    
    trainer.fit(train_loader, val_loader, RUN_DIR, class_names=cfg.stage_names)

if __name__ == "__main__":
    main()