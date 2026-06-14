import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, balanced_accuracy_score
from src.offline.viz import plot_history, plot_epoch_confusion_matrix

class ModelTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, config, task_type):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.task_type = task_type # "binary" or "multiclass"
        
        # Track both Accuracy and F1 in history
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
        self.best_metric = -1.0

    def _get_predictions(self, logits):
        """Abstracts the difference between binary and multiclass prediction."""
        if self.task_type == "binary":
            return (torch.sigmoid(logits) >= self.config.pred_threshold).float()
        else:
            _, preds = torch.max(logits, dim=1)
            return preds

    def _compute_metrics(self, targets, preds):
        """Calculates all metrics and tags the primary one used for checkpointing."""
        bal_acc = balanced_accuracy_score(targets, preds)
        
        if self.task_type == "binary":
            f1 = f1_score(targets, preds, zero_division=0)
            return {'primary': bal_acc, 'acc': bal_acc, 'f1': f1}
        else:
            f1 = f1_score(targets, preds, average='macro', zero_division=0)
            return {'primary': bal_acc, 'acc': bal_acc, 'f1': f1}

    def fit(self, train_loader, val_loader, out_model_path, class_names):
        for epoch in range(self.config.epochs):
            # --- TRAIN ---
            self.model.train()
            train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{self.config.epochs}", leave=False)
            
            for batch in pbar:
                inputs = [b.to(self.device) for b in batch[:-1]]
                targets = batch[-1].to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(*inputs)
                loss = self.criterion(logits, targets.float() if self.task_type == 'binary' else targets.long())
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item()
                
            self.history['train_loss'].append(train_loss / len(train_loader))
            self.scheduler.step()

            # --- VALIDATE ---
            self.model.eval()
            val_loss = 0.0
            all_preds, all_targets = [], []
            
            with torch.no_grad():
                for batch in val_loader:
                    inputs = [b.to(self.device) for b in batch[:-1]]
                    targets = batch[-1].to(self.device)
                    
                    logits = self.model(*inputs)
                    loss = self.criterion(logits, targets.float() if self.task_type == 'binary' else targets.long())
                    val_loss += loss.item()
                    
                    preds = self._get_predictions(logits)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    
            # Extract the dictionary of metrics
            metrics = self._compute_metrics(all_targets, all_preds)
            
            self.history['val_loss'].append(val_loss / len(val_loader))
            self.history['val_acc'].append(metrics['acc'])
            self.history['val_f1'].append(metrics['f1'])
            
            # Print the comprehensive breakdown
            print(f"Epoch {epoch+1:02d}/{self.config.epochs} | "
                  f"Loss (Tr/Vl): {self.history['train_loss'][-1]:.4f}/{self.history['val_loss'][-1]:.4f} | "
                  f"Acc: {metrics['acc']:.4f} | F1: {metrics['f1']:.4f}")

            # --- CHECKPOINTING ---
            # We use metrics['primary'] (Precision for binary, Acc for multiclass) to save the best weights
            run_dir = out_model_path.parent
            if metrics['primary'] > self.best_metric and metrics['f1'] > 0.0:
                self.best_metric = metrics['primary']
                torch.save(self.model.state_dict(), out_model_path)
                plot_epoch_confusion_matrix(all_targets, all_preds, class_names, run_dir)
                
            plot_history(self.history, run_dir)
