import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_history(history: dict, save_path: Path):
    """
    Universally plots training history metrics.
    Automatically detects loss keys and any other evaluation metrics present.
    """
    first_key = list(history.keys())[0]
    epochs = range(1, len(history[first_key]) + 1)

    loss_keys = [k for k in history.keys() if 'loss' in k]
    metric_keys = [k for k in history.keys() if 'loss' not in k]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

    if loss_keys:
        for key in loss_keys:
            clean_label = key.replace('_', ' ').title()
            ax1.plot(epochs, history[key], marker='o', label=clean_label, linewidth=1.5)
            
        ax1.set_title("Model Loss Progression", fontsize=12, fontweight='bold', pad=10)
        ax1.set_xlabel("Epochs", fontsize=10)
        ax1.set_ylabel("Loss Value", fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(fontsize=9)
    else:
        ax1.text(0.5, 0.5, "No Loss Metrics Tracked", ha='center', va='center', fontsize=12)
        ax1.axis('off')
    
    if metric_keys:
        for key in metric_keys:
            clean_label = key.replace('_', ' ').title().replace('Val ', 'Validation ')
            
            values = history[key]
            if all(0.0 <= v <= 1.01 for v in values if v is not None):
                values = [v * 100 for v in values]
                y_label = "Score (%)"
            else:
                y_label = "Value"
                
            ax2.plot(epochs, values, marker='s', label=clean_label, linewidth=1.5)
            
        ax2.set_title("Validation Performance Metrics", fontsize=12, fontweight='bold', pad=10)
        ax2.set_xlabel("Epochs", fontsize=10)
        ax2.set_ylabel(y_label, fontsize=10)
        
        if y_label == "Score (%)":
            ax2.set_ylim(-5, 105)
            
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No Validation Metrics Tracked", ha='center', va='center', fontsize=12)
        ax2.axis('off')
        
    plt.suptitle("Model Training History Summary", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_file = save_path / "training_history_curves.png"
    plt.savefig(save_file, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved training evolution curves to: {save_file.name}")
    

def plot_epoch_confusion_matrix(targets, preds, class_names, save_path: Path):
    """
    Universal Confusion Matrix. Works for Binary (Arousals) and Multiclass (Sleep Stages).
    """
    num_classes = len(class_names)
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))
    
    row_sums = cm.sum(axis=1, keepdims=True) + 1e-8
    cm_percentages = (cm / row_sums) * 100
    
    labels = np.empty_like(cm, dtype=object)
    for r in range(num_classes):
        for c in range(num_classes):
            labels[r, c] = f"{cm[r, c]:,d}\n({cm_percentages[r, c]:.1f}%)"
            
    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)
    
    # FIX: Pass cm_percentages to drive the color, and lock the scale to 0-100
    sns.heatmap(cm_percentages, annot=labels, fmt="", cmap="Blues", cbar=True, ax=ax,
                vmin=0, vmax=100, 
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 9, "weight": "bold"})
    
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
    ax.set_ylabel("True Ground-Truth Label", fontsize=11, labelpad=10)
    
    plt.tight_layout()
    plt.savefig(save_path / "best_model_confusion_matrix.png", bbox_inches='tight')
    plt.close(fig)