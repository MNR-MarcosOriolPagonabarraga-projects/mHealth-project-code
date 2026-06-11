import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

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
    

def plot_epoch_confusion_matrix(tp, fp, fn, tn, save_path):
    """
    Generates and saves a clean Seaborn heatmap of the confusion matrix for a specific epoch.
    """
    # Construct the standard 2x2 confusion matrix array
    # Format: [[TN, FP], [FN, TP]]
    cm = np.array([[tn, fp], [fn, tp]])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Generate labels containing both raw count and relative percentage
    total_samples = np.sum(cm) + 1e-8
    labels = [
        [f"{tn}\n({tn/total_samples*100:.1f}%)", f"{fp}\n({fp/total_samples*100:.1f}%)"],
        [f"{fn}\n({fn/total_samples*100:.1f}%)", f"{tp}\n({tp/total_samples*100:.1f}%)"]
    ]
    labels = np.array(labels)
    
    # Plot using a clean, readable color palette (Blues highlight true predictions nicely)
    sns.heatmap(
        cm, 
        annot=labels, 
        fmt="", 
        cmap="Blues", 
        cbar=False, 
        ax=ax,
        xticklabels=["No Arousal", "Arousal"],
        yticklabels=["No Arousal", "Arousal"],
        annot_kws={"size": 11, "weight": "bold"}
    )
    
    ax.set_title(f"Confusion Matrix", fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
    ax.set_ylabel("True Ground-Truth Label", fontsize=11, labelpad=10)
    
    plt.tight_layout()
    # Save with a sequential naming format inside your timestamped RUN_DIR
    plt.savefig(save_path / f"confusion_matrix.png", dpi=150)
    plt.close(fig)