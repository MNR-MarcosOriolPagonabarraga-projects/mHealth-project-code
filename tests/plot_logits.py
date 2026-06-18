import numpy as np
import matplotlib.pyplot as plt

def plot_logits_comparison():
    # 1. Load the logits from both models
    try:
        py_logits = np.fromfile("outputs/py_logits.bin", dtype=np.float32)
        c_logits = np.fromfile("outputs/c_logits.bin", dtype=np.float32)
    except FileNotFoundError as e:
        print(f"Error loading files. Ensure you have run both models first. {e}")
        return

    stages = ['Wake', 'Light Sleep', 'Deep Sleep', 'REM']
    x = np.arange(len(stages))
    width = 0.35  # width of the bars

    # Calculate absolute differences
    abs_diff = np.abs(py_logits - c_logits)

    # 2. Setup the minimalist, publication-style canvas
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    
    color_py = '#4F46E5'  # Indigo
    color_c = '#06B6D4'   # Cyan

    # Plot the grouped bars
    rects1 = ax.bar(x - width/2, py_logits, width, label='PyTorch (.pt)', color=color_py, alpha=0.9)
    rects2 = ax.bar(x + width/2, c_logits, width, label='Compiled C (.c)', color=color_c, alpha=0.9)

    # 3. Apply Minimalistic Styling
    # Remove all spines (borders)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Remove tick marks but keep labels
    ax.tick_params(axis='both', which='both', length=0, colors='#666666', labelsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontweight='bold', color='#333333')
    
    # Add a faint horizontal grid
    ax.grid(axis='y', linestyle='-', alpha=0.1, color='black')

    # 4. Annotate the differences directly above the bars
    for i in range(len(stages)):
        # Find the highest bar in the group to position the text slightly above it
        max_y = max(py_logits[i], c_logits[i])
        
        # Determine offset direction based on whether value is positive or negative
        y_offset = 0.5 if max_y > 0 else -0.5
        va = 'bottom' if max_y > 0 else 'top'
        
        ax.text(x[i], max_y + y_offset, f"Diff: {abs_diff[i]:.4f}", 
                ha='center', va=va, fontsize=10, color='#E11D48', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    # Add title and legend
    ax.set_title("Logit Output Comparison: PyTorch vs. INT8 Quantized C-Model", 
                 fontsize=14, fontweight='bold', color='#333333', pad=20)
    ax.legend(frameon=False, loc='best', fontsize=11, labelcolor='#333333')

    plt.axhline(0, color='#333333', linewidth=1.2, alpha=0.8) # Stronger baseline at 0
    plt.tight_layout()
    plt.savefig('outputs/logits_comparison.png', dpi=300)

if __name__ == "__main__":
    plot_logits_comparison()