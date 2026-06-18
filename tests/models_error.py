import numpy as np
import matplotlib.pyplot as plt

def plot_flattened_comparison(py_file: str, zig_file: str):
    # 1. Load and flatten the raw text files
    try:
        py_tensor = np.loadtxt(py_file)[10:20, -10:].T.flatten()
        zig_tensor = np.loadtxt(zig_file)[10:20, -10:].T.flatten()
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Ensure shapes match
    if py_tensor.shape != zig_tensor.shape:
        print(f"Shape mismatch! Python: {py_tensor.shape}, Zig: {zig_tensor.shape}")
        return

    # 2. Calculate the Percentage Error (ignoring true zeros)
    abs_diff = np.abs(py_tensor - zig_tensor)
    active_mask = py_tensor != 0.0
    
    if np.any(active_mask):
        pct_errors = (abs_diff[active_mask] / np.abs(py_tensor[active_mask])) * 100
        mean_pct_error = np.mean(pct_errors)
    else:
        mean_pct_error = 0.0

    # 3. Setup the minimalist, publication-style canvas
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    
    # Modern cool color palette
    color_py = '#4F46E5'  # Indigo (Reference)
    color_zig = '#06B6D4' # Cyan (Implementation)

    # Plot the flattened signals
    ax.plot(py_tensor, label='Python (Reference)', color=color_py, linewidth=1.5, alpha=0.9)
    ax.plot(zig_tensor, label='Zig (Implementation)', color=color_zig, linewidth=1.2, linestyle='--', alpha=0.9)

    # 4. Apply Minimalistic Styling
    # Remove all spines (borders)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Remove tick marks but keep the labels in a soft gray
    ax.tick_params(axis='both', which='both', length=0, colors='#666666', labelsize=10)
    
    # Add a very faint, elegant horizontal grid to guide the eye
    ax.grid(axis='y', linestyle='-', alpha=0.1, color='black')

    # 5. Add the Error Label Text
    error_text = f"Average Error: {mean_pct_error:.2f}%"
    ax.text(0.01, 0.95, error_text, transform=ax.transAxes, 
            fontsize=13, fontweight='bold', color='#333333', 
            va='top', ha='left', 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5))

    # Add a clean, frameless legend
    ax.legend(frameon=False, loc='upper right', fontsize=11, labelcolor='#333333')

    plt.tight_layout()

if __name__ == "__main__":
    # Ensure these paths point to where your generated files are located
    plot_flattened_comparison("outputs/py_state_sleep_tensor_raw.txt", "outputs/zig_state_sleep_tensor_raw.txt")
    plt.savefig("outputs/models_difference.png", dpi=300)