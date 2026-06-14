import sys
import wfdb
import numpy as np
import os
import glob

data_dir = sys.argv[1]

# Define the exact channels your pipeline expects
TARGET_CHANNELS = ["C3-M2", "C4-M1"]

# Find all header files
records = glob.glob(os.path.join(data_dir, "*/*.hea"))

for hea_file in records:
    # wfdb expects the path without the extension
    record_name = hea_file.replace('.hea', '')
    print(f"[*] Processing: {record_name}")
    
    # Read the WFDB record
    record = wfdb.rdrecord(record_name)
    
    # Find the indices of the target channels
    try:
        channel_indices = [record.sig_name.index(ch) for ch in TARGET_CHANNELS]
    except ValueError as e:
        print(f"  [!] Skipping {record_name}: Missing required channels. ({e})")
        continue
        
    # Extract ONLY the required channels
    # record.p_signal is shape (samples, total_channels)
    # signals becomes shape (samples, 2)
    signals = record.p_signal[:, channel_indices]
    
    # Ensure it's 32-bit float (matches Zig's f32) and save as raw binary
    bin_path = record_name + ".bin"
    signals.astype(np.float32).tofile(bin_path)
    print(f"  [+] Saved {bin_path} with shape {signals.shape}")
    
print("[+] All arrays extracted to .bin files!")