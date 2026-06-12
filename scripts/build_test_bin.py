import sys
import wfdb
import numpy as np
import os
import glob

data_dir = sys.argv[1]

# Find all header files
records = glob.glob(os.path.join(data_dir, "*/*.hea"))

for hea_file in records:
    # wfdb expects the path without the extension
    record_name = hea_file.replace('.hea', '')
    
    print(f"Processing: {record_name}")
    
    # Read the WFDB record
    record = wfdb.rdrecord(record_name)
    print(record)
    
    # Extract the signal array (usually time x channels)
    signals = record.p_signal 
    
    # Ensure it's 32-bit float (matches Zig's f32) and save as raw binary
    bin_path = record_name + ".bin"
    signals.astype(np.float32).tofile(bin_path)
    
print("All arrays extracted to .bin files!")