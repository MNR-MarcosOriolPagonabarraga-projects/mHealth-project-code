import torch
import numpy as np
from src.networks.sleep_stage_net import SleepStageNet

# 1. Initialize and load your PyTorch model
model = SleepStageNet()
checkpoint = torch.load("models/sleep_stage/sleep_stage_detector.pt", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
model.eval()

# 2. Create a deterministic dummy input (1, 20, 60)
input_array = np.arange(1200, dtype=np.float32).reshape(1, 20, 60) * 0.01

# 3. Save this input array as a raw binary file for your C test
input_array.tofile("outputs/dummy_sleep_input.bin")

# 4. Run PyTorch inference
with torch.no_grad():
    input_tensor = torch.from_numpy(input_array)
    py_logits = model(input_tensor).squeeze().numpy()

print("--- PyTorch Expected Reference Logits ---")
print(f"Wake:        {py_logits[0]:.6f}")
print(f"Light Sleep: {py_logits[1]:.6f}")
print(f"Deep Sleep:  {py_logits[2]:.6f}")
print(f"REM:         {py_logits[3]:.6f}")

# --- ADD THIS: Save the logits for plotting ---
py_logits.astype(np.float32).tofile("outputs/py_logits.bin")