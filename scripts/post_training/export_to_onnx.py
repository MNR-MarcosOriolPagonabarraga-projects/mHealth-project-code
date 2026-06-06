
import os
import sys
import torch
from src.networks.sleep_stage import LowPowerConvNet

def export_to_onnx(model_path):
    # Initialize and load best weights
    model = LowPowerConvNet()
    checkpoint = torch.load(model_path, map_location="cpu") # map_location keeps it safe if trained on CUDA
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # Dummy input
    # (Batch Size, Features/Channels, Time Steps) -> (1, 30, 30)
    dummy_input = torch.randn(1, 120, 30)

    # Export to ONNX
    output_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path).split(".")[0]
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input_features'],
        output_names=['sleep_stage_logits']
    )
    print(f"[+] Successfully exported model to {onnx_path}")

if __name__ == "__main__":
    model_path = sys.argv[1]
    export_to_onnx(model_path)