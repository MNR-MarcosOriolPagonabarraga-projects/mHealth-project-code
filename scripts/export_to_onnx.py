import torch
from src.networks.sleep_stage import EnhancedTinyConvNet

def export_to_onnx():
    # 1. Initialize and load your best weights
    model = EnhancedTinyConvNet()
    model.load_state_dict(torch.load("models/sleep_stage_mlp/best_sleep_mlp.pt"))
    model.eval()

    # 2. Create a dummy input matching your exact feature shape
    # (Batch Size, Features/Channels, Time Steps) -> (1, 30, 30)
    dummy_input = torch.randn(1, 30, 30)

    # 3. Export to ONNX
    onnx_path = "sleep_model.onnx"
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
    export_to_onnx()