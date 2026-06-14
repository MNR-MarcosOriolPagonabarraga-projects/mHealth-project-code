import os
import argparse
import torch
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader

from src.config import PreprocessConfig, TrainArousalsConfig, TrainSleepStagesConfig

# Load both model architectures
from src.networks.arousal_net import ArousalNet 
from src.networks.sleep_stage_net import SleepStageNet  # Assuming this is your class name

# ==========================================
# 1. GENERIC CALIBRATION DATA READER
# ==========================================
class GenericCalibrator(CalibrationDataReader):
    def __init__(self, data_path, input_names, dataset_keys):
        """
        Dynamically maps keys from the .npz dataset to ONNX input names.
        """
        dataset = np.load(data_path)
        
        # Extract a small calibration batch (e.g., 200 samples)
        extracted_data = [dataset[key][:200].astype(np.float32) for key in dataset_keys]
        
        # Zip them together to iterate over paired/single inputs easily
        paired_data = zip(*extracted_data)
        
        formatted_data = []
        for sample_tuple in paired_data:
            feed_dict = {}
            for i, onnx_name in enumerate(input_names):
                # Add batch dimension
                arr = np.expand_dims(sample_tuple[i], axis=0)
                
                # If this is the arousal context window, apply your required transpose
                if onnx_name in ["context_input", "sleep_features"]:
                    arr = arr.transpose(0, 2, 1)
                    
                feed_dict[onnx_name] = arr
            formatted_data.append(feed_dict)

        self.enum_data = iter(formatted_data)

    def get_next(self):
        return next(self.enum_data, None)

# ==========================================
# 2. EXPORT WORKER FUNCTION
# ==========================================
def export_and_quantize_single(model, model_path, dataset_path, dummy_input, input_names, output_names, dataset_keys):
    """Handles the actual PyTorch to ONNX export and INT8 quantization for a single model."""
    
    if not os.path.exists(model_path):
        print(f"[!] Error: Model checkpoint not found at {model_path}")
        print("    Please ensure you have trained the model first.")
        return

    # Load weights
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    # Define output paths next to the original .pt model
    output_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path).split(".")[0]
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    quantized_onnx_path = os.path.join(output_dir, f"{model_name}_int8.onnx")

    # --- STEP A: EXPORT TO ONNX ---
    print(f"    [*] Exporting base PyTorch model to ONNX...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names
    )
    print(f"    [+] Saved base model: {onnx_path}")

    # --- STEP B: QUANTIZE TO INT8 ---
    print(f"    [*] Calibrating and Quantizing model to INT8...")
    calibrator = GenericCalibrator(dataset_path, input_names, dataset_keys)
    
    quantize_static(
        model_input=onnx_path,
        model_output=quantized_onnx_path,
        calibration_data_reader=calibrator
    )
    print(f"    [+] Saved quantized model: {quantized_onnx_path}\n")
    
    # Cleanup base ONNX if you only want the quantized version
    os.remove(onnx_path) 

# ==========================================
# 3. TASK ROUTER
# ==========================================
def process_task(task: str):
    prep_cfg = PreprocessConfig()
    
    if task in ["arousal", "both"]:
        print("========================================")
        print("[*] Starting Arousal Model Pipeline")
        print("========================================")
        train_cfg = TrainArousalsConfig()
        
        # Setup dummy inputs based on PreprocessConfig
        shapes = prep_cfg.arousal_tensor_shape
        dummy_input = tuple(torch.randn(shape) for shape in shapes)
        
        export_and_quantize_single(
            model=ArousalNet(),
            model_path=train_cfg.out_path,
            dataset_path=train_cfg.train_path,  # Using train split for calibration
            dummy_input=dummy_input,
            input_names=['temporal_input', 'context_input'],
            output_names=['arousal_event_logits'],
            dataset_keys=['eeg_windows', 'context_windows']
        )
        
    if task in ["sleep", "both"]:
        print("========================================")
        print("[*] Starting Sleep Stage Model Pipeline")
        print("========================================")
        train_cfg = TrainSleepStagesConfig()
        
        # Setup dummy input based on PreprocessConfig
        dummy_input = torch.randn(prep_cfg.sleep_tensor_shape)
        
        export_and_quantize_single(
            model=SleepStageNet(), # Assuming this is your network class name
            model_path=train_cfg.out_path,
            dataset_path=train_cfg.train_path, # Using train split for calibration
            dummy_input=dummy_input,
            input_names=['sleep_features'],
            output_names=['sleep_stage_logits'],
            dataset_keys=['sleep_features']
        )

# ==========================================
# 4. CLI ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and quantize Arousal and Sleep Stage models.")
    parser.add_argument(
        "--task", 
        choices=["arousal", "sleep", "both"], 
        default="both", 
        help="Choose which model to export/quantize (default: both)"
    )
    
    args = parser.parse_args()
    process_task(args.task)