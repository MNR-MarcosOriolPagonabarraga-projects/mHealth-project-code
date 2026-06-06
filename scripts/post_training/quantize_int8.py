import os
import sys
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader

# 1. Create a Calibration Data Reader
class SleepDataCalibrator(CalibrationDataReader):
    def __init__(self, data_path):
        # Load a small batch of real data (e.g., 200 samples) to calibrate ranges
        dataset = np.load(data_path)
        self.data = dataset['spectral_band_windows'][:200].astype(np.float32)
        self.enum_data = iter([{"input_features": np.expand_dims(x, axis=0)} for x in self.data])

    def get_next(self):
        return next(self.enum_data, None)


def main(model_path, dataset_path):
    model_output_path = os.path.join(
        os.path.dirname(model_path), 
        f"{os.path.basename(model_path).split('.')[0]}_int8.onnx"
    )
    quantize_static(
        model_input=model_path,
        model_output=model_output_path,
        calibration_data_reader=SleepDataCalibrator(dataset_path)
    )
    print("[+] Generated sleep_model_int8.onnx")

if __name__ == "__main__":
    model_path = sys.argv[1]
    dataset_path = sys.argv[2]
    main(model_path, dataset_path)