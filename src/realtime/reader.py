import numpy as np

class SampleProvider:
    def __init__(self, bin_file):
        self.data_content = np.fromfile(bin_file)
        self.counter = 0

    def read(self) -> np.float32:
        if self.counter % 2 == 0:
            return 