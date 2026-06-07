const std = @import("std");
const fft = @import("fft.zig");

pub const FeaturePipeline = struct {
    // 3 windows * 120 steps = 360 total time steps.
    // Each step holds 10 spectral features.
    ring_buffer: [360][10]f32 = undefined,
    head: usize = 0,
    is_full: bool = false,

    /// Pushes a new 10-feature STFT vector into the circular buffer
    pub fn push_features(self: *FeaturePipeline, features: [10]f32) void {
        self.ring_buffer[self.head] = features;
        self.head += 1;

        if (self.head >= 360) {
            self.head = 0;
            self.is_full = true;
        }
    }

    /// Assembles the [120][30] tensor from the rolling past/curr/fut windows
    /// Returns false if we don't have enough data (90 seconds) yet.
    pub fn build_inference_tensor(self: *const FeaturePipeline, out_tensor: *[120][30]f32) bool {
        if (!self.is_full) return false;

        var read_idx = self.head; // The oldest data is at the current head

        for (0..120) |t| {
            // Read Past Window (Features 0-9)
            const past_idx = (read_idx + t) % 360;
            @memcpy(out_tensor[t][0..10], &self.ring_buffer[past_idx]);

            // Read Current Window (Features 10-19)
            const curr_idx = (read_idx + t + 120) % 360;
            @memcpy(out_tensor[t][10..20], &self.ring_buffer[curr_idx]);

            // Read Future Window (Features 20-29)
            const fut_idx = (read_idx + t + 240) % 360;
            @memcpy(out_tensor[t][20..30], &self.ring_buffer[fut_idx]);
        }
        return true;
    }

    /// Placeholder for your actual STFT computation block
    pub fn compute_stft_bandpower(hop_buffer_200: []const f32) [10]f32 {
        var reals = [_]f32{0.0} ** 256;
        var imags = [_]f32{0.0} ** 256;

        // Copy the 200 samples into the 256-length array (auto zero-pads the last 56)
        // Optionally, apply a Hanning window here!
        for (hop_buffer_200, 0..) |val, i| {
            reals[i] = val;
        }

        // Compute the FFT in-place
        fft.compute_fft(&reals, &imags);

        // Compute Power Spectrum (Magnitude Squared)
        var power_spectrum = [_]f32{0.0} ** 128; // Only need the first half (Nyquist limit)
        for (0..128) |i| {
            power_spectrum[i] = (reals[i] * reals[i]) + (imags[i] * imags[i]);
        }

        // Bin into your 10 frequency bands
        var features = [_]f32{0.0} ** 10;
        const bins_per_feature = 128 / 10; // Simple integer division for example

        for (0..10) |f_idx| {
            var sum: f32 = 0.0;
            const start_bin = f_idx * bins_per_feature;
            const end_bin = start_bin + bins_per_feature;

            for (start_bin..end_bin) |b| {
                sum += power_spectrum[b];
            }
            // 5. Apply Z-score norm here (omitted for brevity)
            features[f_idx] = sum;
        }

        return features;
    }
};
