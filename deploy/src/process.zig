const std = @import("std");
const dsp = @import("dsp.zig");

pub const PipelineBuffers = struct {
    // Shared Temporal Data (100Hz)
    raw_history: [2][200]f32 = [_][200]f32{[_]f32{0} ** 200} ** 2,
    raw_idx: usize = 0,

    // Arousal Buffers
    arousal_temporal: [1][2][1500]f32 = undefined, // 15 seconds
    arousal_temporal_idx: usize = 0,
    arousal_context: [1][10][149]f32 = undefined, // 5 mins
    arousal_ctx_idx: usize = 0,
    arousal_stft_counter: usize = 0,

    // Sleep Buffers
    sleep_context: [1][120][30]f32 = undefined, // 90 seconds (3x 30s)
    sleep_ctx_idx: usize = 0,
    sleep_stft_counter: usize = 0,

    pub fn push_sample(self: *@This(), ch0: f32, ch1: f32) void {
        // Keep a rolling 200-sample buffer for STFTs
        self.raw_history[0][self.raw_idx] = ch0;
        self.raw_history[1][self.raw_idx] = ch1;

        // Push to Arousal Temporal (15s = 1500 samples)
        self.arousal_temporal[0][0][self.arousal_temporal_idx] = ch0;
        self.arousal_temporal[0][1][self.arousal_temporal_idx] = ch1;
        self.arousal_temporal_idx = (self.arousal_temporal_idx + 1) % 1500;

        self.raw_idx = (self.raw_idx + 1) % 200;
        self.arousal_stft_counter += 1;
        self.sleep_stft_counter += 1;

        // Arousal Context Step (Every 200 samples = 2 seconds)
        if (self.arousal_stft_counter >= 200) {
            self.arousal_stft_counter = 0;
            self.compute_and_push_context(true);
        }

        // Sleep Context Step (Every 25 samples = 0.25 seconds)
        if (self.sleep_stft_counter >= 25) {
            self.sleep_stft_counter = 0;
            self.compute_and_push_context(false);
        }
    }

    fn compute_and_push_context(self: *@This(), is_arousal: bool) void {
        var ordered_ch0 = [_]f32{0} ** 200;
        var ordered_ch1 = [_]f32{0} ** 200;

        // Unroll the circular history buffer
        for (0..200) |i| {
            const idx = (self.raw_idx + i) % 200;
            ordered_ch0[i] = self.raw_history[0][idx];
            ordered_ch1[i] = self.raw_history[1][idx];
        }

        var bands0 = [_]f32{0} ** 5;
        var bands1 = [_]f32{0} ** 5;
        dsp.compute_bandpowers(&ordered_ch0, &bands0);
        dsp.compute_bandpowers(&ordered_ch1, &bands1);

        if (is_arousal) {
            // Arousal uses shape [1][10][149]
            for (0..5) |b| {
                self.arousal_context[0][b][self.arousal_ctx_idx] = bands0[b];
                self.arousal_context[0][b + 5][self.arousal_ctx_idx] = bands1[b];
            }
            self.arousal_ctx_idx = (self.arousal_ctx_idx + 1) % 149;
        } else {
            // Sleep uses [1][120][30] -> Flattened internally as 360 steps of 10 features
            const flat_idx = self.sleep_ctx_idx;
            const t = flat_idx % 120;
            const feature_offset = (flat_idx / 120) * 10;

            for (0..5) |b| {
                self.sleep_context[0][t][feature_offset + b] = bands0[b];
                self.sleep_context[0][t][feature_offset + 5 + b] = bands1[b];
            }
            self.sleep_ctx_idx = (self.sleep_ctx_idx + 1) % 360;
        }
    }

    pub fn prep_tensors_for_inference(self: *@This()) void {
        // Roll Arousal Temporal so oldest is at index 0
        var temp = self.arousal_temporal;
        for (0..1500) |i| {
            const src_idx = (self.arousal_temporal_idx + i) % 1500;
            self.arousal_temporal[0][0][i] = temp[0][0][src_idx];
            self.arousal_temporal[0][1][i] = temp[0][1][src_idx];
        }
        dsp.zscore_normalize(&self.arousal_temporal[0][0]);
        dsp.zscore_normalize(&self.arousal_temporal[0][1]);

        // Roll Arousal Context so oldest is at index 0
        var ctx_temp = self.arousal_context;
        for (0..10) |c| {
            for (0..149) |i| {
                const src_idx = (self.arousal_ctx_idx + i) % 149;
                self.arousal_context[0][c][i] = ctx_temp[0][c][src_idx];
            }
            dsp.zscore_normalize(&self.arousal_context[0][c]);
        }

        // NOTE: Sleep context [120][30] is naturally handled by rolling memory,
        // but normally requires similar shifting based on `sleep_ctx_idx` before Z-scoring.
    }
};
