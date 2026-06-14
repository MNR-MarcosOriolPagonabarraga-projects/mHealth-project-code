const std = @import("std");
const dsp = @import("dsp.zig");

pub const PipelineBuffers = struct {
    // Shared Temporal Data (100Hz)
    raw_history: [2][200]f32 = [_][200]f32{[_]f32{0} ** 200} ** 2,
    raw_idx: usize = 0,

    // Arousal Buffers
    arousal_temporal: [1][2][1500]f32 = undefined, // 15 seconds
    arousal_temporal_idx: usize = 0,
    arousal_context: [1][10][115]f32 = undefined, // ~1 min
    arousal_ctx_idx: usize = 0,

    // Sleep Buffers
    sleep_context: [1][60][20]f32 = undefined, // 60 seconds (2x 30s)
    sleep_ctx_idx: usize = 0,

    stft_counter: usize = 0,

    pub fn push_sample(self: *@This(), ch0: f32, ch1: f32) void {
        // Keep a rolling 200-sample buffer for STFTs
        self.raw_history[0][self.raw_idx] = ch0;
        self.raw_history[1][self.raw_idx] = ch1;

        // Push to Arousal Temporal (15s = 1500 samples)
        self.arousal_temporal[0][0][self.arousal_temporal_idx] = ch0;
        self.arousal_temporal[0][1][self.arousal_temporal_idx] = ch1;
        self.arousal_temporal_idx = (self.arousal_temporal_idx + 1) % 1500;

        self.raw_idx = (self.raw_idx + 1) % 200;
        self.stft_counter += 1;

        // Context Step (Every 50 samples = 0.5 seconds)
        if (self.stft_counter >= 50) {
            self.stft_counter = 0;
            self.compute_and_push_context(true);
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
            // Arousal uses shape [1][10][115]
            for (0..5) |b| {
                self.arousal_context[0][b][self.arousal_ctx_idx] = bands0[b];
                self.arousal_context[0][b + 5][self.arousal_ctx_idx] = bands1[b];
            }
            self.arousal_ctx_idx = (self.arousal_ctx_idx + 1) % 115;
        } else {
            // Sleep uses [1][60][20] -> Flattened internally as 120 steps of 10 features
            const flat_idx = self.sleep_ctx_idx;
            const t = flat_idx % 60;
            const feature_offset = (flat_idx / 60) * 10;

            for (0..5) |b| {
                self.sleep_context[0][t][feature_offset + b] = bands0[b];
                self.sleep_context[0][t][feature_offset + 5 + b] = bands1[b];
            }
            self.sleep_ctx_idx = (self.sleep_ctx_idx + 1) % 120;
        }
    }

    pub fn prep_tensors_for_inference(self: *@This()) void {
        // --- 1. AROUSAL TEMPORAL ---
        var temp = self.arousal_temporal;
        for (0..1500) |i| {
            const src_idx = (self.arousal_temporal_idx + i) % 1500;
            self.arousal_temporal[0][0][i] = temp[0][0][src_idx];
            self.arousal_temporal[0][1][i] = temp[0][1][src_idx];
        }
        dsp.zscore_normalize(&self.arousal_temporal[0][0]);
        dsp.zscore_normalize(&self.arousal_temporal[0][1]);

        // --- 2. AROUSAL CONTEXT ---
        var ctx_temp = self.arousal_context;
        for (0..10) |c| {
            for (0..115) |i| {
                const src_idx = (self.arousal_ctx_idx + i) % 115;
                self.arousal_context[0][c][i] = ctx_temp[0][c][src_idx];
            }
            dsp.zscore_normalize(&self.arousal_context[0][c]);
        }

        // --- 3. SLEEP CONTEXT ---
        var sleep_temp = self.sleep_context;

        // Unroll the 120 steps into chronological order
        for (0..120) |i| {
            const src_idx = (self.sleep_ctx_idx + i) % 120;
            const src_t = src_idx % 60;
            const src_feat_offset = (src_idx / 60) * 10;

            const dest_t = i % 60;
            const dest_feat_offset = (i / 60) * 10;

            for (0..10) |f| {
                self.sleep_context[0][dest_t][dest_feat_offset + f] = sleep_temp[0][src_t][src_feat_offset + f];
            }
        }

        // Apply Z-Score Normalization independently per feature channel (20 features total)
        for (0..20) |f| {
            var feature_slice = [_]f32{0} ** 60;

            // Extract the time-series for a single feature
            for (0..60) |t| {
                feature_slice[t] = self.sleep_context[0][t][f];
            }

            // Normalize it
            dsp.zscore_normalize(&feature_slice);

            // Place it back into the main buffer
            for (0..60) |t| {
                self.sleep_context[0][t][f] = feature_slice[t];
            }
        }
    }
};
