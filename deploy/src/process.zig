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
    sleep_context: [1][20][60]f32 = undefined, // 20 features x 60 steps (100Hz, 50-sample hops)
    sleep_ctx_idx: usize = 0,

    stft_counter: usize = 0,

    pub fn push_sample(self: *@This(), ch0: f32, ch1: f32) void {
        self.raw_history[0][self.raw_idx] = ch0;
        self.raw_history[1][self.raw_idx] = ch1;

        self.arousal_temporal[0][0][self.arousal_temporal_idx] = ch0;
        self.arousal_temporal[0][1][self.arousal_temporal_idx] = ch1;
        self.arousal_temporal_idx = (self.arousal_temporal_idx + 1) % 1500;

        self.raw_idx = (self.raw_idx + 1) % 200;
        self.stft_counter += 1;

        if (self.stft_counter >= 50) {
            self.stft_counter = 0;
            self.compute_and_push_context(true);
            self.compute_and_push_context(false);
        }
    }

    fn compute_and_push_context(self: *@This(), is_arousal: bool) void {
        var ordered_ch0 = [_]f32{0} ** 200;
        var ordered_ch1 = [_]f32{0} ** 200;

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
            for (0..5) |b| {
                self.arousal_context[0][b][self.arousal_ctx_idx] = bands0[b];
                self.arousal_context[0][b + 5][self.arousal_ctx_idx] = bands1[b];
            }
            self.arousal_ctx_idx = (self.arousal_ctx_idx + 1) % 115;
        } else {
            // Pack incoming features safely into the circular buffer layout
            const t = self.sleep_ctx_idx % 60;
            const feature_offset = (self.sleep_ctx_idx / 60) * 10;

            for (0..5) |b| {
                self.sleep_context[0][feature_offset + b][t] = bands0[b];
                self.sleep_context[0][feature_offset + 5 + b][t] = bands1[b];
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

        // --- 3. SLEEP CONTEXT CHRONOLOGICAL UNROLLING ---
        var sleep_temp = self.sleep_context;

        for (0..60) |i| {
            // Resolve the circular indices for the Oldest step (Past) and Newest step (Current)
            const past_idx = (self.sleep_ctx_idx + i) % 120;
            const curr_idx = (self.sleep_ctx_idx + 60 + i) % 120;

            // Map physical locations in the buffer
            const past_row_base = (past_idx / 60) * 10;
            const past_col = past_idx % 60;

            const curr_row_base = (curr_idx / 60) * 10;
            const curr_col = curr_idx % 60;

            // Pack cleanly into the top and bottom halves
            for (0..10) |sub| {
                self.sleep_context[0][sub][i] = sleep_temp[0][past_row_base + sub][past_col];
                self.sleep_context[0][10 + sub][i] = sleep_temp[0][curr_row_base + sub][curr_col];
            }
        }

        // Apply Z-Score Normalization
        for (0..20) |f| {
            var feature_slice = [_]f32{0} ** 60;

            // Extract single feature timeline
            for (0..60) |t| {
                feature_slice[t] = self.sleep_context[0][f][t];
            }

            dsp.zscore_normalize(&feature_slice);

            // Repack normalized
            for (0..60) |t| {
                self.sleep_context[0][f][t] = feature_slice[t];
            }
        }
    }
};
