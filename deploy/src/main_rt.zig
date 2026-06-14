const std = @import("std");
const dsp = @import("dsp.zig");
const process = @import("process.zig");

const SleepModel = @cImport({
    @cInclude("models/sleep-phase/sleep_phase_model.h");
});
const ArousalModel = @cImport({
    @cInclude("models/arousals/arousal_detector.h");
});

// Statically allocated processing structures in the .bss microcontroller region
var notch = [_]dsp.Biquad{ dsp.get_eeg_notch(), dsp.get_eeg_notch() };
var bp = [_]dsp.Cascade(4){ dsp.get_eeg_bandpass(), dsp.get_eeg_bandpass() };
var decimeter = [_]dsp.Downsampler{ .{ .factor = 2 }, .{ .factor = 2 } };
var buffers = process.PipelineBuffers{};
var sample_count_100hz: usize = 0;

// Globally exposed arrays. Your micro C application reads these directly to take action
export var global_sleep_logits = [_]f32{0} ** 4;
export var global_arousal_logits = [_]f32{0} ** 1;

/// EXPORTED FOR C: Call this function inside your 200Hz hardware timer or ADC interrupt handler!
export fn pipeline_feed_sample(raw_ch0: f32, raw_ch1: f32) void {
    const filt_ch0 = bp[0].process(notch[0].process(raw_ch0));
    const filt_ch1 = bp[1].process(notch[1].process(raw_ch1));

    const ds_ch0 = decimeter[0].push(filt_ch0);
    const ds_ch1 = decimeter[1].push(filt_ch1);

    if (ds_ch0 != null and ds_ch1 != null) {
        buffers.push_sample(ds_ch0.?, ds_ch1.?);
        sample_count_100hz += 1;

        // Run inference every 5 seconds (500 samples at 100Hz)
        if (sample_count_100hz % 500 == 0) {
            buffers.prep_tensors_for_inference();

            // Fire model evaluations locally inside the hardware runtime memory layout
            SleepModel.sleep_stage_entry(&buffers.sleep_context, &global_sleep_logits);
            ArousalModel.arousal_detector_entry(&buffers.arousal_temporal, &buffers.arousal_context, &global_arousal_logits);
        }
    }
}
