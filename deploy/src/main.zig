// src/main.zig
const std = @import("std");
const dsp = @import("dsp.zig");

pub fn main() !void {
    // 1. Initialize your 60 Hz Notch Filter (fs=200)
    // Python arrays: b_notch = [0.9695, 0.5992, 0.9695], a_notch = [1.0, 0.5992, 0.9390]
    var notch_filter = dsp.Biquad{
        .b0 = 0.96953125,
        .b1 = 0.59920327,
        .b2 = 0.96953125,
        .a1 = 0.59920327,
        .a2 = 0.93906251,
    };

    // 2. Initialize your 4-stage Bandpass Filter (fs=200) using your exact printed matrix
    var bp_cascade = dsp.Cascade(4){ .stages = .{
        .{ .b0 = 0.04476726, .b1 = 0.08953452, .b2 = 0.04476726, .a1 = -0.34836077, .a2 = 0.06920663 },
        .{ .b0 = 1.0, .b1 = 2.0, .b2 = 1.0, .a1 = -0.45955167, .a2 = 0.47395034 },
        .{ .b0 = 1.0, .b1 = -2.0, .b2 = 1.0, .a1 = -1.97070403, .a2 = 0.97095643 },
        .{ .b0 = 1.0, .b1 = -2.0, .b2 = 1.0, .a1 = -1.98798668, .a2 = 0.98823347 },
    } };

    var decimeter = dsp.Downsampler{ .factor = 2 };

    // --- Simulated Real-Time Streaming Loop ---
    // Imagine this incoming data stream is arriving raw from your hardware ADC at 200Hz
    var raw_200hz_sample: f32 = 0.0;

    while (get_sensor_sample(&raw_200hz_sample)) {
        // Step A: Strip out the 60Hz hum at 200Hz
        const notched = notch_filter.process(raw_200hz_sample);

        // Step B: Restrict bandwidth to 0.5 - 40Hz (acts as anti-aliasing filter)
        const filtered = bp_cascade.process(notched);

        // Step C: Safely downsample (200Hz -> 100Hz)
        if (decimeter.push(filtered)) |ds_100hz_sample| {
            // This sample is perfectly clean, downsampled, and ready for your
            // 25-sample hop buffer to build the STFT!
            _ = ds_100hz_sample;
        }
    }
}

fn get_sensor_sample(out: *f32) bool {
    out.* = 0.123;
    return true;
} // Dummy data stream hook
