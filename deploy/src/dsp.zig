const std = @import("std");
const fft = @import("fft.zig");

pub const Biquad = struct {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32 = 0,
    x2: f32 = 0,
    y1: f32 = 0,
    y2: f32 = 0,

    pub fn process(self: *@This(), input: f32) f32 {
        const out = (self.b0 * input) + (self.b1 * self.x1) + (self.b2 * self.x2) - (self.a1 * self.y1) - (self.a2 * self.y2);
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = out;
        return out;
    }
};

pub fn Cascade(comptime stages_count: usize) type {
    return struct {
        stages: [stages_count]Biquad,
        pub fn process(self: *@This(), input: f32) f32 {
            var out = input;
            for (&self.stages) |*stage| {
                out = stage.process(out);
            }
            return out;
        }
    };
}

pub const Downsampler = struct {
    factor: usize,
    counter: usize = 0,
    pub fn push(self: *@This(), sample: f32) ?f32 {
        self.counter += 1;
        if (self.counter == self.factor) {
            self.counter = 0;
            return sample;
        }
        return null;
    }
};

/// 50/60Hz Notch
pub fn get_eeg_notch() Biquad {
    return .{ .b0 = 0.96953125, .b1 = 0.59920327, .b2 = 0.96953125, .a1 = 0.59920327, .a2 = 0.93906251 };
}

/// 4-stage Bandpass
pub fn get_eeg_bandpass() Cascade(4) {
    return .{ .stages = .{
        .{ .b0 = 0.04476726, .b1 = 0.08953452, .b2 = 0.04476726, .a1 = -0.34836077, .a2 = 0.06920663 },
        .{ .b0 = 1.0, .b1 = 2.0, .b2 = 1.0, .a1 = -0.45955167, .a2 = 0.47395034 },
        .{ .b0 = 1.0, .b1 = -2.0, .b2 = 1.0, .a1 = -1.97070403, .a2 = 0.97095643 },
        .{ .b0 = 1.0, .b1 = -2.0, .b2 = 1.0, .a1 = -1.98798668, .a2 = 0.98823347 },
    } };
}

/// Computes 5 standard EEG bandpowers for a 256-padded STFT window
pub fn compute_bandpowers(signal_win_200: []const f32, out_bands: *[5]f32) void {
    var reals = [_]f32{0.0} ** 256;
    var imags = [_]f32{0.0} ** 256;

    // Apply Hann window and pad
    for (signal_win_200, 0..) |val, i| {
        // Simple Hann window approx
        const hann = 0.5 * (1.0 - @cos(2.0 * std.math.pi * @as(f32, @floatFromInt(i)) / 199.0));
        reals[i] = val * hann;
    }

    fft.compute_fft(&reals, &imags);

    var power_spec = [_]f32{0.0} ** 128;
    for (0..128) |i| {
        power_spec[i] = (reals[i] * reals[i]) + (imags[i] * imags[i]);
    }

    // Bins based on 100Hz fs / 256 n_fft = 0.3906 Hz/bin
    const band_ranges = [_][2]usize{
        .{ 1, 10 }, // Delta (0.5 - 4.0 Hz)
        .{ 10, 20 }, // Theta (4.0 - 8.0 Hz)
        .{ 20, 31 }, // Alpha (8.0 - 12.0 Hz)
        .{ 31, 41 }, // Sigma (12.0 - 16.0 Hz)
        .{ 41, 77 }, // Beta  (16.0 - 30.0 Hz)
    };

    for (band_ranges, 0..) |rng, b_idx| {
        var sum: f32 = 0.0;
        for (rng[0]..rng[1]) |i| {
            sum += power_spec[i];
        }
        out_bands[b_idx] = @log(1.0 + sum);
    }
}

pub fn zscore_normalize(data: []f32) void {
    var sum: f32 = 0;
    for (data) |val| sum += val;
    const mean = sum / @as(f32, @floatFromInt(data.len));

    var sq_sum: f32 = 0;
    for (data) |val| sq_sum += (val - mean) * (val - mean);
    var std_dev = @sqrt(sq_sum / @as(f32, @floatFromInt(data.len)));

    if (std_dev < 1e-8) std_dev = 1e-8;

    for (data) |*val| val.* = (val.* - mean) / std_dev;
}
