const std = @import("std");

const model = @cImport({
    @cInclude("sleep_phase_model.h");
});

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("[*] Booting Micro Sleep Stager...\n", .{});

    // Fix: Match the C function's exact 2D array signature
    var input_features: [120][30]f32 = undefined;

    // Fill with sample sensor data from your live DSP pipeline
    for (&input_features) |*row| {
        for (row) |*val| {
            val.* = 0.123;
        }
    }

    // Output logits (Matches your model's classification head)
    var output_logits: [4]f32 = undefined;

    // Execute the embedded C loop
    model.entry(&input_features, &output_logits);

    // Compute argmax to get predicted stage
    var max_val: f32 = output_logits[0];
    var predicted_stage: usize = 0;

    for (output_logits, 0..) |val, i| {
        if (val > max_val) {
            max_val = val;
            predicted_stage = i;
        }
    }

    const stage_names = [_][]const u8{ "Wake", "Light Sleep", "Deep Sleep", "REM" };
    try stdout.print("[+] Inference Complete. Class: {s} (Index: {d})\n", .{ stage_names[predicted_stage], predicted_stage });
}
