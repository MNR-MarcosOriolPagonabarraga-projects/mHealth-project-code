const std = @import("std");
const dsp = @import("dsp.zig");
const process = @import("process.zig");

const SleepModel = @cImport({
    @cInclude("models/sleep-phase/sleep_stage_detector.h");
});

const ArousalModel = @cImport({
    @cInclude("models/arousals/arousal_detector.h");
});

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    // Skip argv[0]
    _ = args.skip();

    // Extract the input binary path argument (argv[1])
    const bin_path = args.next() orelse {
        const stderr = std.io.getStdErr().writer();
        try stderr.print("Error: Missing input binary file path.\n", .{});
        try stderr.print("Usage: sleep_stager <path/to/recording.bin>\n", .{});
        std.process.exit(1);
    };

    try process_recording(bin_path, "sleep_preds.csv", "arousal_preds.csv");
}

pub fn process_recording(bin_path: []const u8, sleep_csv_path: []const u8, arousal_csv_path: []const u8) !void {
    var bin_file = try std.fs.cwd().openFile(bin_path, .{});
    defer bin_file.close();

    var sleep_csv = try std.fs.cwd().createFile(sleep_csv_path, .{});
    defer sleep_csv.close();
    var sleep_writer = sleep_csv.writer();
    try sleep_writer.print("timestamp_s,wake,light_sleep,deep_sleep,rem\n", .{});

    var arousal_csv = try std.fs.cwd().createFile(arousal_csv_path, .{});
    defer arousal_csv.close();
    var arousal_writer = arousal_csv.writer();
    try arousal_writer.print("timestamp_s,arousal_logit\n", .{});

    var notch = [_]dsp.Biquad{ dsp.get_eeg_notch(), dsp.get_eeg_notch() };
    var bp = [_]dsp.Cascade(4){ dsp.get_eeg_bandpass(), dsp.get_eeg_bandpass() };
    var decimeter = [_]dsp.Downsampler{ .{ .factor = 2 }, .{ .factor = 2 } };

    var buffers = process.PipelineBuffers{};
    var sample_count_100hz: usize = 0;
    var raw_bytes: [8]u8 = undefined;

    std.debug.print("Starting Pipeline processing on: {s}...\n", .{bin_path});

    while (try bin_file.readAll(&raw_bytes) == 8) {
        const raw_ch0 = @as(f32, @bitCast(raw_bytes[0..4].*));
        const raw_ch1 = @as(f32, @bitCast(raw_bytes[4..8].*));

        const filt_ch0 = bp[0].process(notch[0].process(raw_ch0));
        const filt_ch1 = bp[1].process(notch[1].process(raw_ch1));

        const ds_ch0 = decimeter[0].push(filt_ch0);
        const ds_ch1 = decimeter[1].push(filt_ch1);

        if (ds_ch0 != null and ds_ch1 != null) {
            buffers.push_sample(ds_ch0.?, ds_ch1.?);
            sample_count_100hz += 1;

            // Trigger predictions every 5 seconds (500 samples)
            if (sample_count_100hz % 500 == 0) {
                const current_time_s = @as(f32, @floatFromInt(sample_count_100hz)) / 100.0;
                buffers.prep_tensors_for_inference();

                // 1. Sleep Phase Prediction
                var sleep_logits = [_]f32{0} ** 4;
                SleepModel.sleep_phase_entry(@as([*c]const [60][20]f32, @ptrCast(&buffers.sleep_context)), &sleep_logits);
                try sleep_writer.print("{d:.1},{d:.4},{d:.4},{d:.4},{d:.4}\n", .{ current_time_s, sleep_logits[0], sleep_logits[1], sleep_logits[2], sleep_logits[3] });

                // 2. Arousal Prediction
                var arousal_logits = [_]f32{0} ** 1;
                ArousalModel.arousal_detector_entry(&buffers.arousal_temporal, &buffers.arousal_context, &arousal_logits);
                try arousal_writer.print("{d:.1},{d:.4}\n", .{ current_time_s, arousal_logits[0] });
            }
        }
    }
    std.debug.print("Pipeline finished successfully. Predictions written to CSV files.\n", .{});
}
