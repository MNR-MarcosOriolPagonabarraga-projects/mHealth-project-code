const std = @import("std");
const dsp = @import("dsp.zig");
const process = @import("process.zig");
const SleepModel = @cImport("models/sleep-phase/sleep_phase_model.h");
const ArousalModel = @cImport("models/arousals/arousal_detector.h");

pub fn process_recording(bin_path: []const u8, sleep_csv_path: []const u8, arousal_csv_path: []const u8) !void {
    // 1. Initialize files
    var bin_file = try std.fs.cwd().openFile(bin_path, .{});
    defer bin_file.close();

    var sleep_csv = try std.fs.cwd().createFile(sleep_csv_path, .{});
    defer sleep_csv.close();

    var arousal_csv = try std.fs.cwd().createFile(arousal_csv_path, .{});
    defer arousal_csv.close();

    var sleep_writer = sleep_csv.writer();
    var arousal_writer = arousal_csv.writer();
    try sleep_writer.print("timestamp_s,logit_0,logit_1,logit_2,logit_3\n", .{});
    try arousal_writer.print("timestamp_s,arousal_logit\n", .{});

    // 2. Initialize DSP & Buffers
    var notch = dsp.get_eeg_notch();
    var bp = dsp.get_eeg_bandpass();
    var decimeter = dsp.Downsampler{ .factor = 2 };

    // TODO: Arousal temporal buffer [2][1500]
    // TODO: Context feature buffer [149][10] (Arousal) & [360][10] (Sleep)

    var sample_count_100hz: usize = 0;
    var raw_bytes: [4]u8 = undefined; // Assuming f32 chunks

    // 3. Main processing loop
    while (try bin_file.readAll(&raw_bytes) == 4) {
        const raw_sample = @as(f32, @bitCast(raw_bytes)); // Awaiting your channel formatting clarification

        const filtered = bp.process(notch.process(raw_sample));

        if (decimeter.push(filtered)) |ds_100hz_sample| {
            _ = ds_100hz_sample;
            // Push to Temporal buffer
            // Push to STFT buffer -> if ready, compute and push to Context buffers

            sample_count_100hz += 1;

            // Trigger predictions every 5 seconds (500 samples at 100Hz)
            if (sample_count_100hz % 500 == 0) {
                const current_time_s = @as(f32, @floatFromInt(sample_count_100hz)) / 100.0;
                _ = current_time_s;

                // Sleep Phase Prediction
                // var sleep_logits = [_]f32{0} ** 4;
                // SleepModel.entry(&sleep_input_tensor, &sleep_logits);
                // try sleep_writer.print("{d:.2},{d:.4},{d:.4},{d:.4},{d:.4}\n", .{current_time_s, sleep_logits[0]...});

                // Arousal Prediction
                // var arousal_logits = [_]f32{0} ** 1;
                // ArousalModel.entry(&arousal_temporal_tensor, &arousal_context_tensor, &arousal_logits);
                // try arousal_writer.print("{d:.2},{d:.4}\n", .{current_time_s, arousal_logits[0]});
            }
        }
    }
}
