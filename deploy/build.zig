const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "sleep_stager",
        .root_source_file = .{ .path = "src/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Force Zig to link against the standard C library
    exe.linkLibC();

    // Shared C compiler flags for maximum inference speed
    const c_flags = &[_][]const u8{
        "-std=c99",
        "-O3", // Aggressive math and loop optimizations
        "-Wall",
    };

    // Sleep Phase Model
    exe.addCSourceFile(.{
        .file = .{ .path = "models/sleep_stage/sleep_stage_detector.c" },
        .flags = c_flags,
    });

    // AROUSAL DETECTOR MODEL
    exe.addCSourceFile(.{
        .file = .{ .path = "models/arousals/arousal_detector.c" },
        .flags = c_flags,
    });

    exe.addIncludePath(.{ .path = "src" });

    // Install the compiled binary artifact into the standard zig-out/bin/ path
    b.installArtifact(exe);

    // Create a 'run' step
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
