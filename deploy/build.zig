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
    // This provides the core type mappings like int8_t and basic math symbols
    exe.linkLibC();

    // Add generated onnx2c file directly into the build pipeline
    // This compiles sleep_phase_model.c directly into the final executable machine code
    exe.addCSourceFile(.{
        .file = .{ .path = "src/sleep_phase_model.c" },
        .flags = &[_][]const u8{
            "-std=c99",
            "-O3", // Aggressive math and loop optimizations for inference speed
            "-Wall",
        },
    });
    exe.addIncludePath(.{ .path = "src" });

    // Install the compiled binary artifact into the standard zig-out/bin/ path
    b.installArtifact(exe);

    // Create a 'run' step so you can test your code instantly in your development environment
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
