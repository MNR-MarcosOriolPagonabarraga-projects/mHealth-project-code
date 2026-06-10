const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // 1. CHANGE TO STATIC LIBRARY: Compiles to an standalone archive library (.a)
    const lib = b.addStaticLibrary(.{
        .name = "sleep_stager_rt",
        .root_source_file = .{ .path = "src/main_rt.zig" },
        .target = target,
        .optimize = optimize,
    });

    const c_flags = &[_][]const u8{
        "-std=c99",
        "-O3",
        "-Wall",
        "-ffreestanding", // No host OS assumptions
        "-nostdinc", // Avoid pulling host headers
    };

    lib.addCSourceFile(.{
        .file = .{ .path = "src/models/sleep-phase/sleep_phase_model.c" },
        .flags = c_flags,
    });

    lib.addCSourceFile(.{
        .file = .{ .path = "src/models/arousals/arousal_detector.c" },
        .flags = c_flags,
    });

    lib.addCSourceFile(.{
        .file = .{ .path = "src/models/math_stubs.c" },
        .flags = c_flags,
    });

    lib.addIncludePath(.{ .path = "src" });

    b.installArtifact(lib);
}
