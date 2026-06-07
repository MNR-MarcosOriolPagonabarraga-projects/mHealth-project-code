const std = @import("std");

/// In-place Radix-2 Fast Fourier Transform
/// The input array length must be a power of 2 (e.g., 256).
pub fn compute_fft(reals: []f32, imags: []f32) void {
    const n = reals.len;
    std.debug.assert((n & (n - 1)) == 0); // Ensure power of 2

    // Bit-reversal permutation
    var j: usize = 0;
    for (0..n - 1) |i| {
        if (i < j) {
            std.mem.swap(f32, &reals[i], &reals[j]);
            std.mem.swap(f32, &imags[i], &imags[j]);
        }
        var m: usize = n >> 1;
        while (m >= 1 and j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Cooley-Tukey Decimation-in-Time
    var step: usize = 1;
    while (step < n) {
        const step_f32: f32 = @floatFromInt(step);
        const theta = -std.math.pi / step_f32;
        const w_p_re = @cos(theta);
        _ = w_p_re;
        const w_p_im = @sin(theta);
        _ = w_p_im;

        var m: usize = 0;
        while (m < step) : (m += 1) {
            var w_re: f32 = 1.0;
            var w_im: f32 = 0.0;

            // Advance twiddle factors
            if (m > 0) {
                const angle = @as(f32, @floatFromInt(m)) * theta;
                w_re = @cos(angle);
                w_im = @sin(angle);
            }

            var i: usize = m;
            while (i < n) : (i += step * 2) {
                const k = i + step;
                // Complex multiplication: (w_re + j*w_im) * (reals[k] + j*imags[k])
                const t_re = w_re * reals[k] - w_im * imags[k];
                const t_im = w_re * imags[k] + w_im * reals[k];

                reals[k] = reals[i] - t_re;
                imags[k] = imags[i] - t_im;
                reals[i] += t_re;
                imags[i] += t_im;
            }
        }
        step *= 2;
    }
}
