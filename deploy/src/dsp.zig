const std = @import("std");

/// A single Biquad (Second Order Section) filter stage
pub const Biquad = struct {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,

    // Internal state memory
    x1: f32 = 0,
    x2: f32 = 0,
    y1: f32 = 0,
    y2: f32 = 0,

    pub fn process(self: *Biquad, input: f32) f32 {
        const out = (self.b0 * input) + (self.b1 * self.x1) + (self.b2 * self.x2) - (self.a1 * self.y1) - (self.a2 * self.y2);

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = out;

        return out;
    }
};

/// Cascades multiple Biquad stages together (e.g., your 4-stage bandpass)
pub fn Cascade(comptime stages_count: usize) type {
    return struct {
        stages: [stages_count]Biquad,

        // Change *Self to *Cascade(stages_count)
        pub fn process(self: *Cascade(stages_count), input: f32) f32 {
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

    pub fn push(self: *Downsampler, sample: f32) ?f32 {
        self.counter += 1;
        if (self.counter == self.factor) {
            self.counter = 0;
            return sample;
        }
        return null;
    }
};
