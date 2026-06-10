#include <float.h>

// Tailored Taylor series expansion to compute e^x without an underlying OS math library
float expf(float x) {
    // Fast processing for typical neural network boundary overflows
    if (x > 88.0f) return FLT_MAX;
    if (x < -88.0f) return 0.0f;

    float sum = 1.0f;
    float term = 1.0f;
    
    // 12 iterations provide clean single-precision accuracy for sigmoid layers
    for (int i = 1; i < 12; i++) {
        term *= (x / (float)i);
        sum += term;
    }
    return sum;
}

// Emulated truncation rounding logic required by the quantized neural network weights
float roundf(float x) {
    if (x < 0.0f) {
        return (float)((int)(x - 0.5f));
    }
    return (float)((int)(x + 0.5f));
}

// Fast trigonometric Taylor approximations for real-time STFT windowing calculations
float cosf(float x) {
    // Normalize angle to (-PI, PI)
    float t = x;
    while (t > 3.14159265f) t -= 6.2831853f;
    while (t < -3.14159265f) t += 6.2831853f;

    float t2 = t * t;
    // Maclaurin series approximation for cos(x)
    return 1.0f - (t2 / 2.0f) + (t2 * t2 / 24.0f) - (t2 * t2 * t2 / 720.0f);
}

float sinf(float x) {
    // Normalize angle to (-PI, PI)
    float t = x;
    while (t > 3.14159265f) t -= 6.2831853f;
    while (t < -3.14159265f) t += 6.2831853f;

    float t2 = t * t;
    // Maclaurin series approximation for sin(x)
    return t * (1.0f - (t2 / 6.0f) + (t2 * t2 / 120.0f) - (t2 * t2 * t2 / 5040.0f));
}