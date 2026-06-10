#ifndef BAREMETAL_MATH_H
#define BAREMETAL_MATH_H

// Define INFINITY using Clang's built-in float infinity
#define INFINITY (__builtin_inff())

// Function prototypes connecting to our math_stubs.c
float expf(float x);
float roundf(float x);
float cosf(float x);
float sinf(float x);

#endif
