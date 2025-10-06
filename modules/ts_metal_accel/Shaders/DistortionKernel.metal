/*
 * ts-metal-accel - GPU-accelerated audio processing with Metal
 * Copyright (c) 2025 Travis Skare
 *
 * SPDX-License-Identifier: MIT
 */

#include <metal_stdlib>
using namespace metal;

/**
 * Parameters for the soft clipper distortion effect.
 */
struct DistortionParams
{
    float drive;      // Input gain (pre-distortion), typically 1.0 - 10.0
    float mix;        // Dry/wet mix, 0.0 = dry, 1.0 = wet
    float outputGain; // Output level compensation
};

/**
 * Soft clipper using tanh for smooth saturation.
 *
 * Processes audio samples in parallel on the GPU.
 * Each thread handles one sample from the input buffer.
 *
 * @param inBuffer   Input audio samples [buffer(0)]
 * @param outBuffer  Output audio samples [buffer(1)]
 * @param params     Distortion parameters [buffer(2)]
 * @param bufferSize Total number of samples in buffer [buffer(3)]
 * @param gid        Thread position in grid (sample index)
 */
kernel void soft_clipper(
    const device float* inBuffer  [[buffer(0)]],
    device float*       outBuffer [[buffer(1)]],
    constant DistortionParams& params [[buffer(2)]],
    constant uint& bufferSize [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    // Bounds check to prevent buffer overruns
    if (gid >= bufferSize)
        return;

    // Read input sample
    float input = inBuffer[gid];

    // Apply input drive
    float driven = input * params.drive;

    // Soft clipping using tanh (smooth saturation curve)
    float clipped = tanh(driven);

    // Apply output gain compensation
    float processed = clipped * params.outputGain;

    // Dry/wet mix
    float output = input * (1.0f - params.mix) + processed * params.mix;

    // Write result
    outBuffer[gid] = output;
}

/**
 * Hard clipper for aggressive distortion.
 *
 * Simple clipping to [-1, 1] range with optional threshold.
 */
kernel void hard_clipper(
    const device float* inBuffer  [[buffer(0)]],
    device float*       outBuffer [[buffer(1)]],
    constant DistortionParams& params [[buffer(2)]],
    constant uint& bufferSize [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    // Bounds check to prevent buffer overruns
    if (gid >= bufferSize)
        return;

    float input = inBuffer[gid];
    float driven = input * params.drive;

    // Hard clip
    float clipped = clamp(driven, -1.0f, 1.0f);

    float processed = clipped * params.outputGain;
    float output = input * (1.0f - params.mix) + processed * params.mix;

    outBuffer[gid] = output;
}

/**
 * Simple gain kernel for testing.
 *
 * Demonstrates basic buffer operations without complex processing.
 */
kernel void gain(
    const device float* inBuffer  [[buffer(0)]],
    device float*       outBuffer [[buffer(1)]],
    constant float&     gainAmount [[buffer(2)]],
    constant uint& bufferSize [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    // Bounds check to prevent buffer overruns
    if (gid >= bufferSize)
        return;

    outBuffer[gid] = inBuffer[gid] * gainAmount;
}
