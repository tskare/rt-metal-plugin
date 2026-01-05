#pragma once

namespace ts::metal::waveguide {

constexpr auto kWaveguideKernelSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct WaveguideParams {
    int lengthSamples;
    float inputTap;
    float outputTap;
    float filterCoeff;
    float nonlinearityAmount;
    float gain;
    float pan;
};

struct WaveguideUniforms {
    uint numWaveguides;
    uint numSamples;
    uint maxDelayLength;
    uint halfWaveguideCount;
};

inline float readDelayLine(device const float* delayLine,
                           int length,
                           int writePos,
                           float tapPosition) {
    float readPos = tapPosition * float(length - 1);
    int idx0 = int(readPos);
    int idx1 = idx0 + 1;
    if (idx1 >= length) idx1 = 0;
    float frac = readPos - float(idx0);

    int pos0 = (writePos - 1 - idx0 + length) % length;
    int pos1 = (writePos - 1 - idx1 + length) % length;

    return mix(delayLine[pos0], delayLine[pos1], frac);
}

inline float saturate(float x, float amount) {
    if (amount < 0.001f) return x;
    float scaled = x * (1.0f + amount * 3.0f);
    return tanh(scaled) / tanh(1.0f + amount * 3.0f);
}

kernel void waveguide_process_block(
    device float* delayLines [[buffer(0)]],
    device const WaveguideParams* params [[buffer(1)]],
    device const float* inputL [[buffer(2)]],
    device const float* inputR [[buffer(3)]],
    device atomic_float* outputL [[buffer(4)]],
    device atomic_float* outputR [[buffer(5)]],
    device float* filterState [[buffer(6)]],
    device int* writePositions [[buffer(7)]],
    constant WaveguideUniforms& uniforms [[buffer(8)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= uniforms.numWaveguides) return;

    const WaveguideParams p = params[gid];
    const int length = min(p.lengthSamples, int(uniforms.maxDelayLength));
    const uint maxLen = uniforms.maxDelayLength;

    device float* myDelay = delayLines + gid * maxLen;
    int writePos = writePositions[gid];
    float state = filterState[gid];

    device const float* input = (gid < uniforms.halfWaveguideCount) ? inputL : inputR;

    float panL = (1.0f - p.pan) * 0.5f;
    float panR = (1.0f + p.pan) * 0.5f;

    for (uint s = 0; s < uniforms.numSamples; s++) {
        float delayed = readDelayLine(myDelay, length, writePos, p.outputTap);

        state = p.filterCoeff * state + (1.0f - p.filterCoeff) * delayed;

        float processed = saturate(state, p.nonlinearityAmount);

        float inputSample = input[s];
        int inputPos = int(p.inputTap * float(length - 1));
        inputPos = (writePos + inputPos) % length;
        myDelay[inputPos] += inputSample;

        myDelay[writePos] = processed;
        writePos = (writePos + 1) % length;

        float outSample = processed * p.gain;
        atomic_fetch_add_explicit(&outputL[s], outSample * panL, memory_order_relaxed);
        atomic_fetch_add_explicit(&outputR[s], outSample * panR, memory_order_relaxed);
    }

    filterState[gid] = state;
    writePositions[gid] = writePos;
}

kernel void waveguide_clear_output(
    device atomic_float* outputL [[buffer(0)]],
    device atomic_float* outputR [[buffer(1)]],
    constant uint& numSamples [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < numSamples) {
        atomic_store_explicit(&outputL[gid], 0.0f, memory_order_relaxed);
        atomic_store_explicit(&outputR[gid], 0.0f, memory_order_relaxed);
    }
}
)METAL";

}  // namespace ts::metal::waveguide
