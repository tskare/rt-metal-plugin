#pragma once

namespace ts::metal::fdtd {

inline constexpr const char* kFDTDMetalKernels = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct FDTDUniforms
{
    uint nx;
    uint ny;
    uint nz;
    float coeffVelocity;
    float coeffPressure;
    float boundaryAttenuation;
};

struct SourceCommand
{
    uint index;
};

struct MicCommand
{
    float x;
    float y;
    float z;
    float gain;
};

inline bool isBoundary(uint coord, uint extent)
{
    return coord == 0u || coord == extent - 1u;
}

inline uint linearIndex(uint nx, uint ny, uint x, uint y, uint z)
{
    return x + nx * (y + ny * z);
}

kernel void fdtd_apply_sources(
    device float* pressure                        [[buffer(0)]],
    const device SourceCommand* sources           [[buffer(1)]],
    const device float* samples                   [[buffer(2)]],
    constant FDTDUniforms& uniforms               [[buffer(3)]],
    constant uint& sourceCount                    [[buffer(4)]],
    uint gid                                     [[thread_position_in_grid]])
{
    if (gid >= sourceCount)
        return;

    const uint writeIndex = sources[gid].index;
    pressure[writeIndex] += samples[gid];
}

kernel void fdtd_update_velocity(
    constant FDTDUniforms& uniforms              [[buffer(0)]],
    const device float* pressureCurrent          [[buffer(1)]],
    const device float* velocityXCurrent         [[buffer(2)]],
    const device float* velocityYCurrent         [[buffer(3)]],
    const device float* velocityZCurrent         [[buffer(4)]],
    device float* velocityXNext                  [[buffer(5)]],
    device float* velocityYNext                  [[buffer(6)]],
    device float* velocityZNext                  [[buffer(7)]],
    uint gid                                     [[thread_position_in_grid]])
{
    const uint nx = uniforms.nx;
    const uint ny = uniforms.ny;
    const uint nz = uniforms.nz;
    const uint total = nx * ny * nz;

    if (gid >= total)
        return;

    const uint layer = nx * ny;
    const uint z = gid / layer;
    const uint rem = gid % layer;
    const uint y = rem / nx;
    const uint x = rem % nx;

    float px = 0.0f;
    float py = 0.0f;
    float pz = 0.0f;

    if (x + 1u < nx)
        px = pressureCurrent[gid + 1u] - pressureCurrent[gid];
    if (y + 1u < ny)
        py = pressureCurrent[gid + nx] - pressureCurrent[gid];
    if (z + 1u < nz)
        pz = pressureCurrent[gid + layer] - pressureCurrent[gid];

    const float dampX = isBoundary(x, nx) ? uniforms.boundaryAttenuation : 1.0f;
    const float dampY = isBoundary(y, ny) ? uniforms.boundaryAttenuation : 1.0f;
    const float dampZ = isBoundary(z, nz) ? uniforms.boundaryAttenuation : 1.0f;

    velocityXNext[gid] = dampX * (velocityXCurrent[gid] - uniforms.coeffVelocity * px);
    velocityYNext[gid] = dampY * (velocityYCurrent[gid] - uniforms.coeffVelocity * py);
    velocityZNext[gid] = dampZ * (velocityZCurrent[gid] - uniforms.coeffVelocity * pz);
}

kernel void fdtd_update_pressure(
    constant FDTDUniforms& uniforms              [[buffer(0)]],
    const device float* pressureCurrent          [[buffer(1)]],
    const device float* velocityXNext            [[buffer(2)]],
    const device float* velocityYNext            [[buffer(3)]],
    const device float* velocityZNext            [[buffer(4)]],
    device float* pressureNext                   [[buffer(5)]],
    uint gid                                     [[thread_position_in_grid]])
{
    const uint nx = uniforms.nx;
    const uint ny = uniforms.ny;
    const uint nz = uniforms.nz;
    const uint total = nx * ny * nz;

    if (gid >= total)
        return;

    const uint layer = nx * ny;
    const uint z = gid / layer;
    const uint rem = gid % layer;
    const uint y = rem / nx;
    const uint x = rem % nx;

    float dvx = velocityXNext[gid];
    float dvy = velocityYNext[gid];
    float dvz = velocityZNext[gid];

    if (x > 0u)
        dvx -= velocityXNext[gid - 1u];
    if (y > 0u)
        dvy -= velocityYNext[gid - nx];
    if (z > 0u)
        dvz -= velocityZNext[gid - layer];

    const float divergence = dvx + dvy + dvz;
    const bool boundaryCell = isBoundary(x, nx) || isBoundary(y, ny) || isBoundary(z, nz);
    const float damp = boundaryCell ? uniforms.boundaryAttenuation : 1.0f;

    pressureNext[gid] = damp * (pressureCurrent[gid] - uniforms.coeffPressure * divergence);
}

kernel void fdtd_sample_mics(
    const device float* pressure                  [[buffer(0)]],
    const device MicCommand* mics                 [[buffer(1)]],
    device float* outputs                         [[buffer(2)]],
    constant FDTDUniforms& uniforms               [[buffer(3)]],
    constant uint& sampleIndex                    [[buffer(4)]],
    constant uint& micCount                       [[buffer(5)]],
    uint gid                                     [[thread_position_in_grid]])
{
    if (gid >= micCount)
        return;

    const float nx = static_cast<float>(uniforms.nx);
    const float ny = static_cast<float>(uniforms.ny);
    const float nz = static_cast<float>(uniforms.nz);

    float x = clamp(mics[gid].x, 0.0f, nx - 1.0f);
    float y = clamp(mics[gid].y, 0.0f, ny - 1.0f);
    float z = clamp(mics[gid].z, 0.0f, nz - 1.0f);

    const uint x0 = static_cast<uint>(floor(x));
    const uint y0 = static_cast<uint>(floor(y));
    const uint z0 = static_cast<uint>(floor(z));

    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const float fz = z - static_cast<float>(z0);

    const uint x1 = min(x0 + 1u, uniforms.nx - 1u);
    const uint y1 = min(y0 + 1u, uniforms.ny - 1u);
    const uint z1 = min(z0 + 1u, uniforms.nz - 1u);

    const float p000 = pressure[linearIndex(uniforms.nx, uniforms.ny, x0, y0, z0)];
    const float p001 = pressure[linearIndex(uniforms.nx, uniforms.ny, x0, y0, z1)];
    const float p010 = pressure[linearIndex(uniforms.nx, uniforms.ny, x0, y1, z0)];
    const float p011 = pressure[linearIndex(uniforms.nx, uniforms.ny, x0, y1, z1)];
    const float p100 = pressure[linearIndex(uniforms.nx, uniforms.ny, x1, y0, z0)];
    const float p101 = pressure[linearIndex(uniforms.nx, uniforms.ny, x1, y0, z1)];
    const float p110 = pressure[linearIndex(uniforms.nx, uniforms.ny, x1, y1, z0)];
    const float p111 = pressure[linearIndex(uniforms.nx, uniforms.ny, x1, y1, z1)];

    const float px00 = mix(p000, p100, fx);
    const float px01 = mix(p001, p101, fx);
    const float px10 = mix(p010, p110, fx);
    const float px11 = mix(p011, p111, fx);

    const float pxy0 = mix(px00, px10, fy);
    const float pxy1 = mix(px01, px11, fy);

    const float value = mix(pxy0, pxy1, fz);
    const uint writeIndex = sampleIndex * micCount + gid;
    outputs[writeIndex] = value * mics[gid].gain;
}

// Single-threadgroup kernel that processes an entire block of samples
kernel void fdtd_process_block(
    device float* pressure0                       [[buffer(0)]],
    device float* pressure1                       [[buffer(1)]],
    device float* velocityX0                      [[buffer(2)]],
    device float* velocityX1                      [[buffer(3)]],
    device float* velocityY0                      [[buffer(4)]],
    device float* velocityY1                      [[buffer(5)]],
    device float* velocityZ0                      [[buffer(6)]],
    device float* velocityZ1                      [[buffer(7)]],
    const device float* sourceData                [[buffer(8)]],
    device float* micData                         [[buffer(9)]],
    const device SourceCommand* sources           [[buffer(10)]],
    const device MicCommand* mics                 [[buffer(11)]],
    constant FDTDUniforms& uniforms               [[buffer(12)]],
    constant uint& numSamples                     [[buffer(13)]],
    constant uint& sourceCount                    [[buffer(14)]],
    constant uint& micCount                       [[buffer(15)]],
    constant uint& initialActiveBuffer            [[buffer(16)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]])
{
    const uint nx = uniforms.nx;
    const uint ny = uniforms.ny;
    const uint nz = uniforms.nz;
    const uint totalCells = nx * ny * nz;
    const uint cellsPerThread = (totalCells + tgSize - 1) / tgSize;
    const uint startCell = tid * cellsPerThread;
    const uint endCell = min(startCell + cellsPerThread, totalCells);

    // Local buffer parity tracker
    uint bufferIndex = initialActiveBuffer;

    // Main sample loop
    for (uint sample = 0; sample < numSamples; ++sample)
    {
        // Select active/next buffers based on parity
        device float* pCurr = (bufferIndex == 0) ? pressure0 : pressure1;
        device float* pNext = (bufferIndex == 0) ? pressure1 : pressure0;
        device float* vxCurr = (bufferIndex == 0) ? velocityX0 : velocityX1;
        device float* vxNext = (bufferIndex == 0) ? velocityX1 : velocityX0;
        device float* vyCurr = (bufferIndex == 0) ? velocityY0 : velocityY1;
        device float* vyNext = (bufferIndex == 0) ? velocityY1 : velocityY0;
        device float* vzCurr = (bufferIndex == 0) ? velocityZ0 : velocityZ1;
        device float* vzNext = (bufferIndex == 0) ? velocityZ1 : velocityZ0;

        // 1. Apply sources (only threads 0..sourceCount-1)
        if (tid < sourceCount)
        {
            const uint srcIdx = sources[tid].index;
            const float value = sourceData[sample * sourceCount + tid];
            pCurr[srcIdx] += value;
        }

        // Synchronize after sources
        threadgroup_barrier(mem_flags::mem_device);

        // 2. Update velocity (all threads process their cells)
        for (uint cell = startCell; cell < endCell; ++cell)
        {
            // Compute x, y, z from linear index
            const uint layer = nx * ny;
            const uint z = cell / layer;
            const uint rem = cell % layer;
            const uint y = rem / nx;
            const uint x = rem % nx;

            // Compute pressure gradients
            float px = 0.0f;
            float py = 0.0f;
            float pz = 0.0f;

            if (x + 1u < nx)
                px = pCurr[cell + 1u] - pCurr[cell];
            if (y + 1u < ny)
                py = pCurr[cell + nx] - pCurr[cell];
            if (z + 1u < nz)
                pz = pCurr[cell + layer] - pCurr[cell];

            const float dampX = isBoundary(x, nx) ? uniforms.boundaryAttenuation : 1.0f;
            const float dampY = isBoundary(y, ny) ? uniforms.boundaryAttenuation : 1.0f;
            const float dampZ = isBoundary(z, nz) ? uniforms.boundaryAttenuation : 1.0f;

            vxNext[cell] = dampX * (vxCurr[cell] - uniforms.coeffVelocity * px);
            vyNext[cell] = dampY * (vyCurr[cell] - uniforms.coeffVelocity * py);
            vzNext[cell] = dampZ * (vzCurr[cell] - uniforms.coeffVelocity * pz);
        }

        // Synchronize after velocity
        threadgroup_barrier(mem_flags::mem_device);

        // 3. Update pressure (all threads process their cells)
        for (uint cell = startCell; cell < endCell; ++cell)
        {
            const uint layer = nx * ny;
            const uint z = cell / layer;
            const uint rem = cell % layer;
            const uint y = rem / nx;
            const uint x = rem % nx;

            float dvx = vxNext[cell];
            float dvy = vyNext[cell];
            float dvz = vzNext[cell];

            if (x > 0u)
                dvx -= vxNext[cell - 1u];
            if (y > 0u)
                dvy -= vyNext[cell - nx];
            if (z > 0u)
                dvz -= vzNext[cell - layer];

            const float divergence = dvx + dvy + dvz;
            const bool boundaryCell = isBoundary(x, nx) || isBoundary(y, ny) || isBoundary(z, nz);
            const float damp = boundaryCell ? uniforms.boundaryAttenuation : 1.0f;

            pNext[cell] = damp * (pCurr[cell] - uniforms.coeffPressure * divergence);
        }

        // Synchronize after pressure
        threadgroup_barrier(mem_flags::mem_device);

        // 4. Sample microphones (only threads 0..micCount-1)
        if (tid < micCount)
        {
            const float mx = clamp(mics[tid].x, 0.0f, float(nx - 1u));
            const float my = clamp(mics[tid].y, 0.0f, float(ny - 1u));
            const float mz = clamp(mics[tid].z, 0.0f, float(nz - 1u));

            const uint x0 = uint(floor(mx));
            const uint y0 = uint(floor(my));
            const uint z0 = uint(floor(mz));

            const float fx = mx - float(x0);
            const float fy = my - float(y0);
            const float fz = mz - float(z0);

            const uint x1 = min(x0 + 1u, nx - 1u);
            const uint y1 = min(y0 + 1u, ny - 1u);
            const uint z1 = min(z0 + 1u, nz - 1u);

            // Read pressure values for trilinear interpolation
            const float p000 = pNext[linearIndex(nx, ny, x0, y0, z0)];
            const float p001 = pNext[linearIndex(nx, ny, x0, y0, z1)];
            const float p010 = pNext[linearIndex(nx, ny, x0, y1, z0)];
            const float p011 = pNext[linearIndex(nx, ny, x0, y1, z1)];
            const float p100 = pNext[linearIndex(nx, ny, x1, y0, z0)];
            const float p101 = pNext[linearIndex(nx, ny, x1, y0, z1)];
            const float p110 = pNext[linearIndex(nx, ny, x1, y1, z0)];
            const float p111 = pNext[linearIndex(nx, ny, x1, y1, z1)];

            // Trilinear interpolation
            const float px00 = mix(p000, p100, fx);
            const float px01 = mix(p001, p101, fx);
            const float px10 = mix(p010, p110, fx);
            const float px11 = mix(p011, p111, fx);

            const float pxy0 = mix(px00, px10, fy);
            const float pxy1 = mix(px01, px11, fy);

            const float value = mix(pxy0, pxy1, fz);
            const uint writeIndex = sample * micCount + tid;
            micData[writeIndex] = value * mics[tid].gain;
        }

        // Synchronize before next sample
        threadgroup_barrier(mem_flags::mem_device);

        // Flip buffer parity for next iteration
        bufferIndex = 1u - bufferIndex;
    }
}
)METAL";

}  // namespace ts::metal::fdtd
