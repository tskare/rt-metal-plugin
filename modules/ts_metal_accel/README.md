# ts-metal-accel

JUCE module that lets an audio processor run Metal compute kernels in real time without touching Objective-C++. It supplies a reusable `MetalAccelerator` that hides device setup, kernel compilation, buffer management, and the asynchronous work queue required for glitch-free GPU processing.

## Why use it
- Pure C++23 interface built on Apple's [`metal-cpp`](https://developer.apple.com/metal/cpp/)
- Works inside JUCE plug-ins and standalone hosts
- Real-time–safe asynchronous path with bounded latency and underrun tracking
- `std::expected` errors with detailed `MetalErrorInfo` for better host diagnostics
- Ready-to-use sample kernels and helpers for experiments or production code

## Requirements
- macOS with Metal support (tested on Apple Silicon running macOS 14 and newer)
- JUCE 7.x available to your CMake build (module expects `juce_add_module`)
- Xcode command-line tools and the Metal SDK
- C++23-capable compiler (Apple clang shipped with Xcode 15 or newer)

## Integrating with CMake

```cmake
# Add JUCE first
add_subdirectory(JUCE)

# Add the module directory before juce_add_plugin
juce_add_module(modules/ts_metal_accel)

# Make metal-cpp headers visible to consumers
target_include_directories(ts_metal_accel
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/modules/ts_metal_accel/third_party/metal-cpp)

# Link it into the code that also links JUCE
target_link_libraries(SharedCode
    INTERFACE
    ts_metal_accel
    juce_audio_processors
    juce_audio_utils
    # ... any other modules your project uses
)
```

## Lifecycle at a glance

1. `initialize()` – connect to the shared `MetalContext` and create a command queue.
2. `loadKernel(name, source)` – compile Metal SL into a pipeline state (call for every kernel you plan to run).
3. Optionally configure Phase 2 async processing with `enableAsyncProcessing(config)` in `prepareToPlay()`.
4. In `processBlock()` call either `processBlock()` (blocking, Phase 1) or `processBlockAsync()` (non-blocking, Phase 2).
5. Use `getGPUStats()` to surface telemetry (frames processed, underruns, warm-up state).
6. Call `disableAsyncProcessing()` before changing the async configuration, and `shutdown()` from `releaseResources()`.

## Core API

### `MetalAccelerator`

| Method | When to call | Notes |
| --- | --- | --- |
| `std::expected<void, MetalErrorInfo> initialize()` | Non-real-time (e.g. `prepareToPlay`) | Idempotent; safe to call more than once. Fails if Metal is unavailable. |
| `std::expected<void, MetalErrorInfo> loadKernel(std::string name, std::string mslSource)` | Non-real-time | Compiles the kernel and caches `MTL::ComputePipelineState`. The `name` must match the Metal function symbol. |
| `std::expected<void, MetalErrorInfo> processBlock(std::string name, juce::AudioBuffer<float>& buffer, const void* params = nullptr, size_t paramsSize = 0)` | Audio thread | Phase 1 synchronous path. Copies audio to shared buffers, dispatches, waits for completion, and copies back. Blocks until the GPU has finished. |
| `std::expected<void, MetalErrorInfo> enableAsyncProcessing(const AsyncConfig& config)` | Non-real-time | Allocates frame buffers, primes queues, and starts the worker thread. Resets telemetry counters and sets a warm-up window. |
| `void disableAsyncProcessing()` | Non-real-time | Stops the worker thread, clears buffers, and resets telemetry. Safe to call even if async was not enabled. |
| `std::expected<void, MetalErrorInfo> processBlockAsync(std::string name, juce::AudioBuffer<float>& buffer, const void* params = nullptr, size_t paramsSize = 0)` | Audio thread | Phase 2 path. Returns immediately. Delivers the previous frame’s output (one block of latency) and queues the new frame. |
| `bool isInitialized() const` | Any thread | Lightweight check for initialization. |
| `bool isAsyncEnabled() const` | Any thread | Indicates whether Phase 2 is active. |
| `MetalAccelerator::GPUStats getGPUStats() const` | Any non-real-time thread | Returns counters that are updated atomically. Useful for UI and telemetry. |
| `void shutdown()` | Non-real-time (`releaseResources`) | Tears down all GPU state and the shared context if nobody else is using it. |

### `AsyncConfig`

```
struct AsyncConfig
{
    int maxChannels = 2;         // Upper bound for audio channels processed
    int maxSamplesPerBlock = 512;// Upper bound for samples per channel
    size_t queueDepth = 3;       // Buffered frames; latency == one audio block
    size_t maxParameterBytes = 256; // Optional parameter payload size
};
```

Validation rules:
- `maxChannels` must be between 1 and 32.
- `maxSamplesPerBlock` must be between 1 and 8192.
- `queueDepth` must be at least 1 and strictly less than the internal queue capacity (currently 8). Each additional frame increases GPU/CPU memory usage.
- `maxParameterBytes` should cover the largest parameter blob you pass to `processBlockAsync`. A per-frame copy is taken into the frame’s `paramStorage`.

### `GPUStats`

```
struct GPUStats
{
    uint64_t totalFrames = 0;          // Frames counted after warm-up
    uint64_t underrunCount = 0;        // Times no completed frame was available
    double underrunRate = 0.0;         // underrunCount / totalFrames
    bool asyncEnabled = false;         // Phase 2 status
    uint64_t warmupFramesRemaining = 0;// Frames left before counters start
};
```

The warm-up period defaults to `max(12, queueDepth)` frames. During warm-up the audio thread still outputs silence on underrun, but `totalFrames` and `underrunCount` are not incremented. This avoids reporting spurious underruns while the pipeline fills.

## Synchronous vs asynchronous processing

| Aspect | Phase 1 (`processBlock`) | Phase 2 (`processBlockAsync`) |
| --- | --- | --- |
| Real-time safety | Not safe – audio thread blocks on `waitUntilCompleted()` | Designed to be RT-safe (no blocking, no heap allocations) |
| Latency | 0 samples | One block (queue depth determines buffering, but only one block of extra latency is introduced) |
| Memory | Allocates shared buffers on first use for the requested block size | Pre-allocates frame pool up front based on `AsyncConfig` |
| Failure handling | Returns `std::unexpected`; caller decides how to react | Returns `std::unexpected`, disables async, and the caller can fall back to sync path |
| Best use | Quick prototypes, testing on systems without async support | Production use and any scenario needing deterministic timing |

You can enable async mode at runtime and fall back automatically if Metal returns an error. After disabling async, the next call to `processBlockAsync()` will report `MetalError::NotInitialized`; always check the return value.

## FDTD reverb demo (2x2)

The repository ships with an optional `FDTDReverbEngine` (module `ts_metal_accel_fdtd`) that runs a staggered-grid 3‑D acoustic solver on the GPU. It mirrors the CPU reference implementation so you can A/B behaviour and collect baselines.

### Engine essentials

```cpp
ts::metal::fdtd::FDTDReverbConfig cfg{};
cfg.solver.grid = { 24, 20, 18 };
cfg.solver.dx = 0.03f;
cfg.solver.dt = 1.0f / 48000.0f;
cfg.solver.boundaryAttenuation = 0.995f;
cfg.sources[0] = { 3, 2, 2, 1.0f };     // left input
cfg.sources[1] = { 20, 2, 2, 1.0f };    // right input
cfg.mics[0] = { 12, 10, 9, 1.0f };      // left output
cfg.mics[1] = { 13, 10, 9, 1.0f };      // right output

ts::metal::fdtd::FDTDReverbEngine engine;
std::string error;
if (!engine.prepare(cfg, error))
    throw std::runtime_error(error);

const float* inputs[2] = { inLeft, inRight };
float* outputs[2] = { outLeft, outRight };
engine.processBlock(inputs, 2, outputs, 2, numSamples);
```

- The engine auto-selects the GPU when Metal is available; otherwise it falls back to the CPU solver without changing behaviour.
- Call `engine.statusString()` to surface whether the GPU path is live (`"FDTD GPU (Nx x Ny x Nz)"`) or running on the CPU.
- `prepare()` is idempotent and reconfigures buffer allocations; `reset()` clears state between sessions.
- The example plugin maps parameters to room size (three preset grids), boundary absorption, and wet/dry mix; updates reconfigure the solver off the audio thread.
- GPU mode now batches an entire audio block into a single command buffer: source samples are uploaded once, kernels step the grid block-by-block, and mic taps are collected in one readback.

The standalone plugin toggles the reverb through an "Enable FDTD Reverb" switch, routing two inputs into the solver and exposing two microphone taps on the output. Impulses and microphone placements can be adjusted in `PluginProcessor::configureFDTD()`.

## Error handling

Every public method that can fail returns `std::expected<void, MetalErrorInfo>`. `MetalErrorInfo` contains:

```
enum class MetalError
{
    DeviceNotFound,
    CompilationFailed,
    InvalidKernel,
    InvalidBuffer,
    ExecutionFailed,
    NotInitialized,
    AlreadyInitialized,
    Unknown
};

struct MetalErrorInfo
{
    MetalError code;
    std::string message;
};
```

Handle failures immediately and decide whether to retry, disable async, or report to the host. Error messages come directly from Metal when available (e.g. shader compile errors).

## Threading model and real-time considerations

- The audio thread pushes frames into a single-producer queue and dequeues completed frames from a multi-producer queue (`moodycamel::ConcurrentQueue`).
- A dedicated worker thread drains the job queue, records Metal command buffers, and relies on completion handlers to recycle frames.
- No heap allocations occur on the audio thread once `enableAsyncProcessing()` succeeds.
- Underruns are counted when the audio thread fails to dequeue a completed frame. When that happens, the current block is cleared to silence.
- `setLatencySamples()` is called automatically to keep DAW delay compensation accurate.
- Changing async configuration is coordinated via atomics; call `disableAsyncProcessing()` from a non-real-time context when possible.

## Shader authoring notes

- Kernels receive deinterleaved channel data. The audio buffer is flattened as `[channel][sample]`.
- Argument indices are fixed: `buffer(0)` input samples, `buffer(1)` output samples, `buffer(2)` optional parameter struct, `buffer(3)` total sample count (for bounds checking).
- Threadgroup sizing defaults to the pipeline’s preferred width with clamping to the actual workload.
- Keep kernels branch-light and use `metal::tanh`/`fast::` variants when acceptable.

## Example: enabling async mode with fallback

```cpp
void PluginProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    juce::ignoreUnused(sampleRate);

    if (auto init = accelerator.initialize(); !init)
    {
        DBG("Metal init failed: " + init.error().message);
        return;
    }

    if (auto load = accelerator.loadKernel("soft_clipper", BinaryData::DistortionKernel_metal.toString()); !load)
    {
        DBG("Kernel load failed: " + load.error().message);
        return;
    }

    ts::metal::MetalAccelerator::AsyncConfig asyncConfig;
    asyncConfig.maxChannels = getTotalNumInputChannels();
    asyncConfig.maxSamplesPerBlock = samplesPerBlock;
    asyncConfig.queueDepth = 3;
    asyncConfig.maxParameterBytes = sizeof(DistortionParams);

    if (auto async = accelerator.enableAsyncProcessing(asyncConfig); !async)
    {
        DBG("Async unavailable, staying synchronous: " + async.error().message);
    }
}

void PluginProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi)
{
    juce::ignoreUnused(midi);

    if (accelerator.isAsyncEnabled())
    {
        if (auto async = accelerator.processBlockAsync("soft_clipper", buffer, &params, sizeof(params)); !async)
        {
            DBG("Async failed, dropping back to sync: " + async.error().message);
            accelerator.disableAsyncProcessing();
            if (auto sync = accelerator.processBlock("soft_clipper", buffer, &params, sizeof(params)); !sync)
            {
                DBG("Sync also failed: " + sync.error().message);
                buffer.clear();
            }
        }
        return;
    }

    if (auto sync = accelerator.processBlock("soft_clipper", buffer, &params, sizeof(params)); !sync)
    {
        DBG("Sync processing failed: " + sync.error().message);
        buffer.clear();
    }
}
```

## Roadmap

- More example kernels (FFT, convolution, physical modelling)
- Optional Metal Performance Shaders backend
- Half-precision compute path
- Extended benchmark harness covering async stress cases
- iOS validation once real hardware testing is available

## License

MIT License – see `../../LICENSE`.

### Third-party notices

- `metal-cpp`: Apache License 2.0 (`third_party/LICENSE-metal-cpp.txt`)
- `moodycamel::ConcurrentQueue`: BSD 2-Clause (`third_party/LICENSE-concurrentqueue.md`)
