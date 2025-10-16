# Repository Guidelines

## Project Structure & Module Organization
MetalAudioPlugin extends the Pamplejuce layout for real-time Metal experiments. Core JUCE sources live in `source/` for the processor and editor; unit helpers are in `tests/helpers` with entry points in `tests/*.cpp`. GPU kernels, Metal shaders, and their wrappers live in `modules/ts-metal-accel`, while shared JUCE helpers sit under `modules/melatonin_inspector` and `modules/clap-juce-extensions`. Benchmarks exploring kernel timing belong in `benchmarks/`, runtime assets in `assets/` and `packaging/`, and exploratory notes or scripts in `research/`.

## Build, Test, and Development Commands
- `cmake -S . -B build -G Ninja`: configure a clean tree (choose `-G Xcode` if you rely on IDE schemes).
- `cmake --build build --target MetalAudioPlugin_Standalone --config Debug`: build and copy the standalone host plus plug-in binaries.
- `cmake --build build --target Tests`: produce the Catch2-driven test runner; switch `--config Release` when profiling GPU code.
- `ctest --test-dir build --output-on-failure`: execute the suite; to focus, run `./build/Tests "[gpu]"` for tagged cases.
- `cmake --build build --target Benchmarks`: build the timing harness before profiling.

## Coding Style & Naming Conventions
Use the repo’s `.clang-format` (four-space indent, Allman braces, left-aligned pointers, unlimited column width). Keep JUCE types and plugin classes in `UpperCamelCase`, methods in `camelCase`, and free helpers in `snake_case`. Run `clang-format -i file.cpp` before submission.

## Testing Guidelines
Catch2 3.8 is integrated through `catch_discover_tests`, so every new `.cpp` in `tests/` is auto-registered. Co-locate scenarios with the feature you touch and mark GPU-specific cases with the `[gpu]` tag plus `#ifdef RUN_PAMPLEJUCE_TESTS` guards when Metal hardware is required. Ensure `ctest` and the standalone `Tests` binary pass, keeping fixtures deterministic in `tests/helpers`.

## Commit & Pull Request Guidelines
Write imperative, sub-72 character subjects (e.g., `Wire GPU buffer uploader`) and keep each commit scoped to one behavior change. Update docs or `VERSION` when formats, identifiers, or build flags move. Pull requests should note which plugin formats were built and which host DAWs were used for validation.

## Metal & Configuration Tips
Run `git submodule update --init --recursive` after cloning to fetch JUCE and GPU modules. Ensure Xcode command-line tools and the Metal SDK are installed; regenerate builds with `cmake -S . -B build -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64"` when you need universal binaries. Profile kernels in Release mode and record baselines in `benchmarks/` to catch regressions early.

## GPU Reverb Workflow (2025-03)
- The FDTD engine now runs exclusively through the asynchronous Metal worker (`modules/ts_metal_accel_fdtd/Source/FDTDReverbEngine.*`). The audio thread enqueues frames via lock-free queues and never calls `waitUntilCompleted()`.
- Every block snapshots the input channels before handing work to the GPU. If the accelerator fails (queue exhaustion, watchdog, allocation error) the engine keeps delivering the dry signal while the UI surfaces the warning; we no longer fall back to the CPU solver.
- The async queue depth scales with the grid size and metrics (frames, underruns, watchdog trips, `processBlock()` calls, latency samples) are exposed in `PluginEditor`. Warm-up spans two queue lengths so averages only include steady-state frames.
- Watchdog timeouts default to 0 ns for R&D. Set a non-zero budget via `FDTDReverbEngine::setWatchdogTimeout()` before enabling GPU mode if you need automatic failure detection.

## Shutdown Discipline
- `FDTDReverbEngine::~FDTDReverbEngine()` calls `shutdown()`, which drains and joins the worker thread, clears Metal buffers, and resets state. Always call `fdtdEngine.shutdown()` from `releaseResources()` (already wired in `PluginProcessor`) when you touch lifecycle code.
- The GPU worker completion handlers still fire after `commit()`. Make sure any future changes keep `inFlight` guards and queue drains so that no callbacks reference freed memory.

## Profiling & Diagnostics
- Use the combined metrics label in `PluginEditor` to confirm GPU health. `processBlock()` increments once per audio callback; `Enqueue attempts`/`drops` show queue pressure, and `Latency samples` tells you how many frames the averages represent.
- For targeted profiling, adjust `gpu.async.queueCapacity` or the room size (small/medium/large) and watch how queue depth and watchdog counters respond. The dry path remains audible even when wet frames miss their deadline.
