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
