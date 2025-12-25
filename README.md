# EEG Linear Filter Benchmark Suite

This project is a high-performance computing benchmark suite designed to evaluate different approaches to **linear convolution** on EEG signals. The application processes standard EDF (European Data Format) files and applies a Gaussian filter using various architectural strategies, ranging from naive CPU implementations to highly optimized SIMD vectorization and GPU acceleration via Metal.

The goal is to compare the performance (execution time, GFLOPS, memory throughput) of sequential, parallel, and GPU-based algorithms on Apple Silicon hardware.

## 🚀 Features

* **12 Different Processing Modes**: Comprehensive comparison of CPU vs. GPU.
* **Apple Silicon Optimization**: Utilizes **NEON** instruction set for manual vectorization and **Metal API** for GPU compute.
* **EDF File Support**: Natively reads and parses `.edf` files (using `edflib`).
* **Interactive CLI**: easy-to-use command-line interface for configuring benchmark parameters.
* **Automatic Benchmarking**: Configurable iteration counts and robust result metrics.
* **Python Analysis Suite**: Includes scripts to generate performance graphs, speedup matrices, and scaling tables.

## 🛠️ Supported Architectures & Modes

The application implements the convolution algorithm in the following modes (defined in `config.h`):

### CPU Sequential
* `CPU_SEQ_APPLE`: Baseline using Apple's optimized `vDSP_conv`.
* `CPU_SEQ_NAIVE`: Standard C++ implementation without explicit optimization.
* `CPU_SEQ_NO_VEC`: Sequential processing with vectorization disabled.
* `CPU_SEQ_AUTO_VEC`: Relies on the compiler's auto-vectorizer.
* `CPU_SEQ_MANUAL_VEC`: Optimized using explicit **ARM NEON** intrinsics.

### CPU Parallel
Multithreaded implementations splitting the workload across available cores:
* `CPU_PAR_NAIVE`: Naive parallel implementation.
* `CPU_PAR_NO_VEC`: Parallel processing without vectorization.
* `CPU_PAR_AUTO_VEC`: Parallel processing with auto-vectorization.
* `CPU_PAR_MANUAL_VEC`: Parallel processing combined with **ARM NEON** intrinsics.

### GPU (Metal)
Hardware-accelerated implementations using custom Metal shaders (`.metal`):
* `GPU_NAIVE`: Basic compute kernel.
* `GPU_32BIT`: Optimized kernel using threadgroup memory and loop unrolling (FP32).
* `GPU_16BIT`: Highly optimized kernel using half-precision floating point (FP16).

## 💻 Requirements

* **Hardware**: Mac with Apple Silicon (M1/M2/M3/...) recommended for NEON and Metal support.
* **OS**: macOS 12.0 or later.
* **Tools**: 
    * Xcode 14+ (for C++ & Metal compilation)
    * Python 3.x (for results analysis)

## ⏯️ Usage

Run the application directly from Xcode (`Cmd + R`) or via the terminal executable. The application features an interactive menu:

1.  **Input File**: Provide the path to an `.edf` file. If the file is missing, the app can attempt to download a sample dataset.
2.  **Select Mode**: Choose a specific algorithm index (0-11) or select `-1` to run the **Whole Benchmark Suite**.
3.  **Iterations**: Set the number of test runs for statistical robustness (default: 10).
4.  **Save Results**: Choose `y` to save filtered data to EDF file.
5.  **Output Path**: Define where results and filtered data should be stored.

## 📊 Analyzing Results

The project includes Python scripts to visualize the benchmark data.

1.  **Locate the logs**:
    By default, results are saved to `EegLinearFilter/logs/benchmark_results.csv`.

2.  **Run the analysis script**:
    ```bash
    python3 python/benchmark.py
    ```

3.  **View Reports**:
    The script generates PDF reports in `python/benchmark_results/`, including:
    * `combined_summary.pdf`: Comparison of all modes.
    * `speedup_matrix.pdf`: Heatmap showing relative speedups between modes.
    * `scaling_analysis_gflops.pdf`: Performance scaling by kernel radius.

## 📂 Project Structure

* `EegLinearFilter/`
  * `src/`: C++ source code.
      * `main.cpp`: Entry point.
      * `processors/`: Implementations of convolution algorithms (CPU & GPU).
      * `io/`: Input/Output handling (EDF loading, User input).
  * `lib/`: External libraries (`edflib`, `metal-cpp`, `magic_enum`).
* `python/`: Analysis scripts.
* `EegLinearFilter.xcodeproj`: Xcode project settings.

## 📜 License

This project is open-source and available for educational purposes.
