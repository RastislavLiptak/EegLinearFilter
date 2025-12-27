import torch
import time

BATCH_SIZE = 128
CHANNELS = 64
SIGNAL_LENGTH = 10000
KERNEL_SIZE = 513
ITERATIONS = 10

def run_benchmark():
    if not torch.backends.mps.is_available():
        print("Error: MPS (Metal Performance Shaders) is not available.")
        return

    device = torch.device("mps")
    print(f"Running on device: {device}")
    print(f"Configuration: Batch={BATCH_SIZE}, Channels={CHANNELS}, Length={SIGNAL_LENGTH}, Kernel={KERNEL_SIZE}")
    print(f"Iterations: {ITERATIONS}")

    input_signal = torch.randn(BATCH_SIZE, CHANNELS, SIGNAL_LENGTH, device=device)
    filters = torch.randn(CHANNELS, CHANNELS, KERNEL_SIZE, device=device)

    _ = torch.nn.functional.conv1d(input_signal, filters, padding='same')
    torch.mps.synchronize()

    start_time = time.time()

    for _ in range(ITERATIONS):
        _ = torch.nn.functional.conv1d(input_signal, filters, padding='same')

    torch.mps.synchronize()
    end_time = time.time()

    total_duration = end_time - start_time
    avg_duration = total_duration / ITERATIONS

    ops_per_pass = 2 * BATCH_SIZE * CHANNELS * SIGNAL_LENGTH * KERNEL_SIZE * CHANNELS
    gflops = (ops_per_pass / avg_duration) / 1e9

    print("-" * 40)
    print(f"Total Duration:     {total_duration:.4f} s")
    print(f"Average Time/Pass:  {avg_duration:.4f} s")
    print(f"Sustained Perf:     {gflops:.2f} GFLOPS")
    print("-" * 40)

if __name__ == "__main__":
    run_benchmark()