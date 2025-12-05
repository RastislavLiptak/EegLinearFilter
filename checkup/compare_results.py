import os
import glob
import itertools
import numpy as np
import shutil
import csv
import pyedflib
import matplotlib.pyplot as plt

GENERATE_CSV = False
GENERATE_PLOTS = True 
DATA_FOLDER = "EegLinearFilter/out"
OUTPUT_FOLDER = "checkup/result_comparison"
CHUNK_SIZE_BYTES = 100 * 1024 * 1024 

def get_header_size(num_channels):
    return 256 + (num_channels * 256)

def get_edf_layout(filename):
    with pyedflib.EdfReader(filename) as f:
        n_signals = f.signals_in_file
        samples_per_record = []
        for i in range(n_signals):
            samples_per_record.append(f.getNSamples()[i] // f.datarecords_in_file)
    return np.array(samples_per_record, dtype=int)

def map_flat_index_to_channel(flat_index, layout, samples_per_record_sum):
    record_idx = flat_index // samples_per_record_sum
    offset_in_record = flat_index % samples_per_record_sum
    
    current_pos = 0
    channel_idx = 0
    sample_in_record_offset = 0
    
    for i, n_samples in enumerate(layout):
        if offset_in_record < (current_pos + n_samples):
            channel_idx = i
            sample_in_record_offset = offset_in_record - current_pos
            break
        current_pos += n_samples
        
    global_sample_idx = (record_idx * layout[channel_idx]) + sample_in_record_offset
    
    return channel_idx, global_sample_idx

def compare_raw_binary(file_a, file_b, n_channels, writer=None, file_name_pair=None):
    header_size = get_header_size(n_channels)

    size_a = os.path.getsize(file_a)
    size_b = os.path.getsize(file_b)
    
    if size_a != size_b:
        return None, None, f"SIZE MISMATCH ({size_a} vs {size_b})"

    layout = None
    layout_sum = 0
    if writer:
        layout = get_edf_layout(file_a)
        layout_sum = np.sum(layout)

    n_values = (size_a - header_size) // 2
    
    mmap_a = np.memmap(file_a, dtype='<i2', mode='r', offset=header_size)
    mmap_b = np.memmap(file_b, dtype='<i2', mode='r', offset=header_size)
    
    total_mismatches = 0
    max_raw_diff = 0
    
    chunk_len = CHUNK_SIZE_BYTES // 2
    
    for i in range(0, n_values, chunk_len):
        end = min(i + chunk_len, n_values)
        
        chunk_a = mmap_a[i:end]
        chunk_b = mmap_b[i:end]
        
        diff = np.abs(chunk_a - chunk_b)
        
        local_max = np.max(diff)
        if local_max == 0:
            continue
            
        max_raw_diff = max(max_raw_diff, local_max)
        count_diff = np.count_nonzero(diff)
        total_mismatches += count_diff
        
        if writer and count_diff > 0:
            error_indices_local = np.where(diff > 0)[0]
            
            for local_idx in error_indices_local:
                val_a = chunk_a[local_idx]
                val_b = chunk_b[local_idx]
                d = diff[local_idx]
                flat_idx = i + local_idx
                ch_idx, sample_idx = map_flat_index_to_channel(flat_idx, layout, layout_sum)        
                writer.writerow([ch_idx, sample_idx, val_a, val_b, d])

    del mmap_a
    del mmap_b
    
    match_percent = ((n_values - total_mismatches) / n_values) * 100.0
    return match_percent, total_mismatches, max_raw_diff

def plot_diff_heatmap(files, diff_matrix, output_folder):
    n = len(files)
    file_names = [os.path.basename(f).replace('.edf', '') for f in files]
    
    fig_size = max(10, n * 0.9)
    plt.figure(figsize=(fig_size, fig_size))
    
    masked_data = np.ma.masked_where(diff_matrix < 0, diff_matrix)
    
    cmap = plt.cm.Reds
    cmap.set_bad(color='gray')
    
    plt.imshow(masked_data, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Diff Count')
    
    ticks = np.arange(n)
    plt.xticks(ticks, file_names, rotation=90, fontsize=9)
    plt.yticks(ticks, file_names, fontsize=9)
    
    plt.title("Difference Matrix", fontsize=14, pad=20)
    
    thresh = np.max(diff_matrix) / 2 if np.max(diff_matrix) > 0 else 1
    
    for i in range(n):
        for j in range(n):
            val = int(diff_matrix[i, j])
            
            if val == -1:
                text_val = "ERR"
                color = "white"
                fw = 'normal'
            elif val == 0:
                text_val = "MATCH" if i != j else ""
                color = "green"
                fw = 'bold'
            else:
                if val > 1000000: text_val = f"{val/1000000:.1f}M"
                elif val > 1000: text_val = f"{val/1000:.0f}k"
                else: text_val = str(val)
                
                color = "white" if val > thresh else "black"
                fw = 'normal'
            
            if i != j:
                plt.text(j, i, text_val, ha="center", va="center", color=color, fontweight=fw, fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(output_folder, "heatmap_diff_counts.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

def compare_files():
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    if GENERATE_CSV or GENERATE_PLOTS:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    search_path = os.path.join(DATA_FOLDER, "*.edf")
    files = sorted(glob.glob(search_path))
    n_files = len(files)
    
    if n_files < 2:
        print("Too few files for comparison.")
        return
    
    print(f"Found {n_files} EDF files.")
    
    diff_matrix = np.zeros((n_files, n_files))
    file_map = {f: i for i, f in enumerate(files)}

    print("-" * 100)
    print(f"{'FILE A':<25} | {'FILE B':<25} | {'MATCH %':<10} | {'DIFF COUNT':<12} | {'RAW MAX DIFF':<12}")
    print("-" * 100)

    for file_a_path, file_b_path in itertools.combinations(files, 2):
        name_a = os.path.basename(file_a_path)
        name_b = os.path.basename(file_b_path)
        
        idx_a = file_map[file_a_path]
        idx_b = file_map[file_b_path]
        
        csv_file = None
        writer = None
        
        try:
            f_a = pyedflib.EdfReader(file_a_path)
            f_b = pyedflib.EdfReader(file_b_path)
            
            n_chan_a = f_a.signals_in_file
            n_chan_b = f_b.signals_in_file
            
            if n_chan_a != n_chan_b:
                print(f"{name_a:<25} | {name_b:<25} | {'0.00%':<10} | {'-':<12} | {'-':<12}")
                diff_matrix[idx_a, idx_b] = -1 # Error flag
                diff_matrix[idx_b, idx_a] = -1
                f_a.close(); f_b.close()
                continue
                
            phys_min_a = f_a.getPhysicalMinimum(0)
            phys_min_b = f_b.getPhysicalMinimum(0)
            
            f_a.close()
            f_b.close()
            
            if phys_min_a != phys_min_b:
                 print(f"{name_a:<25} | {name_b:<25} | {'???':<10} | {'-':<12} | {'-':<12}")
                 diff_matrix[idx_a, idx_b] = -1
                 diff_matrix[idx_b, idx_a] = -1
                 continue

            if GENERATE_CSV:
                csv_path = os.path.join(OUTPUT_FOLDER, f"diff_{name_a}_vs_{name_b}.csv")
                csv_file = open(csv_path, 'w', newline='')
                writer = csv.writer(csv_file)
                writer.writerow(['Channel', 'Sample Index', 'Value A (Raw)', 'Value B (Raw)', 'Diff'])

            match_pct, mismatches, max_diff = compare_raw_binary(
                file_a_path, file_b_path, n_chan_a, writer=writer, file_name_pair=(name_a, name_b)
            )
            
            if not isinstance(max_diff, str):
                diff_matrix[idx_a, idx_b] = mismatches
                diff_matrix[idx_b, idx_a] = mismatches
            else:
                diff_matrix[idx_a, idx_b] = -1
                diff_matrix[idx_b, idx_a] = -1
            
            if csv_file:
                csv_file.close()
                if mismatches == 0:
                    os.remove(csv_path)

            if isinstance(max_diff, str):
                print(f"{name_a:<25} | {name_b:<25} | {'0.00%':<10} | {'-':<12} | {'-':<12} | {max_diff}")
                continue

            print(f"{name_a:<25} | {name_b:<25} | {match_pct:9.2f}% | {mismatches:<12} | {max_diff:<12}")
            
        except Exception as e:
            if csv_file: csv_file.close()
            diff_matrix[idx_a, idx_b] = -1
            diff_matrix[idx_b, idx_a] = -1
            print(f"Error comparing {name_a} and {name_b}: {e}")

    if GENERATE_PLOTS:
        plot_diff_heatmap(files, diff_matrix, OUTPUT_FOLDER)

if __name__ == "__main__":
    compare_files()