import os
import glob
import itertools
import numpy as np
import shutil
import csv
import pyedflib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

EXPORT_DETAILED_CSV = False
VISUALIZE_ERRORS = False
DATA_FOLDER = "EegLinearFilter/out"
OUTPUT_ROOT = "python/results_similarity"
FOLDER_CSV = os.path.join(OUTPUT_ROOT, "csv_tables")
FOLDER_PLOTS = os.path.join(OUTPUT_ROOT, "plots")

CHUNK_SIZE_BYTES = 50 * 1024 * 1024 
MAX_PLOT_POINTS = 50000
VIZ_FIXED_MAX_DIFF = 1000
ERROR_THRESHOLD = 0

def get_header_size(num_channels):
    return 256 + (num_channels * 256)

def get_edf_layout(filename):
    with pyedflib.EdfReader(filename) as f:
        n_signals = f.signals_in_file
        samples_per_record = []
        for i in range(n_signals):
            samples_per_record.append(f.getNSamples()[i] // f.datarecords_in_file)
    return np.array(samples_per_record, dtype=int)

def build_record_lookup_tables(layout):
    record_len = np.sum(layout)
    channel_map = np.zeros(record_len, dtype=np.int16)
    sample_offset_map = np.zeros(record_len, dtype=np.int32)
    
    cursor = 0
    for ch_idx, n_samples in enumerate(layout):
        channel_map[cursor : cursor + n_samples] = ch_idx
        sample_offset_map[cursor : cursor + n_samples] = np.arange(n_samples)
        cursor += n_samples
        
    return channel_map, sample_offset_map, record_len

def plot_error_distribution(error_data, total_errors, name_a, name_b, save_path):
    if not error_data:
        return

    channels = [e[0] for e in error_data]
    samples = [e[1] for e in error_data]
    diffs = [e[2] for e in error_data]

    plt.figure(figsize=(14, 8))
    
    cmap = plt.cm.jet
    vmin_val = max(1, ERROR_THRESHOLD)
    norm = colors.LogNorm(vmin=vmin_val, vmax=max(vmin_val * 10, VIZ_FIXED_MAX_DIFF), clip=True)

    sc = plt.scatter(samples, channels, c=diffs, cmap=cmap, norm=norm, s=10, marker='|', alpha=0.8, rasterized=True)
    
    cbar = plt.colorbar(sc)
    cbar.set_label('Max Absolute Difference (Log Scale, Fixed)', rotation=270, labelpad=20)

    displayed_count = len(channels)
    
    info_parts = []
    if ERROR_THRESHOLD > 0:
        info_parts.append(f"Threshold > {ERROR_THRESHOLD}")
        
    if displayed_count >= total_errors:
        info_parts.append(f"Displayed: {displayed_count} (All)")
    else:
        info_parts.append(f"Displayed: {displayed_count} (Subsampled) | Total: {total_errors}")

    subtitle = " | ".join(info_parts)
    title_text = f"{name_a} vs {name_b}\n{subtitle}"
    
    plt.title(title_text, fontsize=12)
    plt.xlabel("Sample Index (Time)", fontsize=10)
    plt.ylabel("Channel Index", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4)
    
    if len(channels) > 0:
        max_ch = max(channels)
        plt.yticks(range(0, max_ch + 2))
        plt.ylim(-0.5, max_ch + 0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def compare_raw_binary_streaming(file_a, file_b, n_channels, writer=None):
    header_size = get_header_size(n_channels)
    size_a = os.path.getsize(file_a)
    
    if os.path.getsize(file_b) != size_a:
        return None, None, f"SIZE MISMATCH", []

    layout = get_edf_layout(file_a)
    ch_lookup, samp_offset_lookup, record_len = build_record_lookup_tables(layout)
    
    n_values = (size_a - header_size) // 2
    
    target_time_bins = max(1, MAX_PLOT_POINTS // n_channels)
    
    max_samples_per_channel = (n_values // record_len) * np.max(layout)
    samples_per_bin_x = max(1, int(max_samples_per_channel / target_time_bins))
    
    viz_bins = {} 

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
        if local_max <= ERROR_THRESHOLD:
            continue
            
        max_raw_diff = max(max_raw_diff, local_max)
        
        error_indices_local = np.where(diff > ERROR_THRESHOLD)[0]
        count_diff = len(error_indices_local)
        total_mismatches += count_diff
        
        if count_diff > 0:
            global_indices = i + error_indices_local
            record_indices = global_indices // record_len
            offsets_in_record = global_indices % record_len
            
            channels = ch_lookup[offsets_in_record]
            local_sample_offsets = samp_offset_lookup[offsets_in_record]
            samples_per_channel = layout[channels]
            final_sample_indices = (record_indices * samples_per_channel) + local_sample_offsets
            
            vals_a = chunk_a[error_indices_local]
            vals_b = chunk_b[error_indices_local]
            diffs = diff[error_indices_local]

            if writer:
                rows = zip(channels, final_sample_indices, vals_a, vals_b, diffs)
                writer.writerows(rows)
            
            if VISUALIZE_ERRORS:
                bin_indices = final_sample_indices // samples_per_bin_x
                
                sort_order = np.lexsort((diffs, channels, bin_indices))
                
                sorted_bins = bin_indices[sort_order]
                sorted_channels = channels[sort_order]
                sorted_diffs = diffs[sort_order]
                sorted_samples = final_sample_indices[sort_order]
                
                group_ids = sorted_bins.astype(np.int64) * 1000 + sorted_channels
                
                _, unique_indices = np.unique(group_ids, return_index=True)
                
                max_indices = np.append(unique_indices[1:] - 1, len(group_ids) - 1)
                
                best_bins = sorted_bins[max_indices]
                best_channels = sorted_channels[max_indices]
                best_diffs = sorted_diffs[max_indices]
                best_samples = sorted_samples[max_indices]
                
                for b, ch, d, s in zip(best_bins, best_channels, best_diffs, best_samples):
                    key = (b, ch)
                    if key not in viz_bins or d > viz_bins[key][0]:
                        viz_bins[key] = (d, s)

    del mmap_a
    del mmap_b
    
    final_viz_errors = []
    if VISUALIZE_ERRORS:
        for (b, ch), (d, s) in viz_bins.items():
            final_viz_errors.append((ch, s, d))
            
    match_percent = ((n_values - total_mismatches) / n_values) * 100.0
    return match_percent, total_mismatches, max_raw_diff, final_viz_errors

def compare_files():
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)
    
    if EXPORT_DETAILED_CSV:
        os.makedirs(FOLDER_CSV, exist_ok=True)
    if VISUALIZE_ERRORS:
        os.makedirs(FOLDER_PLOTS, exist_ok=True)
    
    search_path = os.path.join(DATA_FOLDER, "*.edf")
    files = sorted(glob.glob(search_path))
    n_files = len(files)
    
    if n_files < 2:
        print("Too few files for comparison.")
        return
    
    print(f"Found {n_files} EDF files.")
    if ERROR_THRESHOLD > 0:
        print(f"Error Threshold: {ERROR_THRESHOLD}")
        
    print("-" * 115)
    print(f"{'FILE A':<25} | {'FILE B':<25} | {'MATCH %':<10} | {'DIFF COUNT':<12} | {'RAW MAX DIFF':<12}")
    print("-" * 115)

    for file_a_path, file_b_path in itertools.combinations(files, 2):
        name_a = os.path.basename(file_a_path).replace('.edf', '')
        name_b = os.path.basename(file_b_path).replace('.edf', '')
        
        csv_file = None
        writer = None
        csv_path = None
        
        try:
            f_a = pyedflib.EdfReader(file_a_path)
            f_b = pyedflib.EdfReader(file_b_path)
            n_chan_a = f_a.signals_in_file
            n_chan_b = f_b.signals_in_file
            
            if n_chan_a != n_chan_b:
                print(f"{name_a:<25} | {name_b:<25} | {'0.00%':<10} | {'CHANNEL MISMATCH':<27}")
                f_a.close(); f_b.close()
                continue
            
            phys_min_a = f_a.getPhysicalMinimum(0)
            phys_min_b = f_b.getPhysicalMinimum(0)
            f_a.close(); f_b.close()
            
            if phys_min_a != phys_min_b:
                 print(f"{name_a:<25} | {name_b:<25} | {'???':<10} | {'PHYS MIN MISMATCH':<27}")
                 continue

            if EXPORT_DETAILED_CSV:
                csv_path = os.path.join(FOLDER_CSV, f"diff_{name_a}_vs_{name_b}.csv")
                csv_file = open(csv_path, 'w', newline='')
                writer = csv.writer(csv_file)
                writer.writerow(['Channel', 'Sample Index', 'Value A (Raw)', 'Value B (Raw)', 'Diff'])

            match_pct, mismatches, max_diff, viz_data = compare_raw_binary_streaming(
                file_a_path, file_b_path, n_chan_a, writer=writer
            )
            
            if csv_file:
                csv_file.close()
                if mismatches == 0 and csv_path:
                    os.remove(csv_path)

            if isinstance(max_diff, str):
                print(f"{name_a:<25} | {name_b:<25} | {'0.00%':<10} | {'-':<12} | {max_diff}")
            else:
                print(f"{name_a:<25} | {name_b:<25} | {match_pct:9.2f}% | {mismatches:<12} | {max_diff:<12}")

            if VISUALIZE_ERRORS and mismatches > 0 and not isinstance(max_diff, str):
                plot_path = os.path.join(FOLDER_PLOTS, f"dist_{name_a}_vs_{name_b}.pdf")
                plot_error_distribution(viz_data, mismatches, name_a, name_b, plot_path)
            
        except Exception as e:
            if csv_file: csv_file.close()
            print(f"Error comparing {name_a} and {name_b}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    compare_files()