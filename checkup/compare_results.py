import os
import glob
import itertools
import numpy as np
import csv
import shutil
import matplotlib.pyplot as plt

GENERATE_CSV = False
GENERATE_PLOTS = False
INCLUDE_MATCHING_ROWS = False
DATA_FOLDER = "EegLinearFilter/out"
OUTPUT_FOLDER = "checkup/result_comparison"
TOLERANCE = 1e-5

def compare_files():
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)

    if GENERATE_CSV or GENERATE_PLOTS:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    search_path = os.path.join(DATA_FOLDER, "*.txt")
    files = sorted(glob.glob(search_path))
    if len(files) < 2:
        print(f"Found too few files ({len(files)}). Need at least 2 to compare.")
        return
    print(f"Found {len(files)} files. Loading data...")
   
    data_cache = {}
    for f in files:
        try:
            data_cache[f] = np.loadtxt(f)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            return

    print("-" * 110)
    print(f"{'FILE A':<30} | {'FILE B':<30} | {'MATCH %':<10} | {'DIFF COUNT':<12} | {'MAX DIFF':<15}")
    print("-" * 110)

    for file_a, file_b in itertools.combinations(files, 2):
        vec_a = data_cache[file_a]
        vec_b = data_cache[file_b]
       
        name_a = os.path.basename(file_a)
        name_b = os.path.basename(file_b)
        
        if vec_a.shape != vec_b.shape:
            print(f"{name_a:<30} | {name_b:<30} | DIFFERENT LENGTHS!")
            continue

        diff = np.abs(vec_a - vec_b)
        diff_flat = diff.flatten()
        
        matches = np.sum(diff < TOLERANCE)
        total_elements = vec_a.size
        mismatches = total_elements - matches
        similarity_percent = (matches / total_elements) * 100.0
        max_diff = np.max(diff)

        print(f"{name_a:<30} | {name_b:<30} | {similarity_percent:9.2f}% | {mismatches:<12} | {max_diff:.6f}")
        
        if mismatches > 0:
            if GENERATE_PLOTS:
                plot_path = os.path.join(OUTPUT_FOLDER, f"plot_{name_a}_vs_{name_b}.pdf")
                
                errors_to_plot = diff_flat[diff_flat >= TOLERANCE]
                
                if errors_to_plot.size > 0:
                    plt.figure(figsize=(10, 6))
                    
                    plt.hist(errors_to_plot, bins=50, color='red', alpha=0.7, log=True)
                    
                    plt.title(f"Distribution of Errors > Tolerance ({TOLERANCE})\n{name_a} vs {name_b}")
                    plt.xlabel("Absolute Difference Magnitude")
                    plt.ylabel("Count (Log Scale)")
                    plt.grid(True, which="both", ls="-", alpha=0.5)
                    
                    plt.tight_layout()
                    plt.savefig(plot_path)
                    plt.close()

            if GENERATE_CSV:
                vec_a_flat = vec_a.flatten()
                vec_b_flat = vec_b.flatten()
                
                csv_path = os.path.join(OUTPUT_FOLDER, f"comparison_{name_a}_vs_{name_b}.csv")
                
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Index', 'Value A', 'Value B', 'Difference'])
                    
                    if INCLUDE_MATCHING_ROWS:
                        indices_to_process = range(total_elements)
                    else:
                        indices_to_process = np.where(diff_flat >= TOLERANCE)[0]

                    for idx in indices_to_process:
                        current_diff = diff_flat[idx]
                        
                        if current_diff >= TOLERANCE:
                            writer.writerow([idx, vec_a_flat[idx], vec_b_flat[idx], current_diff])
                        else:
                            writer.writerow([idx, "-", "-", "-"])

if __name__ == "__main__":
    compare_files()