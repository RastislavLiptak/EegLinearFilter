import os
import glob
import itertools
import numpy as np
import csv
import shutil

INCLUDE_MATCHING_ROWS = False
DATA_FOLDER = "EegLinearFilter/out"
OUTPUT_FOLDER = "checkup/result_comparison"
TOLERANCE = 1e-5

def compare_files():
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
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
        matches = np.sum(diff < TOLERANCE)
        total_elements = vec_a.size
        mismatches = total_elements - matches
        similarity_percent = (matches / total_elements) * 100.0
        max_diff = np.max(diff)

        print(f"{name_a:<30} | {name_b:<30} | {similarity_percent:9.2f}% | {mismatches:<12} | {max_diff:.6f}")
        
        if mismatches > 0:
            vec_a_flat = vec_a.flatten()
            vec_b_flat = vec_b.flatten()
            diff_flat = diff.flatten()
            
            csv_path = os.path.join(OUTPUT_FOLDER, f"comparison_{name_a}_vs_{name_b}.csv")
            
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Index', 'Value A', 'Value B', 'Difference'])
                
                els_to_save = total_elements
                if not INCLUDE_MATCHING_ROWS:
                    els_to_save = mismatches

                for idx in range(els_to_save):
                    current_diff = diff_flat[idx]
                    
                    if current_diff >= TOLERANCE:
                        writer.writerow([idx, vec_a_flat[idx], vec_b_flat[idx], current_diff])
                    else:
                        writer.writerow([idx, "-", "-", "-"])

if __name__ == "__main__":
    compare_files()