import os
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import math
import statistics

# --- CONFIGURATION ---
LOG_DATA_PATH = "EegLinearFilter/logs/benchmark_results.csv"
OUTPUT_DIR = "python/benchmark_results/"

ALL_METRICS = [
    "TotalTimeSec",
    "ComputeTimeSec",
    "OverheadTimeSec", 
    "CpuMemOpsSec", 
    "GpuMemOpsSec"
]

COMPARISON_METRICS = [
    "TotalTimeSec",
    "ComputeTimeSec"
]

ARCH_COLORS = {
    'Sequential': 'tab:blue',
    'Parallel': 'tab:green',
    'GPU': 'tab:red',
    'Other': 'gray'
}

# --- DATA PROCESSING ---

def load_data(filepath):
    def parse_val(val, target_type, default):
        try:
            return target_type(val)
        except (ValueError, TypeError):
            return default

    benchmark_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'metrics': defaultdict(list), 'meta': {}})))
    
    try:
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            
            for row in reader:
                filename = row['Filename']
                mode = row['Mode']
                
                try:
                    kernel_radius = int(row['KernelRadius'])
                    iteration = int(row['Iteration'])
                    output_elements = int(row.get('OutputElements', 0))
                except ValueError:
                    kernel_radius = row['KernelRadius']
                    output_elements = 0

                current_config = benchmark_data[filename][kernel_radius][mode]
                
                if output_elements > 0:
                    current_config['meta']['OutputElements'] = output_elements

                for metric in ALL_METRICS:
                    val = parse_val(row.get(metric), float, 0.0)
                    metric_list = current_config['metrics'][metric]
                    
                    if iteration == 1 or not metric_list:
                        metric_list.append([])
                    metric_list[-1].append(val)
                        
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return {}
    except Exception as e:
        print(f"File load failed: {e}")
        return {}

    return benchmark_data

def aggregate_data_by_radius(original_data):
    aggregated = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'metrics': defaultdict(list), 'meta': {}})))
    
    for filename, radiuses in original_data.items():
        for radius, modes in radiuses.items():
            for mode, data in modes.items():
                output_elements = data['meta'].get('OutputElements', 0)
                if output_elements == 0:
                    continue
                
                target = aggregated[radius][output_elements][mode]
                target['meta']['OutputElements'] = output_elements
                
                for metric, runs in data['metrics'].items():
                    for run in runs:
                        target['metrics'][metric].append(run)
                        
    return aggregated

# --- CALCULATION ---

def calculate_median_trajectory(runs):
    if not runs:
        return []
    max_len = max(len(r) for r in runs)
    robust_values = []
    for i in range(max_len):
        vals_at_i = [r[i] for r in runs if i < len(r)]
        robust_values.append(statistics.median(vals_at_i) if vals_at_i else 0)
    return robust_values

def get_arch_style(mode_name):
    if "GPU" in mode_name: return ARCH_COLORS['GPU'], 'GPU'
    if "PAR" in mode_name: return ARCH_COLORS['Parallel'], 'Parallel'
    if "SEQ" in mode_name: return ARCH_COLORS['Sequential'], 'Sequential'
    return ARCH_COLORS['Other'], 'Other'

def get_active_metrics(modes_data, metric_list):
    active = []
    for metric in metric_list:
        has_data = False
        for _, data in modes_data.items():
            runs = data['metrics'][metric]
            if sum(val for run in runs for val in run) > 0:
                has_data = True
                break
        if has_data:
            active.append(metric)
    return active

def calculate_gflops(median_compute_time, output_elements, radius):
    if median_compute_time <= 1e-9 or output_elements == 0:
        return 0.0
    kernel_size = 2 * radius + 1
    total_operations = output_elements * kernel_size * 2.0
    gflops = (total_operations / median_compute_time) / 1e9
    return gflops

def format_large_number(num):
    if num >= 1e9:
        return f"{num / 1e9:.1f}G"
    elif num >= 1e6:
        return f"{num / 1e6:.1f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.1f}k"
    return str(num)

# --- PLOTTING FUNCTIONS ---

def plot_single_mode_detail(filename, radius, mode, metric_data, output_dir):
    active_metrics = []
    for metric in ALL_METRICS:
        runs = metric_data[metric]
        if sum(val for run in runs for val in run) > 0:
            active_metrics.append(metric)
            
    if not active_metrics: return

    num_metrics = len(active_metrics)
    cols = 2
    rows = math.ceil(num_metrics / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    fig.suptitle(f"{filename} (R={radius}) - {mode}", fontsize=16, y=0.98)
    
    axes_flat = [axes] if num_metrics == 1 else axes.flatten()
    
    detail_handles = []
    detail_labels = []

    for i, metric in enumerate(active_metrics):
        ax = axes_flat[i]
        runs = metric_data[metric]
        
        if runs:
            for run_idx, run_values in enumerate(runs):
                l, = ax.plot(run_values, marker='.', linestyle='-', 
                        linewidth=1.5, markersize=3, alpha=0.25, 
                        label=f"Run {run_idx + 1}")
                if i == 0 and len(runs) <= 5 and f"Run {run_idx + 1}" not in detail_labels:
                    detail_handles.append(l)
                    detail_labels.append(f"Run {run_idx + 1}")
            
            median_values = calculate_median_trajectory(runs)
            l_avg, = ax.plot(median_values, marker='o', linestyle='-', 
                    linewidth=2.5, color='black', markersize=5, 
                    label='Median', zorder=10)
            
            if i == 0:
                detail_handles.insert(0, l_avg)
                detail_labels.insert(0, 'Median')

            ax.set_title(metric)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Time (s)')
            ax.grid(True, linestyle='--', alpha=0.6)

    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    if detail_handles:
        ncols = 1 if len(detail_handles) == 1 else min(len(detail_handles), 6)
        if len(detail_handles) > 7:
            fig.legend([detail_handles[0]], ['Median'], loc='upper center', 
                bbox_to_anchor=(0.5, 0.94), ncol=1, frameon=False)
        else:
            fig.legend(detail_handles, detail_labels, loc='upper center', 
                bbox_to_anchor=(0.5, 0.94), ncol=ncols, frameon=False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.savefig(os.path.join(output_dir, f"{mode}.pdf"), dpi=150)
    plt.close(fig)

def plot_combined_summary(filename, radius, modes_data, output_dir):
    active_metrics = get_active_metrics(modes_data, COMPARISON_METRICS)
    if not active_metrics: return

    num_metrics = len(active_metrics)
    fig, axes = plt.subplots(2, num_metrics, figsize=(7 * num_metrics, 12))
    fig.suptitle(f"Combined Benchmark Summary: {filename} (R={radius})", fontsize=20, fontweight='bold', y=0.98)
    
    if num_metrics == 1:
        axes_detailed = [axes[0]]
        axes_grouped = [axes[1]]
    else:
        axes_detailed = axes[0]
        axes_grouped = axes[1]

    lines_handles_detailed = []
    lines_labels_detailed = []
    
    for i, metric in enumerate(active_metrics):
        ax = axes_detailed[i]
        for mode, data in modes_data.items():
            runs = data['metrics'][metric]
            if not runs: continue
            
            median_values = calculate_median_trajectory(runs)
            if sum(median_values) > 0:
                line, = ax.plot(median_values, marker='o', linestyle='-', linewidth=2, markersize=5, label=mode)
                if i == 0:
                    lines_handles_detailed.append(line)
                    lines_labels_detailed.append(mode)
        
        ax.set_title(f"Detailed: {metric}", fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (s)')
        ax.grid(True, linestyle='--', alpha=0.6)

    used_groups = set()
    for i, metric in enumerate(active_metrics):
        ax = axes_grouped[i]
        for mode, data in modes_data.items():
            runs = data['metrics'][metric]
            if not runs: continue
            median_values = calculate_median_trajectory(runs)
            if sum(median_values) > 0:
                color, group_name = get_arch_style(mode)
                used_groups.add(group_name)
                ax.plot(median_values, marker='o', linestyle='-', linewidth=2, markersize=5, color=color, alpha=0.7)
        ax.set_title(f"Grouped: {metric}", fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (s)')
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.02, 1, 0.88])
    plt.subplots_adjust(hspace=0.50) 

    if lines_handles_detailed:
        num_cols = min(len(lines_labels_detailed), 4)
        fig.legend(lines_handles_detailed, lines_labels_detailed, 
                   loc='lower center', bbox_to_anchor=(0.5, 0.86), 
                   ncol=num_cols, frameon=False, fontsize='medium')

    custom_lines = []
    custom_labels = []
    preferred_order = ['Sequential', 'Parallel', 'GPU', 'Other']
    for group in preferred_order:
        if group in used_groups:
            custom_lines.append(Line2D([0], [0], color=ARCH_COLORS[group], lw=2))
            custom_labels.append(group)
            
    if custom_lines:
        fig.legend(custom_lines, custom_labels, 
                   loc='lower center', bbox_to_anchor=(0.5, 0.405),
                   ncol=len(custom_labels), frameon=False, fontsize='medium')

    plt.savefig(os.path.join(output_dir, "combined_summary.pdf"), dpi=150)
    plt.close(fig)

def create_radius_scaling_time_pdf(filename, radiuses_data, output_dir):
    unique_radiuses = sorted(radiuses_data.keys())
    if len(unique_radiuses) < 2: return

    scaling_data = defaultdict(lambda: defaultdict(list))
    
    for radius in unique_radiuses:
        modes = radiuses_data[radius]
        for mode, data in modes.items():
            for metric in COMPARISON_METRICS:
                runs = data['metrics'][metric]
                all_vals = [val for run in runs for val in run]
                if all_vals:
                    med_val = statistics.median(all_vals)
                    scaling_data[metric][mode].append((radius, med_val))

    if not scaling_data:
        return

    num_metrics = len(COMPARISON_METRICS)
    fig, axes = plt.subplots(2, num_metrics, figsize=(7 * num_metrics, 12))
    fig.suptitle(f"Radius Scaling Analysis: {filename}", fontsize=20, fontweight='bold', y=0.98)
    
    if num_metrics == 1:
        axes_detailed = [axes[0]]
        axes_grouped = [axes[1]]
    else:
        axes_detailed = axes[0]
        axes_grouped = axes[1]

    lines_handles_detailed = []
    lines_labels_detailed = []
    
    for i, metric in enumerate(COMPARISON_METRICS):
        ax = axes_detailed[i]
        
        for mode, points in scaling_data[metric].items():
            if not points: continue
            
            points.sort(key=lambda x: x[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            
            if ys:
                line, = ax.plot(xs, ys, marker='o', linestyle='-', linewidth=2, markersize=5, label=mode)
                if i == 0:
                    lines_handles_detailed.append(line)
                    lines_labels_detailed.append(mode)
        
        ax.set_title(f"Detailed: {metric}", fontweight='bold')
        ax.set_xlabel('Kernel Radius')
        ax.set_ylabel('Time (s) [Median]')
        ax.grid(True, linestyle='--', alpha=0.6)
        if len(unique_radiuses) < 20:
            ax.set_xticks(unique_radiuses)

    used_groups = set()
    
    for i, metric in enumerate(COMPARISON_METRICS):
        ax = axes_grouped[i]
        
        for mode, points in scaling_data[metric].items():
            if not points: continue
            
            points.sort(key=lambda x: x[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            
            if ys:
                color, group_name = get_arch_style(mode)
                used_groups.add(group_name)
                ax.plot(xs, ys, marker='o', linestyle='-', linewidth=2, markersize=5, color=color, alpha=0.7)
        
        ax.set_title(f"Grouped: {metric}", fontweight='bold')
        ax.set_xlabel('Kernel Radius')
        ax.set_ylabel('Time (s) [Median]')
        ax.grid(True, linestyle='--', alpha=0.6)
        if len(unique_radiuses) < 20:
            ax.set_xticks(unique_radiuses)

    plt.tight_layout(rect=[0, 0.02, 1, 0.88])
    plt.subplots_adjust(hspace=0.50) 

    if lines_handles_detailed:
        num_cols = min(len(lines_labels_detailed), 4)
        fig.legend(lines_handles_detailed, lines_labels_detailed, 
                   loc='lower center', bbox_to_anchor=(0.5, 0.86), 
                   ncol=num_cols, frameon=False, fontsize='medium')

    custom_lines = []
    custom_labels = []
    preferred_order = ['Sequential', 'Parallel', 'GPU', 'Other']
    for group in preferred_order:
        if group in used_groups:
            custom_lines.append(Line2D([0], [0], color=ARCH_COLORS[group], lw=2))
            custom_labels.append(group)
            
    if custom_lines:
        fig.legend(custom_lines, custom_labels, 
                   loc='lower center', bbox_to_anchor=(0.5, 0.405),
                   ncol=len(custom_labels), frameon=False, fontsize='medium')

    output_path = os.path.join(output_dir, "scaling_analysis_time.pdf")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def create_radius_scaling_gflops_pdf(filename, radiuses_data, output_dir):
    unique_radiuses = sorted(radiuses_data.keys())
    if len(unique_radiuses) < 2: return

    scaling_data = defaultdict(lambda: defaultdict(list))
    
    target_metrics = ["ComputeTimeSec"]
    
    for radius in unique_radiuses:
        modes = radiuses_data[radius]
        for mode, data in modes.items():
            output_elements = data['meta'].get('OutputElements', 0)
            
            for metric in target_metrics:
                runs = data['metrics'][metric]
                all_vals = [val for run in runs for val in run]
                
                if all_vals:
                    med_time = statistics.median(all_vals)
                    gflops_val = calculate_gflops(med_time, output_elements, radius)
                    scaling_data[metric][mode].append((radius, gflops_val))

    if not scaling_data:
        return

    num_metrics = len(target_metrics)
    
    fig, axes = plt.subplots(2, num_metrics, figsize=(12, 14)) 
    
    fig.suptitle(f"GFLOPS Scaling Analysis: {filename}", fontsize=20, fontweight='bold', y=0.98)
    
    if num_metrics == 1:
        axes_detailed = [axes[0]]
        axes_grouped = [axes[1]]
    else:
        axes_detailed = axes[0]
        axes_grouped = axes[1]

    lines_handles_detailed = []
    lines_labels_detailed = []
    
    for i, metric in enumerate(target_metrics):
        ax = axes_detailed[i]
        
        for mode, points in scaling_data[metric].items():
            if not points: continue
            
            points.sort(key=lambda x: x[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            
            if ys:
                line, = ax.plot(xs, ys, marker='o', linestyle='-', linewidth=2, markersize=5, label=mode)
                if i == 0:
                    lines_handles_detailed.append(line)
                    lines_labels_detailed.append(mode)
        
        ax.set_title(f"Detailed: {metric}", fontweight='bold')
        ax.set_xlabel('Kernel Radius')
        ax.set_ylabel('Performance (GFLOPS)')
        ax.grid(True, linestyle='--', alpha=0.6)
        if len(unique_radiuses) < 20:
            ax.set_xticks(unique_radiuses)

    used_groups = set()
    
    for i, metric in enumerate(target_metrics):
        ax = axes_grouped[i]
        
        for mode, points in scaling_data[metric].items():
            if not points: continue
            
            points.sort(key=lambda x: x[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            
            if ys:
                color, group_name = get_arch_style(mode)
                used_groups.add(group_name)
                ax.plot(xs, ys, marker='o', linestyle='-', linewidth=2, markersize=5, color=color, alpha=0.7)
        
        ax.set_title(f"Grouped: {metric}", fontweight='bold')
        ax.set_xlabel('Kernel Radius')
        ax.set_ylabel('Performance (GFLOPS)')
        ax.grid(True, linestyle='--', alpha=0.6)
        if len(unique_radiuses) < 20:
            ax.set_xticks(unique_radiuses)

    plt.tight_layout(rect=[0, 0.02, 1, 0.88])
    plt.subplots_adjust(hspace=0.40) 

    if lines_handles_detailed:
        num_cols = min(len(lines_labels_detailed), 4)
        fig.legend(lines_handles_detailed, lines_labels_detailed, 
                   loc='lower center', bbox_to_anchor=(0.5, 0.86), 
                   ncol=num_cols, frameon=False, fontsize='medium')

    custom_lines = []
    custom_labels = []
    preferred_order = ['Sequential', 'Parallel', 'GPU', 'Other']
    for group in preferred_order:
        if group in used_groups:
            custom_lines.append(Line2D([0], [0], color=ARCH_COLORS[group], lw=2))
            custom_labels.append(group)
            
    if custom_lines:
        fig.legend(custom_lines, custom_labels, 
                   loc='lower center', bbox_to_anchor=(0.5, 0.415),
                   ncol=len(custom_labels), frameon=False, fontsize='medium')

    output_path = os.path.join(output_dir, "scaling_analysis_gflops.pdf")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def create_data_scaling_time_pdf(radius, size_data, output_dir):
    unique_sizes = sorted(size_data.keys())
    if len(unique_sizes) < 2: return

    scaling_data = defaultdict(lambda: defaultdict(list))
    
    for size in unique_sizes:
        modes = size_data[size]
        for mode, data in modes.items():
            for metric in COMPARISON_METRICS:
                runs = data['metrics'][metric]
                all_vals = [val for run in runs for val in run]
                if all_vals:
                    med_val = statistics.median(all_vals)
                    scaling_data[metric][mode].append((size, med_val))

    if not scaling_data:
        return

    num_metrics = len(COMPARISON_METRICS)
    fig, axes = plt.subplots(2, num_metrics, figsize=(7 * num_metrics, 12))
    fig.suptitle(f"Data Size Scaling Analysis (Fixed R={radius})", fontsize=20, fontweight='bold', y=0.98)
    
    if num_metrics == 1:
        axes_detailed = [axes[0]]
        axes_grouped = [axes[1]]
    else:
        axes_detailed = axes[0]
        axes_grouped = axes[1]

    lines_handles_detailed = []
    lines_labels_detailed = []
    
    for i, metric in enumerate(COMPARISON_METRICS):
        ax = axes_detailed[i]
        for mode, points in scaling_data[metric].items():
            if not points: continue
            points.sort(key=lambda x: x[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            if ys:
                line, = ax.plot(xs, ys, marker='o', linestyle='-', linewidth=2, markersize=5, label=mode)
                if i == 0:
                    lines_handles_detailed.append(line)
                    lines_labels_detailed.append(mode)
        
        ax.set_title(f"Detailed: {metric}", fontweight='bold')
        ax.set_xlabel('Output Elements')
        ax.set_ylabel('Time (s) [Median]')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.ticklabel_format(style='plain', axis='x')

    used_groups = set()
    for i, metric in enumerate(COMPARISON_METRICS):
        ax = axes_grouped[i]
        for mode, points in scaling_data[metric].items():
            if not points: continue
            points.sort(key=lambda x: x[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            if ys:
                color, group_name = get_arch_style(mode)
                used_groups.add(group_name)
                ax.plot(xs, ys, marker='o', linestyle='-', linewidth=2, markersize=5, color=color, alpha=0.7)
        
        ax.set_title(f"Grouped: {metric}", fontweight='bold')
        ax.set_xlabel('Output Elements')
        ax.set_ylabel('Time (s) [Median]')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.ticklabel_format(style='plain', axis='x')

    plt.tight_layout(rect=[0, 0.02, 1, 0.88])
    plt.subplots_adjust(hspace=0.50) 
    
    if lines_handles_detailed:
        num_cols = min(len(lines_labels_detailed), 4)
        fig.legend(lines_handles_detailed, lines_labels_detailed, loc='lower center', bbox_to_anchor=(0.5, 0.86), ncol=num_cols, frameon=False, fontsize='medium')

    custom_lines = []
    custom_labels = []
    preferred_order = ['Sequential', 'Parallel', 'GPU', 'Other']
    for group in preferred_order:
        if group in used_groups:
            custom_lines.append(Line2D([0], [0], color=ARCH_COLORS[group], lw=2))
            custom_labels.append(group)
            
    if custom_lines:
        fig.legend(custom_lines, custom_labels, loc='lower center', bbox_to_anchor=(0.5, 0.405), ncol=len(custom_labels), frameon=False, fontsize='medium')

    output_path = os.path.join(output_dir, "data_scaling_time.pdf")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def create_data_scaling_gflops_pdf(radius, size_data, output_dir):
    unique_sizes = sorted(size_data.keys())
    if len(unique_sizes) < 2: return

    scaling_data = defaultdict(lambda: defaultdict(list))
    
    target_metrics = ["ComputeTimeSec"]
    
    for size in unique_sizes:
        modes = size_data[size]
        for mode, data in modes.items():
            for metric in target_metrics:
                runs = data['metrics'][metric]
                all_vals = [val for run in runs for val in run]
                if all_vals:
                    med_time = statistics.median(all_vals)
                    gflops_val = calculate_gflops(med_time, size, radius)
                    scaling_data[metric][mode].append((size, gflops_val))

    if not scaling_data:
        return

    num_metrics = len(target_metrics)
    fig, axes = plt.subplots(2, num_metrics, figsize=(12, 14)) 
    fig.suptitle(f"Data GFLOPS Analysis (Fixed R={radius})", fontsize=20, fontweight='bold', y=0.98)
    
    if num_metrics == 1:
        axes_detailed = [axes[0]]
        axes_grouped = [axes[1]]
    else:
        axes_detailed = axes[0]
        axes_grouped = axes[1]

    lines_handles_detailed = []
    lines_labels_detailed = []
    
    for i, metric in enumerate(target_metrics):
        ax = axes_detailed[i]
        for mode, points in scaling_data[metric].items():
            if not points: continue
            points.sort(key=lambda x: x[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            if ys:
                line, = ax.plot(xs, ys, marker='o', linestyle='-', linewidth=2, markersize=5, label=mode)
                if i == 0:
                    lines_handles_detailed.append(line)
                    lines_labels_detailed.append(mode)
        
        ax.set_title(f"Detailed: {metric}", fontweight='bold')
        ax.set_xlabel('Output Elements')
        ax.set_ylabel('Performance (GFLOPS)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.ticklabel_format(style='plain', axis='x')

    used_groups = set()
    for i, metric in enumerate(target_metrics):
        ax = axes_grouped[i]
        for mode, points in scaling_data[metric].items():
            if not points: continue
            points.sort(key=lambda x: x[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            if ys:
                color, group_name = get_arch_style(mode)
                used_groups.add(group_name)
                ax.plot(xs, ys, marker='o', linestyle='-', linewidth=2, markersize=5, color=color, alpha=0.7)
        ax.set_title(f"Grouped: {metric}", fontweight='bold')
        ax.set_xlabel('Output Elements')
        ax.set_ylabel('Performance (GFLOPS)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.ticklabel_format(style='plain', axis='x')

    plt.tight_layout(rect=[0, 0.02, 1, 0.88])
    plt.subplots_adjust(hspace=0.40) 

    if lines_handles_detailed:
        num_cols = min(len(lines_labels_detailed), 4)
        fig.legend(lines_handles_detailed, lines_labels_detailed, loc='lower center', bbox_to_anchor=(0.5, 0.86), ncol=num_cols, frameon=False, fontsize='medium')

    custom_lines = []
    custom_labels = []
    preferred_order = ['Sequential', 'Parallel', 'GPU', 'Other']
    for group in preferred_order:
        if group in used_groups:
            custom_lines.append(Line2D([0], [0], color=ARCH_COLORS[group], lw=2))
            custom_labels.append(group)
            
    if custom_lines:
        fig.legend(custom_lines, custom_labels, loc='lower center', bbox_to_anchor=(0.5, 0.415), ncol=len(custom_labels), frameon=False, fontsize='medium')

    output_path = os.path.join(output_dir, "data_scaling_gflops.pdf")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

# --- TABLE GENERATION ---

def create_summary_table_pdf(filename, radius, modes_data, output_dir):
    table_rows = []
    
    for mode, data in modes_data.items():
        total_runs = data['metrics']['TotalTimeSec']
        compute_runs = data['metrics']['ComputeTimeSec']
        
        all_total_vals = [val for run in total_runs for val in run]
        all_compute_vals = [val for run in compute_runs for val in run]
        
        if not all_total_vals: continue

        total_min = min(all_total_vals)
        total_max = max(all_total_vals)
        total_med = statistics.median(all_total_vals)
        
        if all_compute_vals:
            comp_min = min(all_compute_vals)
            comp_max = max(all_compute_vals)
            comp_med = statistics.median(all_compute_vals)
        else:
            comp_min = comp_max = comp_med = 0.0

        output_elems = data['meta'].get('OutputElements', 0)
        gflops = calculate_gflops(comp_med, output_elems, radius)
        
        table_rows.append({
            'Mode': mode,
            'TotalRange': f"{total_min:.4f} - {total_max:.4f}",
            'TotalMedian': total_med,
            'ComputeRange': f"{comp_min:.4f} - {comp_max:.4f}",
            'ComputeMedian': comp_med,
            'GFLOPS': gflops
        })

    table_rows.sort(key=lambda x: x['GFLOPS'], reverse=True)

    if not table_rows: return

    columns = [
        "Mode", 
        "Total Time (s)\n(Min - Max)", 
        "Total Time (s)\nMedian", 
        "Compute Time (s)\n(Min - Max)", 
        "Compute Time (s)\nMedian", 
        "Performance\n(GFLOPS)"
    ]
    
    cell_text = []
    for row in table_rows:
        cell_text.append([
            row['Mode'],
            row['TotalRange'],
            f"{row['TotalMedian']:.4f}",
            row['ComputeRange'],
            f"{row['ComputeMedian']:.4f}",
            f"{row['GFLOPS']:.2f}"
        ])

    num_rows = len(table_rows) + 1
    calc_height = num_rows * 0.4 + 1.0
    
    fig, ax = plt.subplots(figsize=(14, calc_height))
    ax.axis('off')
    ax.set_title(f"Performance Statistics: {filename} (R={radius})", fontsize=16, fontweight='bold')
    table = ax.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        elif i > 0:
            if i % 2 == 0:
                cell.set_facecolor('#f2f2f2')
    
    plt.savefig(os.path.join(output_dir, "performance_table.pdf"), dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def create_radius_analysis_table_pdf(filename, radiuses_data, output_dir):
    unique_radiuses = sorted(radiuses_data.keys())
    if len(unique_radiuses) < 2: return

    unique_modes = set()
    data_map = defaultdict(dict)
    radius_meta = {} 

    for radius in unique_radiuses:
        modes = radiuses_data[radius]
        for mode, data in modes.items():
            unique_modes.add(mode)
            
            if radius not in radius_meta:
                 output_elems = data['meta'].get('OutputElements', 0)
                 radius_meta[radius] = output_elems
            
            runs = data['metrics']['ComputeTimeSec']
            all_vals = [val for run in runs for val in run]
            med_time = statistics.median(all_vals) if all_vals else 0.0
            
            output_elements = data['meta'].get('OutputElements', 0)
            gflops = calculate_gflops(med_time, output_elements, radius)
            
            data_map[mode][radius] = (med_time, gflops)

    mode_total_times = {}
    for mode in unique_modes:
        total_time = 0.0
        for r in unique_radiuses:
            if r in data_map[mode]:
                total_time += data_map[mode][r][0]
        mode_total_times[mode] = total_time

    sorted_modes = sorted(list(unique_modes), key=lambda m: mode_total_times[m])

    if not sorted_modes or not unique_radiuses:
        return

    col_labels = ["Mode"]
    for r in unique_radiuses:
        out_el = radius_meta.get(r, 0)
        kernel_size = 2 * r + 1
        total_ops = out_el * kernel_size * 2
        ops_str = format_large_number(int(total_ops))
        col_labels.append(f"R={r}\n({ops_str} Ops)")

    cell_text = []
    for mode in sorted_modes:
        row_data = [mode]
        for r in unique_radiuses:
            if r in data_map[mode]:
                time_val, gflops_val = data_map[mode][r]
                cell_str = f"{time_val:.4f} s\n({gflops_val:.2f} GF)"
            else:
                cell_str = "N/A"
            row_data.append(cell_str)
        cell_text.append(row_data)

    width = max(10, len(unique_radiuses) * 2.5 + 2)
    height = len(sorted_modes) * 0.6 + 1.5
    
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis('off')
    ax.set_title(f"Scaling Analysis: {filename}", fontsize=16, fontweight='bold', y=0.98)
    
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center', bbox=[0, 0, 1, 0.90])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
            cell.set_height(0.15)
        else:
            cell.set_height(0.1)
            if i % 2 == 0:
                cell.set_facecolor('#f2f2f2')
            else:
                cell.set_facecolor('#ffffff')

    output_path = os.path.join(output_dir, "scaling_analysis_table.pdf")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def create_data_analysis_table_pdf(radius, size_data, output_dir):
    unique_sizes = sorted(size_data.keys())
    if len(unique_sizes) < 2: return

    unique_modes = set()
    data_map = defaultdict(dict)

    for size in unique_sizes:
        modes = size_data[size]
        for mode, data in modes.items():
            unique_modes.add(mode)
            runs = data['metrics']['ComputeTimeSec']
            all_vals = [val for run in runs for val in run]
            med_time = statistics.median(all_vals) if all_vals else 0.0
            gflops = calculate_gflops(med_time, size, radius)
            data_map[mode][size] = (med_time, gflops)

    mode_total_times = {}
    for mode in unique_modes:
        total_time = 0.0
        for s in unique_sizes:
            if s in data_map[mode]:
                total_time += data_map[mode][s][0]
        mode_total_times[mode] = total_time

    sorted_modes = sorted(list(unique_modes), key=lambda m: mode_total_times[m])

    if not sorted_modes or not unique_sizes:
        return

    col_labels = ["Mode"]
    for s in unique_sizes:
        kernel_size = 2 * radius + 1
        total_ops = s * kernel_size * 2
        ops_str = format_large_number(int(total_ops))
        size_str = format_large_number(s)
        col_labels.append(f"Size={size_str}\n({ops_str} Ops)")

    cell_text = []
    for mode in sorted_modes:
        row_data = [mode]
        for s in unique_sizes:
            if s in data_map[mode]:
                time_val, gflops_val = data_map[mode][s]
                cell_str = f"{time_val:.4f} s\n({gflops_val:.2f} GF)"
            else:
                cell_str = "N/A"
            row_data.append(cell_str)
        cell_text.append(row_data)

    width = max(10, len(unique_sizes) * 2.5 + 2)
    height = len(sorted_modes) * 0.6 + 1.5
    
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis('off')
    ax.set_title(f"Data Scaling Analysis Table (Fixed R={radius})", fontsize=16, fontweight='bold', y=0.98)
    
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center', bbox=[0, 0, 1, 0.90])
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
            cell.set_height(0.15)
        else:
            cell.set_height(0.1)
            if i % 2 == 0:
                cell.set_facecolor('#f2f2f2')
            else:
                cell.set_facecolor('#ffffff')

    output_path = os.path.join(output_dir, "data_scaling_table.pdf")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

# --- MATRIX GENERATION ---

def create_speedup_matrix_pdf(filename, radius, modes_data, output_dir):
    mode_stats = []
    for mode, data in modes_data.items():
        runs = data['metrics']['ComputeTimeSec']
        all_vals = [val for run in runs for val in run]
        if all_vals:
            med_time = statistics.median(all_vals)
            if med_time < 1e-9: med_time = 1e-9
            mode_stats.append((mode, med_time))
    
    if len(mode_stats) < 2:
        return

    mode_stats.sort(key=lambda x: x[1])
    
    sorted_modes = [x[0] for x in mode_stats]
    times = [x[1] for x in mode_stats]
    n = len(sorted_modes)
    
    matrix_data = [[0.0] * n for _ in range(n)]
    
    for r in range(n):
        for c in range(n):
            matrix_data[r][c] = times[c] / times[r]

    fig_size = max(8, n * 1.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    _ = ax.imshow(matrix_data, cmap='RdYlGn', norm=LogNorm(vmin=0.1, vmax=10))
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(sorted_modes, rotation=45, ha="right")
    ax.set_yticklabels(sorted_modes)
    
    ax.set_title(f"Speedup Matrix: {filename} (R={radius})\n(Row is X times faster than Column)", fontsize=16, fontweight='bold', pad=20)

    for i in range(n):
        for j in range(n):
            val = matrix_data[i][j]
            text_color = "black"
            if val > 3 or val < 0.3:
                text_color = "white"
            
            if val >= 100:
                text_val = f"{val:.0f}x"
            elif val >= 10:
                text_val = f"{val:.1f}x"
            else:
                text_val = f"{val:.2f}x"
                
            ax.text(j, i, text_val, ha="center", va="center", color=text_color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speedup_matrix.pdf"), dpi=150, bbox_inches='tight')
    plt.close(fig)

# --- MAIN DRIVER ---

def process_cross_file_analysis(data, output_dir_base):
    radius_grouped_data = aggregate_data_by_radius(data)
    
    cross_analysis_dir = os.path.join(output_dir_base, "cross_dataset_analysis")
    os.makedirs(cross_analysis_dir, exist_ok=True)
    
    for radius, size_data in radius_grouped_data.items():
        if len(size_data) < 2:
            continue

        radius_dir = os.path.join(cross_analysis_dir, str(radius))
        os.makedirs(radius_dir, exist_ok=True)
        
        create_data_scaling_time_pdf(radius, size_data, radius_dir)
        create_data_scaling_gflops_pdf(radius, size_data, radius_dir)
        create_data_analysis_table_pdf(radius, size_data, radius_dir)

def process_benchmarks(data, output_dir_base):
    dataset_specific_root = os.path.join(output_dir_base, "dataset_specific_analysis")
    
    for filename, radiuses in data.items():
        safe_filename = filename.replace('.', '_')
        
        file_level_output_dir = os.path.join(dataset_specific_root, safe_filename)
        os.makedirs(file_level_output_dir, exist_ok=True)

        create_radius_scaling_time_pdf(filename, radiuses, file_level_output_dir)
        create_radius_scaling_gflops_pdf(filename, radiuses, file_level_output_dir)
        create_radius_analysis_table_pdf(filename, radiuses, file_level_output_dir)

        for radius, modes in radiuses.items():
            specific_output_dir = os.path.join(file_level_output_dir, str(radius))
            os.makedirs(specific_output_dir, exist_ok=True)
            
            for mode, config_data in modes.items():
                plot_single_mode_detail(filename, radius, mode, config_data['metrics'], specific_output_dir)

            plot_combined_summary(filename, radius, modes, specific_output_dir)
            create_summary_table_pdf(filename, radius, modes, specific_output_dir)
            create_speedup_matrix_pdf(filename, radius, modes, specific_output_dir)
            
    process_cross_file_analysis(data, output_dir_base)

def main():
    data = load_data(LOG_DATA_PATH)
    if not data:
        print("No data found.")
        return
    process_benchmarks(data, OUTPUT_DIR)
    
if __name__ == "__main__":
    main()