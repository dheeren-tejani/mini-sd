import re
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def parse_log_line(line):
    """Parse a single log line and extract metrics."""
    # Pattern to match log lines with metrics
    pattern = r'\[.*?\] \[INFO\] Step (\d+) - loss: ([\d.]+) \| lr: ([\d.]+) \| step: (\d+) \| timestamp: ([\d.]+)'
    
    match = re.search(pattern, line)
    if match:
        step, loss, lr, step2, timestamp = match.groups()
        
        return {
            "loss": float(loss),
            "lr": float(lr),
            "step": int(step),
            "timestamp": float(timestamp)
        }
    return None

def parse_log_file(file_path):
    """Parse a single log file and return list of metrics."""
    metrics = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            metric = parse_log_line(line)
            if metric:
                metrics.append(metric)
    return metrics

def merge_metrics_with_overlap_handling(all_metrics):
    """
    Merge metrics from multiple files, handling overlaps.
    When overlap detected, prefer the newer file (later timestamp for same step).
    """
    # Group by step
    step_dict = {}
    
    for metric in all_metrics:
        step = metric['step']
        if step not in step_dict:
            step_dict[step] = metric
        else:
            # If duplicate step, keep the one with later timestamp
            if metric['timestamp'] > step_dict[step]['timestamp']:
                step_dict[step] = metric
    
    # Sort by step and return as list
    merged = sorted(step_dict.values(), key=lambda x: x['step'])
    return merged

def save_json(metrics, output_path):
    """Save metrics to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved {len(metrics)} entries to {output_path}")

def plot_metrics(metrics, output_path):
    """Plot training metrics in 2-panel layout."""
    # Extract data
    steps = [m['step'] for m in metrics]
    losses = [m['loss'] for m in metrics]
    lrs = [m['lr'] for m in metrics]
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Plot 1: Loss
    axes[0].plot(steps, losses, linewidth=1.5, color='#1f77b4')
    axes[0].set_xlabel('Step', fontsize=11)
    axes[0].set_ylabel('loss', fontsize=11)
    axes[0].set_title('loss over training', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=9)
    
    # Plot 2: Learning Rate
    axes[1].plot(steps, lrs, linewidth=1.5, color='#d62728')
    axes[1].set_xlabel('Step', fontsize=11)
    axes[1].set_ylabel('learning rate', fontsize=11)
    axes[1].set_title('learning rate over training', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(labelsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot to {output_path}")
    plt.close()

def main():
    """Main execution function."""
    print("="*80)
    print("UNET TRAINING LOG PARSER & PLOTTER")
    print("="*80)
    
    # Define paths (update these to match your setup)
    log_dir = "./log-dir"  # Current directory, or specify path
    output_json = "unet_training_metrics.json"
    output_plot = "unet_training_curves.png"
    
    # Find all log files matching pattern
    log_pattern = "*unet_training*.log"
    log_files = sorted(glob.glob(f"{log_dir}/{log_pattern}"))
    
    if not log_files:
        print(f"❌ No log files found matching pattern: {log_pattern}")
        print(f"   Looking in directory: {Path(log_dir).absolute()}")
        return
    
    print(f"\n🔍 Found {len(log_files)} log files:")
    for f in log_files:
        print(f"   - {Path(f).name}")
    
    # Parse all log files
    print("\n📊 Parsing log files...")
    all_metrics = []
    
    for log_file in log_files:
        metrics = parse_log_file(log_file)
        all_metrics.extend(metrics)
        print(f"   ✓ {Path(log_file).name}: {len(metrics)} entries")
    
    print(f"\n📈 Total entries before deduplication: {len(all_metrics)}")
    
    # Handle overlaps and merge
    print("🔄 Handling overlaps and merging...")
    merged_metrics = merge_metrics_with_overlap_handling(all_metrics)
    print(f"   ✓ After deduplication: {len(merged_metrics)} entries")
    
    if merged_metrics:
        print(f"   ✓ Step range: {merged_metrics[0]['step']} → {merged_metrics[-1]['step']}")
        
        # Save to JSON
        print("\n💾 Saving to JSON...")
        save_json(merged_metrics, output_json)
        
        # Plot metrics
        print("\n📊 Generating plots...")
        plot_metrics(merged_metrics, output_plot)
        
        # Summary statistics
        print("\n📉 Training Summary:")
        print(f"   Initial loss: {merged_metrics[0]['loss']:.6f}")
        print(f"   Final loss: {merged_metrics[-1]['loss']:.6f}")
        print(f"   Best loss: {min(m['loss'] for m in merged_metrics):.6f}")
        print(f"   Initial lr: {merged_metrics[0]['lr']:.6f}")
        print(f"   Final lr: {merged_metrics[-1]['lr']:.6f}")
        print(f"   Total steps: {len(merged_metrics)}")
        
        print("\n" + "="*80)
        print("✅ ALL DONE!")
        print("="*80)
    else:
        print("❌ No metrics found in log files!")

if __name__ == "__main__":
    main()