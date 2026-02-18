import re
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def parse_log_line(line):
    """Parse a single log line and extract metrics."""
    # Pattern to match log lines with metrics
    pattern = r'\[.*?\] \[INFO\] Step (\d+) - loss: ([\d.]+) \| acc_image: ([\d.]+) \| acc_text: ([\d.]+) \| lr: ([\d.]+) \| step: (\d+) \| timestamp: ([\d.]+)'
    
    match = re.search(pattern, line)
    if match:
        step, loss, acc_image, acc_text, lr, step2, timestamp = match.groups()
        
        return {
            "loss": float(loss),
            "acc_image": float(acc_image),
            "acc_text": float(acc_text),
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
    """Plot training metrics in 3-panel layout."""
    # Extract data
    steps = [m['step'] for m in metrics]
    losses = [m['loss'] for m in metrics]
    acc_images = [m['acc_image'] for m in metrics]
    acc_texts = [m['acc_text'] for m in metrics]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    # Plot 1: Loss
    axes[0].plot(steps, losses, linewidth=1.5, color='#1f77b4')
    axes[0].set_xlabel('Step', fontsize=11)
    axes[0].set_ylabel('loss', fontsize=11)
    axes[0].set_title('loss over training', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=9)
    
    # Plot 2: Image Accuracy
    axes[1].plot(steps, acc_images, linewidth=1.5, color='#ff7f0e')
    axes[1].set_xlabel('Step', fontsize=11)
    axes[1].set_ylabel('acc_image', fontsize=11)
    axes[1].set_title('acc_image over training', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(labelsize=9)
    
    # Plot 3: Text Accuracy
    axes[2].plot(steps, acc_texts, linewidth=1.5, color='#2ca02c')
    axes[2].set_xlabel('Step', fontsize=11)
    axes[2].set_ylabel('acc_text', fontsize=11)
    axes[2].set_title('acc_text over training', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(labelsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot to {output_path}")
    plt.close()

def main():
    """Main execution function."""
    print("="*80)
    print("CLIP TRAINING LOG PARSER & PLOTTER")
    print("="*80)
    
    # Define paths (update these to match your setup)
    log_dir = r"C:\Users\dheer\Coding Programs\Projects\ToyStableDiffusion\metadata\v2\clip"  # Current directory, or specify path
    output_json = "clip_training_metrics.json"
    output_plot = "clip_training_curves.png"
    
    # Find all log files matching pattern
    log_pattern = "*clip_training*.log"
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
        print(f"   Final acc_image: {merged_metrics[-1]['acc_image']:.6f}")
        print(f"   Final acc_text: {merged_metrics[-1]['acc_text']:.6f}")
        print(f"   Best acc_image: {max(m['acc_image'] for m in merged_metrics):.6f}")
        print(f"   Best acc_text: {max(m['acc_text'] for m in merged_metrics):.6f}")
        print(f"   Total steps: {len(merged_metrics)}")
        
        print("\n" + "="*80)
        print("✅ ALL DONE!")
        print("="*80)
    else:
        print("❌ No metrics found in log files!")

if __name__ == "__main__":
    main()