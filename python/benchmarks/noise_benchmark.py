#!/usr/bin/env python3
"""
Benchmark Composite BKI performance across different noise levels.

This script:
1. Generates noisy labels at different noise percentages (logic in this script)
2. Runs Composite BKI refinement on each
3. Evaluates against ground truth (using osm_bki metrics logic)
4. Outputs results to CSV

NOTE: This benchmark maintains 1:1 logic parity with osm_bki:
- Noise generation: Matches the standalone noise script logic
- Metrics calculation: Uses same compute_metrics logic (skips class 0, only GT classes)
- BKI parameters: l_scale=3.0, sigma_0=1.0, prior_delta=5.0, alpha_0=0.01
"""

import numpy as np
import sys
import csv
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import osm_bki_cpp
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import osm_bki_cpp

from benchmark_utils import find_label_file


# SemanticKITTI valid classes for noise generation
VALID_CLASSES = [
    10, 11, 13, 15, 16, 18, 20,  # Vehicles
    30, 31, 32,                  # Humans
    40, 44, 48, 49,              # Flat
    50, 51, 52,                  # Construction
    70, 71, 72,                  # Nature
    80, 81                       # Objects
]

# MCD valid classes
MCD_CLASSES = [
    1, 2, 3, 5, 6, 7, 8, 9, 10,
    13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28
]


def check_files_exist(file_dict):
    """
    Check if required files exist.
    
    Args:
        file_dict: dict mapping description to Path object
        
    Returns:
        bool: True if all exist, False otherwise
    """
    missing = [name for name, path in file_dict.items() if not path.exists()]
    
    if missing:
        print("❌ Missing required files:")
        for name in missing:
            print(f"  - {name}: {file_dict[name]}")
        return False
    return True


def add_noise(labels_raw, noise_percent, noise_pool):
    """
    Add random noise to semantic labels (matches the standalone noise logic).
    
    Args:
        labels_raw: numpy array of raw uint32 labels
        noise_percent: percentage of labels to corrupt (0-100)
        noise_pool: list of valid class IDs for noise generation
        
    Returns:
        numpy array of noisy uint32 labels (with upper bits preserved)
    """
    # Extract lower 16 bits for semantic label
    labels = labels_raw & 0xFFFF
    
    n_points = len(labels)
    fraction = noise_percent / 100.0
    n_noise = int(n_points * fraction)
    
    if n_noise == 0:
        return labels_raw.copy()
    
    # Select random indices to corrupt
    noise_indices = np.random.choice(n_points, n_noise, replace=False)
    
    # Generate random labels for these indices
    random_labels = np.random.choice(noise_pool, n_noise)
    
    # Apply noise
    new_labels = labels.copy()
    new_labels[noise_indices] = random_labels
    
    # Preserve original upper bits (instance IDs)
    upper_bits = labels_raw & 0xFFFF0000
    final_data = upper_bits | new_labels.astype(np.uint32)
    
    return final_data


def calculate_metrics(pred_labels, gt_labels):
    """
    Calculate accuracy and mIoU (matches osm_bki::compute_metrics exactly).
    
    This function is a 1:1 copy of the compute_metrics function from osm_bki
    to ensure identical evaluation logic. Key behaviors:
    - Only evaluates classes present in ground truth (unique_gt)
    - Skips class 0 (unlabeled/invalid)
    - Only includes classes with union > 0 in mIoU calculation
    
    Args:
        pred_labels: predicted labels
        gt_labels: ground truth labels
        
    Returns:
        dict with 'accuracy' and 'miou'
    """
    intersection = {}
    union = {}
    correct = {}
    total = {}
    
    total_correct = 0
    total_valid = 0
    
    unique_gt = np.unique(gt_labels)
    
    for cls in unique_gt:
        if cls == 0:  # Skip class 0 (matches osm_bki line 292)
            continue
        
        gt_mask = (gt_labels == cls)
        pred_mask = (pred_labels == cls)
        
        inter = np.sum(gt_mask & pred_mask)
        uni = np.sum(gt_mask | pred_mask)
        count = np.sum(gt_mask)
        
        intersection[cls] = inter
        union[cls] = uni
        correct[cls] = inter
        total[cls] = count
        
        total_correct += inter
        total_valid += count
    
    iou_list = []
    
    for cls in intersection:
        if union[cls] > 0:
            val = intersection[cls] / union[cls]
            iou_list.append(val)
    
    miou = np.mean(iou_list) if iou_list else 0.0
    accuracy = total_correct / total_valid if total_valid > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'miou': miou
    }


def run_single_benchmark(
    bki,
    lidar_path,
    noisy_labels_raw,
    gt_labels,
    noisy_labels_path
):
    """
    Run a single benchmark iteration using a pre-constructed BKI instance.
    
    Args:
        bki: pre-constructed PyContinuousBKI instance (will be cleared and reused)
        lidar_path: path to LiDAR data
        noisy_labels_raw: noisy labels (uint32 with upper bits)
        gt_labels: ground truth labels (semantic only)
        noisy_labels_path: file path to save noisy labels (persistent)
        
    Returns:
        dict with 'metrics_before' and 'metrics_after'
    """
    noisy_labels = (noisy_labels_raw & 0xFFFF).astype(np.uint32)
    metrics_before = calculate_metrics(noisy_labels, gt_labels)

    noisy_labels_raw.tofile(noisy_labels_path)

    bki.clear()

    points = np.fromfile(str(lidar_path), dtype=np.float32).reshape((-1, 4))[:, :3]
    noisy_labels_semantic = (noisy_labels_raw & 0xFFFF).astype(np.uint32)
    bki.update(noisy_labels_semantic, points)

    refined_labels = bki.infer(points)
    refined_labels = np.array(refined_labels, dtype=np.uint32)

    metrics_after = calculate_metrics(refined_labels, gt_labels)

    return {
        'before': metrics_before,
        'after': metrics_after
    }

def get_scan_gt_pairs(scan_dir, gt_dir):
    """Return list of (scan_path, gt_path) for scans that have matching GT labels."""
    scan_dir = Path(scan_dir)
    gt_dir = Path(gt_dir)
    pairs = []
    for scan_path in sorted(scan_dir.glob("*.bin")):
        stem = scan_path.stem
        gt_path = find_label_file(gt_dir, stem)
        if gt_path:
            pairs.append((scan_path, Path(gt_path)))
    return pairs


def run_benchmark(
    scan_dir,
    gt_dir,
    osm_path,
    config_path,
    noise_levels,
    output_csv,
    num_runs=3,
    use_kitti=False,
    max_scans=None
):
    """
    Run benchmark across different noise levels.
    
    Args:
        scan_dir: directory of LiDAR point clouds (.bin)
        gt_dir: directory of ground truth labels
        osm_path: path to OSM geometries
        config_path: path to config YAML
        noise_levels: list of noise percentages to test
        output_csv: path to output CSV file
        num_runs: number of runs per noise level for averaging
        use_kitti: if True, use SemanticKITTI class list for noise
        max_scans: max scans to process (default: all)
    """
    print("=" * 80)
    print("Composite BKI Noise Benchmark")
    print("=" * 80)
    print()
    
    pairs = get_scan_gt_pairs(scan_dir, gt_dir)
    if not pairs:
        raise FileNotFoundError(f"No matching scan/GT pairs in {scan_dir} and {gt_dir}")
    if max_scans is not None:
        pairs = pairs[:max_scans]
    
    # Check files exist
    if not check_files_exist({
        "Scan Dir": Path(scan_dir),
        "GT Dir": Path(gt_dir),
        "OSM geometries": Path(osm_path),
        "Config": Path(config_path)
    }):
        raise FileNotFoundError("Required files missing")
    
    print(f"Processing {len(pairs)} scans...")
    
    # Prepare results: collect per-scan, per-noise-level, per-run
    all_run_results = []  # list of (scan_stem, noise_level, run_idx, metrics)
    
    # Create directory for noisy labels
    noisy_labels_dir = Path(output_csv).parent / "noisy_labels"
    noisy_labels_dir.mkdir(exist_ok=True)
    print(f"Saving noisy labels to: {noisy_labels_dir}")
    print()
    
    noise_pool = VALID_CLASSES if use_kitti else MCD_CLASSES

    bki = osm_bki_cpp.PyContinuousBKI(
        osm_path=str(osm_path),
        config_path=str(config_path),
        resolution=1.0,
        l_scale=3.0,
        sigma_0=1.0,
        prior_delta=5.0,
        height_sigma=0.3,
        use_semantic_kernel=True,
        use_spatial_kernel=True,
        num_threads=-1,
        alpha0=0.01,
        seed_osm_prior=False,
        osm_prior_strength=0.0
    )

    for scan_path, gt_path in pairs:
        scan_stem = scan_path.stem
        gt_raw = np.fromfile(gt_path, dtype=np.uint32)
        gt_labels = (gt_raw & 0xFFFF).astype(np.uint32)
        print(f"Scan {scan_stem}: {len(gt_labels)} points")

        for noise_level in noise_levels:
            for run in range(num_runs):
                noisy_labels_raw = add_noise(gt_raw, noise_level, noise_pool)
                noisy_labels_path = noisy_labels_dir / f"{scan_stem}_noisy_{int(noise_level)}pct_run{run}.labels"

                metrics = run_single_benchmark(
                    bki=bki,
                    lidar_path=scan_path,
                    noisy_labels_raw=noisy_labels_raw,
                    gt_labels=gt_labels,
                    noisy_labels_path=noisy_labels_path
                )
                all_run_results.append((scan_stem, noise_level, run, metrics))

    # Aggregate by noise level (across scans and runs)
    results = []
    for noise_level in noise_levels:
        level_results = [r for r in all_run_results if r[1] == noise_level]
        accuracies_before = [r[3]['before']['accuracy'] for r in level_results]
        mious_before = [r[3]['before']['miou'] for r in level_results]
        accuracies_after = [r[3]['after']['accuracy'] for r in level_results]
        mious_after = [r[3]['after']['miou'] for r in level_results]

        results.append({
            'noise_level': noise_level,
            'accuracy_before': np.mean(accuracies_before),
            'miou_before': np.mean(mious_before),
            'accuracy_after': np.mean(accuracies_after),
            'miou_after': np.mean(mious_after),
            'accuracy_improvement': np.mean(accuracies_after) - np.mean(accuracies_before),
            'miou_improvement': np.mean(mious_after) - np.mean(mious_before),
            'accuracy_before_std': np.std(accuracies_before),
            'miou_before_std': np.std(mious_before),
            'accuracy_after_std': np.std(accuracies_after),
            'miou_after_std': np.std(mious_after)
        })
        result = results[-1]
        print(f"Noise {noise_level}%: Before Acc={result['accuracy_before']*100:.2f}%, mIoU={result['miou_before']*100:.2f}% | "
              f"After Acc={result['accuracy_after']*100:.2f}%, mIoU={result['miou_after']*100:.2f}% | "
              f"Δ Acc={result['accuracy_improvement']*100:+.2f}%, mIoU={result['miou_improvement']*100:+.2f}%")
    
    # Write results to CSV
    print(f"Writing results to {output_csv}...")
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Noise Level',
            'Accuracy_Before', 'mIoU_Before',
            'Accuracy_After', 'mIoU_After',
            'Accuracy_Improvement', 'mIoU_Improvement',
            'Accuracy_Before_Std', 'mIoU_Before_Std',
            'Accuracy_After_Std', 'mIoU_After_Std'
        ])
        
        for result in results:
            writer.writerow([
                result['noise_level'],
                f"{result['accuracy_before']*100:.4f}",
                f"{result['miou_before']*100:.4f}",
                f"{result['accuracy_after']*100:.4f}",
                f"{result['miou_after']*100:.4f}",
                f"{result['accuracy_improvement']*100:.4f}",
                f"{result['miou_improvement']*100:.4f}",
                f"{result['accuracy_before_std']*100:.4f}",
                f"{result['miou_before_std']*100:.4f}",
                f"{result['accuracy_after_std']*100:.4f}",
                f"{result['miou_after_std']*100:.4f}"
            ])
    
    print()
    print("=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)
    print()
    print("Results Summary:")
    print(f"{'Noise':<8} {'Acc Before':<12} {'mIoU Before':<13} {'Acc After':<12} {'mIoU After':<13} {'Acc Δ':<10} {'mIoU Δ':<10}")
    print(f"{'Level':<8} {'(%)':<12} {'(%)':<13} {'(%)':<12} {'(%)':<13} {'(%)':<10} {'(%)':<10}")
    print("-" * 88)
    for result in results:
        print(f"{result['noise_level']:<8.0f} "
              f"{result['accuracy_before']*100:<12.2f} "
              f"{result['miou_before']*100:<13.2f} "
              f"{result['accuracy_after']*100:<12.2f} "
              f"{result['miou_after']*100:<13.2f} "
              f"{result['accuracy_improvement']*100:<+10.2f} "
              f"{result['miou_improvement']*100:<+10.2f}")
    print()
    print(f"📊 Results saved to: {output_csv}")
    print(f"🏷️  Noisy labels saved to: {Path(output_csv).parent / 'noisy_labels'}/")
    
    # Count total noisy label files generated
    total_files = len(noise_levels) * num_runs
    print(f"   Generated {total_files} noisy label files for reproducibility")
    print()


def main():
    """Main benchmark execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark Composite BKI across different noise levels"
    )
    
    parser.add_argument(
        "--scan-dir",
        type=str,
        default="../example_data/mcd-data/data",
        help="Directory of LiDAR point clouds (.bin)"
    )
    
    parser.add_argument(
        "--gt-dir",
        type=str,
        default="../example_data/mcd-data/labels_groundtruth",
        help="Directory of ground truth labels"
    )
    
    parser.add_argument(
        "--max-scans",
        type=int,
        default=None,
        help="Max scans to process (default: all)"
    )
    
    parser.add_argument(
        "--osm",
        type=str,
        default="../example_data/mcd-data/kth_day_06_osm_geometries.bin",
        help="Path to OSM geometries"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/mcd_config.yaml",
        help="Path to configuration YAML"
    )
    
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        help="Noise levels to test (percentages)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: auto-generated with timestamp)"
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per noise level for averaging"
    )

    parser.add_argument(
        "--kitti-labels",
        action="store_true",
        help="Use SemanticKITTI class list for noise generation (default: MCD)"
    )
    
    args = parser.parse_args()
    
    # Set default output path with timestamp
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = Path(__file__).parent / f"noise_benchmark_{timestamp}.csv"
    else:
        output_csv = Path(args.output)
    
    # Convert relative paths to absolute
    script_dir = Path(__file__).parent
    scan_dir = (script_dir / args.scan_dir).resolve()
    gt_dir = (script_dir / args.gt_dir).resolve()
    osm_path = (script_dir / args.osm).resolve()
    config_path = (script_dir / args.config).resolve()
    
    # Run benchmark
    run_benchmark(
        scan_dir=scan_dir,
        gt_dir=gt_dir,
        osm_path=osm_path,
        config_path=config_path,
        noise_levels=args.noise_levels,
        output_csv=output_csv,
        num_runs=args.runs,
        use_kitti=args.kitti_labels,
        max_scans=args.max_scans
    )


if __name__ == "__main__":
    main()
