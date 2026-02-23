#!/usr/bin/env python3
"""
Kernel Ablation Study: Compare Composite BKI performance with different kernel configurations.

This benchmark evaluates the contribution of each kernel component:
1. Spatial kernel only (no semantic/OSM priors)
2. Semantic kernel only (no spatial distance weighting)
3. Both kernels (full algorithm)

This helps understand which component contributes more to the overall performance.
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
    """Check if required files exist."""
    missing = [name for name, path in file_dict.items() if not path.exists()]
    
    if missing:
        print("❌ Missing required files:")
        for name in missing:
            print(f"  - {name}: {file_dict[name]}")
        return False
    return True


def add_noise(labels_raw, noise_percent, noise_pool):
    """Add random noise to semantic labels (matches standalone noise logic)."""
    labels = labels_raw & 0xFFFF
    
    n_points = len(labels)
    fraction = noise_percent / 100.0
    n_noise = int(n_points * fraction)
    
    if n_noise == 0:
        return labels_raw.copy()
    
    noise_indices = np.random.choice(n_points, n_noise, replace=False)
    random_labels = np.random.choice(noise_pool, n_noise)
    
    new_labels = labels.copy()
    new_labels[noise_indices] = random_labels
    
    upper_bits = labels_raw & 0xFFFF0000
    final_data = upper_bits | new_labels.astype(np.uint32)
    
    return final_data


def calculate_metrics(pred_labels, gt_labels):
    """Calculate accuracy and mIoU (matches osm_bki logic exactly)."""
    intersection = {}
    union = {}
    correct = {}
    total = {}
    
    total_correct = 0
    total_valid = 0
    
    unique_gt = np.unique(gt_labels)
    
    for cls in unique_gt:
        if cls == 0:
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


def run_single_config(
    lidar_path,
    noisy_labels_path,
    gt_labels,
    osm_path,
    config_path,
    use_semantic_kernel,
    use_spatial_kernel,
    config_name
):
    """
    Run BKI with specific kernel configuration.
    
    Args:
        lidar_path: path to LiDAR data
        noisy_labels_path: path to noisy label file
        gt_labels: ground truth labels (semantic only)
        osm_path: path to OSM geometries
        config_path: path to config YAML
        use_semantic_kernel: enable semantic kernel
        use_spatial_kernel: enable spatial kernel
        config_name: descriptive name for this configuration
        
    Returns:
        dict with metrics
    """
    # Extract noisy labels for baseline
    noisy_raw = np.fromfile(noisy_labels_path, dtype=np.uint32)
    noisy_labels = (noisy_raw & 0xFFFF).astype(np.uint32)
    
    # Calculate metrics BEFORE refinement
    metrics_before = calculate_metrics(noisy_labels, gt_labels)
    
    print(f"    Running with {config_name}...", end=" ", flush=True)
    
    # Run refinement with specified kernel configuration
    # Using PyContinuousBKI directly
    bki = osm_bki_cpp.PyContinuousBKI(
        osm_path=str(osm_path),
        config_path=str(config_path),
        resolution=1.0,
        l_scale=1.0,
        sigma_0=1.0,
        prior_delta=0.5,
        height_sigma=0.3,
        use_semantic_kernel=use_semantic_kernel,
        use_spatial_kernel=use_spatial_kernel,
        num_threads=-1,
        alpha0=1,
        seed_osm_prior=True,
        osm_prior_strength=0.0
    )

    # Load points
    points = np.fromfile(str(lidar_path), dtype=np.float32).reshape((-1, 4))[:, :3]
    
    # Load noisy labels and mask
    noisy_raw = np.fromfile(str(noisy_labels_path), dtype=np.uint32)
    noisy_labels_semantic = (noisy_raw & 0xFFFF).astype(np.uint32)
    
    # Update
    bki.update(noisy_labels_semantic, points)
    
    # Infer
    refined_labels = bki.infer(points)
    refined_labels = np.array(refined_labels, dtype=np.uint32)
    
    # Calculate metrics AFTER refinement
    metrics_after = calculate_metrics(refined_labels, gt_labels)
    
    print(f"Acc: {metrics_after['accuracy']*100:.2f}%, mIoU: {metrics_after['miou']*100:.2f}%")
    
    return {
        'config_name': config_name,
        'use_semantic': use_semantic_kernel,
        'use_spatial': use_spatial_kernel,
        'accuracy_before': metrics_before['accuracy'],
        'miou_before': metrics_before['miou'],
        'accuracy_after': metrics_after['accuracy'],
        'miou_after': metrics_after['miou'],
        'accuracy_improvement': metrics_after['accuracy'] - metrics_before['accuracy'],
        'miou_improvement': metrics_after['miou'] - metrics_before['miou']
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


def run_ablation_study(
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
    Run kernel ablation study across different noise levels.
    
    Compares three configurations:
    1. Spatial kernel only (no semantic)
    2. Semantic kernel only (no spatial)
    3. Both kernels (full algorithm)
    """
    print("=" * 80)
    print("Composite BKI - Kernel Ablation Study")
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
    
    # Create directory for noisy labels
    noisy_labels_dir = Path(output_csv).parent / "ablation_noisy_labels"
    noisy_labels_dir.mkdir(exist_ok=True)
    
    # Kernel configurations to test
    configs = [
        ("Spatial Only", False, True),      # No semantic, yes spatial
        ("Semantic Only", True, False),     # Yes semantic, no spatial
        ("Both Kernels", True, True)        # Yes semantic, yes spatial (full algorithm)
    ]
    
    results = []
    noise_pool = VALID_CLASSES if use_kitti else MCD_CLASSES

    for scan_path, gt_path in pairs:
        scan_stem = scan_path.stem
        gt_raw = np.fromfile(gt_path, dtype=np.uint32)
        gt_labels = (gt_raw & 0xFFFF).astype(np.uint32)
        print(f"Scan {scan_stem}: {len(gt_labels)} points")

        for noise_level in noise_levels:
            for run in range(num_runs):
                noisy_labels_raw = add_noise(gt_raw, noise_level, noise_pool)
                noisy_labels_path = noisy_labels_dir / f"{scan_stem}_noisy_{int(noise_level)}pct_run{run}.labels"
                noisy_labels_raw.tofile(noisy_labels_path)

                for config_name, use_semantic, use_spatial in configs:
                    result = run_single_config(
                        lidar_path=scan_path,
                        noisy_labels_path=noisy_labels_path,
                        gt_labels=gt_labels,
                        osm_path=osm_path,
                        config_path=config_path,
                        use_semantic_kernel=use_semantic,
                        use_spatial_kernel=use_spatial,
                        config_name=config_name
                    )
                    result['noise_level'] = noise_level
                    result['run'] = run
                    result['scan'] = scan_stem
                    results.append(result)
    
    # Write detailed results to CSV
    print(f"Writing results to {output_csv}...")
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Scan', 'Noise_Level', 'Run', 'Configuration',
            'Use_Semantic', 'Use_Spatial',
            'Accuracy_Before', 'mIoU_Before',
            'Accuracy_After', 'mIoU_After',
            'Accuracy_Improvement', 'mIoU_Improvement'
        ])
        
        for result in results:
            writer.writerow([
                result['scan'],
                result['noise_level'],
                result['run'],
                result['config_name'],
                'Yes' if result['use_semantic'] else 'No',
                'Yes' if result['use_spatial'] else 'No',
                f"{result['accuracy_before']*100:.4f}",
                f"{result['miou_before']*100:.4f}",
                f"{result['accuracy_after']*100:.4f}",
                f"{result['miou_after']*100:.4f}",
                f"{result['accuracy_improvement']*100:.4f}",
                f"{result['miou_improvement']*100:.4f}"
            ])
    
    # Compute and display aggregated statistics
    print()
    print("=" * 80)
    print("Ablation Study Complete!")
    print("=" * 80)
    print()
    
    # Aggregate by noise level and configuration
    print("Average Results by Noise Level and Configuration:")
    print()
    
    for noise_level in noise_levels:
        print(f"Noise Level: {noise_level}%")
        print(f"{'Configuration':<20} {'Acc After':<12} {'mIoU After':<12} {'Acc Δ':<10} {'mIoU Δ':<10}")
        print("-" * 64)
        
        for config_name, use_semantic, use_spatial in configs:
            # Filter results for this noise level and config (aggregate across scans/runs)
            filtered = [r for r in results
                       if r['noise_level'] == noise_level
                       and r['config_name'] == config_name]
            
            if filtered:
                avg_acc_after = np.mean([r['accuracy_after'] for r in filtered])
                avg_miou_after = np.mean([r['miou_after'] for r in filtered])
                avg_acc_imp = np.mean([r['accuracy_improvement'] for r in filtered])
                avg_miou_imp = np.mean([r['miou_improvement'] for r in filtered])
                
                print(f"{config_name:<20} "
                      f"{avg_acc_after*100:<12.2f} "
                      f"{avg_miou_after*100:<12.2f} "
                      f"{avg_acc_imp*100:<+10.2f} "
                      f"{avg_miou_imp*100:<+10.2f}")
        print()
    
    print(f"📊 Detailed results saved to: {output_csv}")
    print(f"🏷️  Noisy labels saved to: {noisy_labels_dir}/")
    print()


def main():
    """Main ablation study execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Kernel ablation study for Composite BKI"
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
        default=[10, 30, 50, 70, 90, 99],
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
        default=1,
        help="Number of runs per configuration for averaging"
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
        output_csv = Path(__file__).parent / f"kernel_ablation_{timestamp}.csv"
    else:
        output_csv = Path(args.output)
    
    # Convert relative paths to absolute
    script_dir = Path(__file__).parent
    scan_dir = (script_dir / args.scan_dir).resolve()
    gt_dir = (script_dir / args.gt_dir).resolve()
    osm_path = (script_dir / args.osm).resolve()
    config_path = (script_dir / args.config).resolve()
    
    # Run ablation study
    run_ablation_study(
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
