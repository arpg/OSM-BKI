#!/usr/bin/env python3
"""
Multi-Scan Convergence Benchmark.

Evaluates map quality (Accuracy, mIoU) as a function of the number of integrated scans.
This measures how the map improves (or degrades) as more evidence is accumulated.

Key features:
- Uses world-frame poses to align scans.
- Evaluates on the full dataset (or a subset) at specific checkpoints (e.g., after 1, 5, 10... scans).
- Can use predicted labels or GT labels as input (to test "denoising" vs "fusion" power).
"""

import sys
import argparse
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
import time

# Add parent directory to path to import osm_bki_cpp
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import osm_bki_cpp

# Import benchmark utilities
from benchmark_utils import (
    load_poses, transform_points_to_world, load_scan, load_labels,
    find_label_file, get_frame_number, compute_metrics, check_files_exist
)

def run_evaluation(bki, test_files, label_dir, gt_dir, poses):
    """Evaluate current BKI map on a set of test files."""
    all_metrics = []
    
    for scan_path in test_files:
        stem = scan_path.stem
        frame = get_frame_number(stem)
        
        # Load GT
        gt_path = find_label_file(gt_dir, stem)
        if not gt_path:
            continue
        gt = load_labels(gt_path)
        
        # Load Points
        points_xyz, _ = load_scan(str(scan_path))
        
        # Transform to world
        if poses is not None and frame is not None and frame in poses:
            points_xyz = transform_points_to_world(points_xyz, poses[frame])
        else:
            # Skip if no pose (cannot align)
            continue
            
        # Infer
        pred = bki.infer(points_xyz)
        
        # Compute metrics
        # Truncate if sizes mismatch (shouldn't happen if data is consistent)
        n = min(len(pred), len(gt))
        if n > 0:
            metrics = compute_metrics(pred[:n], gt[:n])
            all_metrics.append(metrics)
            
    if not all_metrics:
        return 0.0, 0.0
        
    avg_acc = np.mean([m["accuracy"] for m in all_metrics])
    avg_miou = np.mean([m["miou"] for m in all_metrics])
    
    return avg_acc, avg_miou

def main():
    parser = argparse.ArgumentParser(description="Multi-Scan Convergence Benchmark")
    
    # Data paths
    parser.add_argument("--scan-dir", default="../example_data/mcd-data/data", help="Directory of .bin scans")
    parser.add_argument("--label-dir", default="../example_data/mcd-data/labels_predicted", help="Directory of input labels")
    parser.add_argument("--gt-dir", default="../example_data/mcd-data/labels_groundtruth", help="Directory of GT labels")
    parser.add_argument("--osm", default="../example_data/mcd-data/kth_day_06_osm_geometries.bin", help="Path to OSM geometries")
    parser.add_argument("--pose", default="../example_data/mcd-data/pose_inW.csv", help="Pose CSV file")
    parser.add_argument("--config", default="../configs/mcd_config.yaml", help="Path to YAML config")
    
    # BKI Parameters (Defaults)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--l-scale", type=float, default=3.0)
    parser.add_argument("--sigma-0", type=float, default=1.0)
    parser.add_argument("--prior-delta", type=float, default=0.5)
    parser.add_argument("--height-sigma", type=float, default=0.3)
    parser.add_argument("--alpha0", type=float, default=1.0)
    parser.add_argument("--seed-osm-prior", dest="seed_osm_prior", action="store_true", help="Enable OSM prior seeding")
    parser.add_argument("--no-seed-osm-prior", dest="seed_osm_prior", action="store_false", help="Disable OSM prior seeding")
    parser.set_defaults(seed_osm_prior=True)
    parser.add_argument("--osm-prior-strength", type=float, default=0.1)
    parser.add_argument("--disable-osm-fallback", action="store_true")
    
    # Benchmark Options
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100], help="Scan counts to evaluate at")
    parser.add_argument("--use-gt-input", action="store_true", help="Use GT labels as input (instead of predicted)")
    parser.add_argument("--output", default=None, help="Output CSV file")
    
    args = parser.parse_args()

    # Resolution guardrails: avoid too coarse/fine settings for fair comparisons
    if args.resolution < 0.2 or args.resolution > 1.0 or abs(args.resolution - 0.5) < 1e-9:
        parser.error(
            "--resolution must be in [0.2, 1.0] and cannot be exactly 0.5"
        )

    # Keep kernel length-scale reasonably aligned with voxel size
    lscale_ratio = args.l_scale / args.resolution
    if lscale_ratio < 1.5 or lscale_ratio > 6.0:
        parser.error(
            "--l-scale must be reasonably aligned with --resolution: "
            "expected 1.5 <= (l-scale / resolution) <= 6.0"
        )
    
    # Resolve paths
    script_dir = Path(__file__).parent
    scan_dir = (script_dir / args.scan_dir).resolve()
    label_dir = (script_dir / args.label_dir).resolve()
    gt_dir = (script_dir / args.gt_dir).resolve()
    osm_path = (script_dir / args.osm).resolve()
    pose_path = (script_dir / args.pose).resolve()
    config_path = (script_dir / args.config).resolve()
    
    # Override input label dir if using GT
    if args.use_gt_input:
        print("ℹ️  Using Ground Truth labels as INPUT.")
        label_dir = gt_dir
        
    # Check files
    if not check_files_exist({
        "Scan Dir": scan_dir,
        "Label Dir": label_dir,
        "GT Dir": gt_dir,
        "OSM": osm_path,
        "Pose": pose_path,
        "Config": config_path
    }):
        return 1
        
    # Load Poses
    print("Loading poses...")
    poses = load_poses(pose_path)
    
    # Get Scan Files
    scan_files = sorted(scan_dir.glob("*.bin"))
    if not scan_files:
        print(f"No scans found in {scan_dir}")
        return 1
    
    # Sort checkpoints
    checkpoints = sorted(list(set(args.checkpoints)))
    max_checkpoint = max(checkpoints)
    if max_checkpoint > len(scan_files):
        print(f"⚠️  Max checkpoint ({max_checkpoint}) > num scans ({len(scan_files)}). Truncating.")
        checkpoints = [c for c in checkpoints if c <= len(scan_files)]
        # Always include the final scan if not present
        if len(scan_files) not in checkpoints:
            checkpoints.append(len(scan_files))
            
    # Initialize BKI
    print("Initializing BKI...")
    bki = osm_bki_cpp.PyContinuousBKI(
        osm_path=str(osm_path),
        config_path=str(config_path),
        resolution=args.resolution,
        l_scale=args.l_scale,
        sigma_0=args.sigma_0,
        prior_delta=args.prior_delta,
        height_sigma=args.height_sigma,
        use_semantic_kernel=True,
        use_spatial_kernel=True,
        num_threads=-1,
        alpha0=args.alpha0,
        seed_osm_prior=args.seed_osm_prior,
        osm_prior_strength=args.osm_prior_strength,
        osm_fallback_in_infer=not args.disable_osm_fallback
    )
    
    # Prepare Output
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = script_dir / "multi_scan_results"
        output_dir.mkdir(exist_ok=True)
        output_csv = output_dir / f"multi_scan_benchmark_{timestamp}.csv"
    else:
        output_csv = Path(args.output)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        
    results = []
    
    print(f"\n🚀 Starting Multi-Scan Benchmark")
    print(f"   Total Scans: {len(scan_files)}")
    print(f"   Checkpoints: {checkpoints}")
    print(f"   Input Labels: {'GT' if args.use_gt_input else 'Predicted'}")
    print("-" * 60)
    
    start_time = time.time()
    
    # Training Loop
    for i, scan_path in enumerate(scan_files):
        scan_idx = i + 1
        
        # Load Data
        stem = scan_path.stem
        frame = get_frame_number(stem)
        label_path = find_label_file(label_dir, stem)
        
        if not label_path:
            print(f"⚠️  Skipping scan {stem}: Label file not found")
            continue
            
        points_xyz, _ = load_scan(str(scan_path))
        labels = load_labels(label_path)
        
        # Transform
        if poses is not None and frame is not None and frame in poses:
            points_xyz = transform_points_to_world(points_xyz, poses[frame])
        else:
            print(f"⚠️  Skipping scan {stem}: Pose not found")
            continue
            
        # Update Map
        bki.update(labels, points_xyz)
        
        # Checkpoint Evaluation
        if scan_idx in checkpoints:
            print(f"Evaluating at checkpoint: {scan_idx} scans...", end=" ", flush=True)
            eval_start = time.time()
            
            # Evaluate on ALL scans (or a subset if we wanted to optimize)
            # Using all scans gives the best "global map quality" metric
            acc, miou = run_evaluation(bki, scan_files, label_dir, gt_dir, poses)
            
            eval_time = time.time() - eval_start
            print(f"Acc: {acc*100:.2f}%, mIoU: {miou*100:.2f}% ({eval_time:.1f}s)")
            
            results.append({
                "scans_integrated": scan_idx,
                "accuracy": acc,
                "miou": miou,
                "map_size": bki.get_size(),
                "time_elapsed": time.time() - start_time
            })
            
    # Write Results
    print(f"\nWriting results to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["scans_integrated", "accuracy", "miou", "map_size", "time_elapsed"])
        writer.writeheader()
        writer.writerows(results)
        
    print("\nDone.")

if __name__ == "__main__":
    main()
