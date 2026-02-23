#!/usr/bin/env python3
"""
Calibration Benchmark.

Evaluates how well-calibrated the BKI output probabilities are.
A well-calibrated model produces confidence scores that match the true accuracy.
e.g., if the model predicts class X with 0.8 confidence, it should be correct 80% of the time.

Metrics:
- ECE (Expected Calibration Error)
- MCE (Maximum Calibration Error)
- Reliability Diagram (binned accuracy vs confidence)
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
    find_label_file, get_frame_number, check_files_exist
)

def compute_calibration_metrics(probs, gt, num_bins=10):
    """
    Compute ECE and reliability diagram stats.
    probs: (N, K) array of probabilities
    gt: (N,) array of ground truth labels
    """
    # Get max probability and predicted class
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    
    accuracies = (predictions == gt)
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    
    ece = 0.0
    mce = 0.0
    bin_stats = []
    
    total_samples = len(gt)
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        # Indices in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            
            diff = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += diff * prop_in_bin
            mce = max(mce, diff)
            
            bin_stats.append({
                "bin_idx": i,
                "lower": bin_lower,
                "upper": bin_upper,
                "samples": np.sum(in_bin),
                "accuracy": accuracy_in_bin,
                "confidence": avg_confidence_in_bin
            })
            
    return ece, mce, bin_stats

def main():
    parser = argparse.ArgumentParser(description="Calibration Benchmark")
    
    # Data paths
    parser.add_argument("--scan-dir", default="../example_data/mcd-data/data", help="Directory of .bin scans")
    parser.add_argument("--label-dir", default="../example_data/mcd-data/labels_predicted", help="Directory of input labels")
    parser.add_argument("--gt-dir", default="../example_data/mcd-data/labels_groundtruth", help="Directory of GT labels")
    parser.add_argument("--osm", default="../example_data/mcd-data/kth_day_06_osm_geometries.bin", help="Path to OSM geometries")
    parser.add_argument("--pose", default="../example_data/mcd-data/pose_inW.csv", help="Pose CSV file")
    parser.add_argument("--config", default="../configs/mcd_config.yaml", help="Path to YAML config")
    
    # BKI Parameters
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--l-scale", type=float, default=3.0)
    parser.add_argument("--sigma-0", type=float, default=1.0)
    parser.add_argument("--prior-delta", type=float, default=0.5)
    parser.add_argument("--height-sigma", type=float, default=0.3)
    parser.add_argument("--alpha0", type=float, default=1.0)
    parser.add_argument("--seed-osm-prior", dest="seed_osm_prior", action="store_true")
    parser.add_argument("--no-seed-osm-prior", dest="seed_osm_prior", action="store_false", help="Disable OSM prior seeding")
    parser.set_defaults(seed_osm_prior=True)
    parser.add_argument("--osm-prior-strength", type=float, default=0.1)
    parser.add_argument("--disable-osm-fallback", action="store_true")
    
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
    poses = load_poses(pose_path)
    scan_files = sorted(scan_dir.glob("*.bin"))
    
    # Initialize BKI
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
    
    all_probs = []
    all_gt = []
    
    print(f"\n🚀 Starting Calibration Benchmark")
    print(f"   Total Scans: {len(scan_files)}")
    print("-" * 60)
    
    for i, scan_path in enumerate(scan_files):
        stem = scan_path.stem
        frame = get_frame_number(stem)
        
        label_path = find_label_file(label_dir, stem)
        gt_path = find_label_file(gt_dir, stem)
        
        if not label_path or not gt_path:
            continue
            
        points_xyz, _ = load_scan(str(scan_path))
        input_labels = load_labels(label_path)
        gt_labels = load_labels(gt_path)
        
        if poses is not None and frame is not None and frame in poses:
            points_xyz = transform_points_to_world(points_xyz, poses[frame])
        else:
            continue
            
        # Update
        bki.update(input_labels, points_xyz)
        
        # Infer Probabilities
        probs = bki.infer_probs(points_xyz)
        
        # Match lengths
        n = min(len(probs), len(gt_labels))
        if n == 0: continue
        
        # Filter ignore labels (0)
        mask = gt_labels[:n] != 0
        
        if np.any(mask):
            all_probs.append(probs[:n][mask])
            all_gt.append(gt_labels[:n][mask])
            
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} scans...", flush=True)
            
    # Concatenate all results
    if not all_probs:
        print("No valid data collected.")
        return
        
    full_probs = np.vstack(all_probs)
    full_gt = np.concatenate(all_gt)
    
    # Compute Metrics
    ece, mce, bin_stats = compute_calibration_metrics(full_probs, full_gt)
    
    print(f"\nCalibration Results:")
    print(f"  ECE (Expected Calibration Error): {ece:.4f}")
    print(f"  MCE (Maximum Calibration Error):  {mce:.4f}")
    
    # Write Results
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = script_dir / "calibration_results"
        output_dir.mkdir(exist_ok=True)
        output_csv = output_dir / f"calibration_bins_{timestamp}.csv"
    else:
        output_csv = Path(args.output)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        
    print(f"\nWriting reliability diagram data to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ["bin_idx", "lower", "upper", "samples", "accuracy", "confidence"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bin_stats)
        
    print("\nDone.")

if __name__ == "__main__":
    main()
