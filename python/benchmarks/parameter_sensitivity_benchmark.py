#!/usr/bin/env python3
"""
Parameter Sensitivity Benchmark.

Performs one-at-a-time parameter sweeps to understand the impact of hyperparameters
on map quality and performance.

Parameters swept:
- resolution
- l_scale
- sigma_0
- prior_delta
- alpha0
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
        
        gt_path = find_label_file(gt_dir, stem)
        if not gt_path:
            continue
        gt = load_labels(gt_path)
        
        points_xyz, _ = load_scan(str(scan_path))
        
        if poses is not None and frame is not None and frame in poses:
            points_xyz = transform_points_to_world(points_xyz, poses[frame])
        else:
            continue
            
        pred = bki.infer(points_xyz)
        
        n = min(len(pred), len(gt))
        if n > 0:
            metrics = compute_metrics(pred[:n], gt[:n])
            all_metrics.append(metrics)
            
    if not all_metrics:
        return 0.0, 0.0
        
    avg_acc = np.mean([m["accuracy"] for m in all_metrics])
    avg_miou = np.mean([m["miou"] for m in all_metrics])
    
    return avg_acc, avg_miou

def run_config(param_name, param_value, scan_files, label_dir, gt_dir, poses, osm_path, config_path, args):
    """Run a single configuration."""
    print(f"\nRunning {param_name}={param_value}...", end=" ", flush=True)
    
    # Default parameters
    params = {
        "resolution": args.resolution,
        "l_scale": args.l_scale,
        "sigma_0": args.sigma_0,
        "prior_delta": args.prior_delta,
        "height_sigma": args.height_sigma,
        "alpha0": args.alpha0,
        "seed_osm_prior": args.seed_osm_prior,
        "osm_prior_strength": args.osm_prior_strength,
        "osm_fallback_in_infer": not args.disable_osm_fallback
    }
    
    # Override swept parameter
    params[param_name] = param_value

    # Keep l_scale aligned when sweeping resolution
    if param_name == "resolution":
        base_ratio = args.l_scale / args.resolution
        params["l_scale"] = param_value * base_ratio
    
    bki = osm_bki_cpp.PyContinuousBKI(
        osm_path=str(osm_path),
        config_path=str(config_path),
        resolution=params["resolution"],
        l_scale=params["l_scale"],
        sigma_0=params["sigma_0"],
        prior_delta=params["prior_delta"],
        height_sigma=params["height_sigma"],
        use_semantic_kernel=True,
        use_spatial_kernel=True,
        num_threads=-1,
        alpha0=params["alpha0"],
        seed_osm_prior=params["seed_osm_prior"],
        osm_prior_strength=params["osm_prior_strength"],
        osm_fallback_in_infer=params["osm_fallback_in_infer"]
    )
    
    start_time = time.time()
    
    # Run through all scans
    for i, scan_path in enumerate(scan_files):
        stem = scan_path.stem
        frame = get_frame_number(stem)
        label_path = find_label_file(label_dir, stem)
        
        if not label_path:
            continue
            
        points_xyz, _ = load_scan(str(scan_path))
        labels = load_labels(label_path)
        
        if poses is not None and frame is not None and frame in poses:
            points_xyz = transform_points_to_world(points_xyz, poses[frame])
        else:
            continue
            
        bki.update(labels, points_xyz)
        
    # Evaluate at the end
    acc, miou = run_evaluation(bki, scan_files, label_dir, gt_dir, poses)
    elapsed = time.time() - start_time
    
    print(f"Acc: {acc*100:.2f}%, mIoU: {miou*100:.2f}% ({elapsed:.1f}s)")
            
    return {
        "parameter": param_name,
        "value": param_value,
        "accuracy": acc,
        "miou": miou,
        "map_size": bki.get_size(),
        "time_elapsed": elapsed
    }

def main():
    parser = argparse.ArgumentParser(description="Parameter Sensitivity Benchmark")
    
    # Data paths
    parser.add_argument("--scan-dir", default="../example_data/mcd-data/data", help="Directory of .bin scans")
    parser.add_argument("--label-dir", default="../example_data/mcd-data/labels_predicted", help="Directory of input labels")
    parser.add_argument("--gt-dir", default="../example_data/mcd-data/labels_groundtruth", help="Directory of GT labels")
    parser.add_argument("--osm", default="../example_data/mcd-data/kth_day_06_osm_geometries.bin", help="Path to OSM geometries")
    parser.add_argument("--pose", default="../example_data/mcd-data/pose_inW.csv", help="Pose CSV file")
    parser.add_argument("--config", default="../configs/mcd_config.yaml", help="Path to YAML config")
    
    # Default BKI Parameters (Base configuration)
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
    
    # Benchmark Options
    parser.add_argument("--limit-scans", type=int, default=None, help="Limit number of scans for faster sweeping")
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
        
    if args.limit_scans:
        print(f"Limiting to first {args.limit_scans} scans.")
        scan_files = scan_files[:args.limit_scans]
        
    # Define Sweeps
    sweeps = {
        "resolution": [0.2, 0.4, 0.6, 0.8, 1.0],
        "l_scale": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        "sigma_0": [0.25, 0.5, 1.0, 2.0, 4.0],
        "prior_delta": [0.2, 0.35, 0.5, 0.75, 1.0],
        "alpha0": [0.1, 0.5, 1.0, 2.0, 5.0]
    }
    
    all_results = []
    
    print(f"\n🚀 Starting Parameter Sensitivity Benchmark")
    print(f"   Total Scans: {len(scan_files)}")
    print("-" * 60)
    
    for param_name, values in sweeps.items():
        print(f"\n--- Sweeping {param_name} ---")
        for value in values:
            res = run_config(param_name, value, scan_files, label_dir, gt_dir, poses, osm_path, config_path, args)
            all_results.append(res)
            
    # Write Results
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = script_dir / "parameter_sensitivity_results"
        output_dir.mkdir(exist_ok=True)
        output_csv = output_dir / f"parameter_sensitivity_{timestamp}.csv"
    else:
        output_csv = Path(args.output)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        
    print(f"\nWriting results to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ["parameter", "value", "accuracy", "miou", "map_size", "time_elapsed"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
        
    print("\nDone.")

if __name__ == "__main__":
    main()
