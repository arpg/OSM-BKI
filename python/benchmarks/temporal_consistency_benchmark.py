#!/usr/bin/env python3
"""
Temporal Consistency Benchmark.

Measures the stability of the map over time by tracking "label flips".
A label flip occurs when a voxel's predicted class changes from one scan to the next.
High flip rate indicates instability; low flip rate indicates convergence.

This benchmark evaluates a set of fixed test points (e.g., from the first scan)
at every time step to see how their predictions evolve.
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

def main():
    parser = argparse.ArgumentParser(description="Temporal Consistency Benchmark")
    
    # Data paths
    parser.add_argument("--scan-dir", default="../example_data/mcd-data/data", help="Directory of .bin scans")
    parser.add_argument("--label-dir", default="../example_data/mcd-data/labels_predicted", help="Directory of input labels")
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
    
    # Benchmark Options
    parser.add_argument("--test-points-scan", default="0000000011", help="Stem of scan to use as fixed test points")
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
    osm_path = (script_dir / args.osm).resolve()
    pose_path = (script_dir / args.pose).resolve()
    config_path = (script_dir / args.config).resolve()
    
    if not check_files_exist({
        "Scan Dir": scan_dir,
        "Label Dir": label_dir,
        "OSM": osm_path,
        "Pose": pose_path,
        "Config": config_path
    }):
        return 1
        
    # Load Poses
    poses = load_poses(pose_path)
    scan_files = sorted(scan_dir.glob("*.bin"))
    
    # Load Test Points (Fixed set to track)
    test_scan_path = scan_dir / f"{args.test_points_scan}.bin"
    if not test_scan_path.exists():
        print(f"Test scan {test_scan_path} not found.")
        return 1
        
    print(f"Loading fixed test points from {test_scan_path.name}...")
    test_points, _ = load_scan(str(test_scan_path))
    
    # Transform test points to world frame
    frame = get_frame_number(args.test_points_scan)
    if frame in poses:
        test_points = transform_points_to_world(test_points, poses[frame])
    else:
        print("Pose for test scan not found.")
        return 1
        
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
    
    print(f"\n🚀 Starting Temporal Consistency Benchmark")
    print(f"   Tracking {len(test_points)} points over {len(scan_files)} scans.")
    print("-" * 60)
    
    results = []
    prev_labels = None
    
    for i, scan_path in enumerate(scan_files):
        stem = scan_path.stem
        frame = get_frame_number(stem)
        
        # Load Data
        label_path = find_label_file(label_dir, stem)
        if not label_path:
            continue
            
        points_xyz, _ = load_scan(str(scan_path))
        labels = load_labels(label_path)
        
        if poses is not None and frame is not None and frame in poses:
            points_xyz = transform_points_to_world(points_xyz, poses[frame])
        else:
            continue
            
        # Update Map
        bki.update(labels, points_xyz)
        
        # Infer on Test Points
        curr_labels = bki.infer(test_points)
        curr_labels = np.array(curr_labels, dtype=np.uint32)
        
        # Compute Flip Rate
        flip_rate = 0.0
        flips = 0
        
        if prev_labels is not None:
            flips = np.sum(curr_labels != prev_labels)
            flip_rate = flips / len(test_points)
            
        print(f"Scan {i+1}: Flip Rate = {flip_rate*100:.2f}% ({flips} flips)")
        
        results.append({
            "scan_idx": i + 1,
            "scan_name": stem,
            "flip_rate": flip_rate,
            "total_flips": flips,
            "map_size": bki.get_size()
        })
        
        prev_labels = curr_labels
        
    # Write Results
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = script_dir / "temporal_consistency_results"
        output_dir.mkdir(exist_ok=True)
        output_csv = output_dir / f"temporal_consistency_{timestamp}.csv"
    else:
        output_csv = Path(args.output)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        
    print(f"\nWriting results to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ["scan_idx", "scan_name", "flip_rate", "total_flips", "map_size"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print("\nDone.")

if __name__ == "__main__":
    main()
