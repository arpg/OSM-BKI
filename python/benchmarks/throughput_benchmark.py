#!/usr/bin/env python3
"""
Throughput and Scaling Benchmark.

Evaluates the computational performance of the Composite BKI system.
Measures execution time and memory usage across different dimensions:
1. Thread Scaling: Speedup vs number of OpenMP threads.
2. Point Density: Throughput (points/sec) vs input point cloud size.
3. Resolution: Performance and memory impact of voxel resolution.
4. OSM Rasterization: Time taken to build the OSM prior raster (constructor time).

This benchmark typically runs on a single representative scan to ensure controlled conditions.
"""

import sys
import argparse
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import os

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False
    print("⚠️  psutil not found. Memory measurements will be skipped. (pip install psutil)")

# Add parent directory to path to import osm_bki_cpp
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import osm_bki_cpp

# Import benchmark utilities
from benchmark_utils import (
    load_scan, load_labels, find_label_file, check_files_exist
)

def measure_execution(func, *args, **kwargs):
    """Measure execution time of a function call."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return result, end - start

def get_process_memory():
    """Get current process memory usage in MB."""
    if not _HAS_PSUTIL:
        return 0.0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def subsample_points(points, labels, target_count):
    """Randomly subsample points to a target count."""
    if len(points) <= target_count:
        return points, labels
    
    indices = np.random.choice(len(points), target_count, replace=False)
    return points[indices], labels[indices]

def run_thread_scaling(scan_path, label_path, osm_path, config_path, args, scan_name=None):
    """Benchmark performance vs number of threads."""
    scan_name = scan_name or scan_path.stem
    print(f"\n--- Thread Scaling Benchmark ({scan_name}) ---")
    
    points_raw, _ = load_scan(str(scan_path))
    labels_raw = load_labels(label_path)
    
    # Use full points for thread scaling
    points = points_raw
    labels = labels_raw
    
    results = []
    threads_to_test = [1, 2, 4, 8, 12, 16, -1] # -1 is all available
    
    for n_threads in threads_to_test:
        print(f"Testing with {n_threads} threads...", end=" ", flush=True)
        
        # Initialize (exclude from timing)
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
            num_threads=n_threads,
            alpha0=args.alpha0,
            seed_osm_prior=args.seed_osm_prior,
            osm_prior_strength=args.osm_prior_strength,
            osm_fallback_in_infer=not args.disable_osm_fallback
        )
        
        # Warmup (optional, but good for JIT/cache)
        # bki.update(labels[:100], points[:100]) 
        
        # Measure Update
        _, update_time = measure_execution(bki.update, labels, points)
        
        # Measure Infer
        _, infer_time = measure_execution(bki.infer, points)
        
        print(f"Update: {update_time:.4f}s, Infer: {infer_time:.4f}s")
        
        results.append({
            "experiment": "thread_scaling",
            "scan": scan_name,
            "threads": n_threads,
            "points": len(points),
            "resolution": args.resolution,
            "update_time": update_time,
            "infer_time": infer_time,
            "update_throughput": len(points) / update_time,
            "infer_throughput": len(points) / infer_time,
            "map_size": bki.get_size()
        })
        
    return results

def run_point_scaling(scan_path, label_path, osm_path, config_path, args, scan_name=None):
    """Benchmark performance vs point cloud size."""
    scan_name = scan_name or scan_path.stem
    print(f"\n--- Point Density Scaling Benchmark ({scan_name}) ---")
    
    points_full, _ = load_scan(str(scan_path))
    labels_full = load_labels(label_path)
    
    results = []
    # Sizes: 10k, 25k, 50k, 100k, Full (~120k usually)
    sizes = [10000, 25000, 50000, 75000, 100000, len(points_full)]
    sizes = sorted(list(set([s for s in sizes if s <= len(points_full)])))
    
    for size in sizes:
        print(f"Testing with {size} points...", end=" ", flush=True)
        
        points, labels = subsample_points(points_full, labels_full, size)
        
        # Initialize
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
            num_threads=-1, # Max performance
            alpha0=args.alpha0,
            seed_osm_prior=args.seed_osm_prior,
            osm_prior_strength=args.osm_prior_strength,
            osm_fallback_in_infer=not args.disable_osm_fallback
        )
        
        _, update_time = measure_execution(bki.update, labels, points)
        _, infer_time = measure_execution(bki.infer, points)
        
        print(f"Update: {update_time:.4f}s, Infer: {infer_time:.4f}s")
        
        results.append({
            "experiment": "point_scaling",
            "scan": scan_name,
            "threads": -1,
            "points": size,
            "resolution": args.resolution,
            "update_time": update_time,
            "infer_time": infer_time,
            "update_throughput": size / update_time,
            "infer_throughput": size / infer_time,
            "map_size": bki.get_size()
        })
        
    return results

def run_resolution_scaling(scan_path, label_path, osm_path, config_path, args, scan_name=None):
    """Benchmark performance and memory vs resolution."""
    scan_name = scan_name or scan_path.stem
    print(f"\n--- Resolution Scaling Benchmark ({scan_name}) ---")
    
    points, _ = load_scan(str(scan_path))
    labels = load_labels(label_path)
    
    results = []
    resolutions = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    for res in resolutions:
        print(f"Testing resolution {res}m...", end=" ", flush=True)
        
        start_mem = get_process_memory()
        
        # Measure Constructor (OSM Raster Build Time)
        # Keep l_scale/resolution ratio fixed during resolution sweep
        effective_l_scale = args.l_scale * (res / args.resolution)
        start_init = time.perf_counter()
        bki = osm_bki_cpp.PyContinuousBKI(
            osm_path=str(osm_path),
            config_path=str(config_path),
            resolution=res,
            l_scale=effective_l_scale,
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
        init_time = time.perf_counter() - start_init
        
        _, update_time = measure_execution(bki.update, labels, points)
        _, infer_time = measure_execution(bki.infer, points)
        
        end_mem = get_process_memory()
        mem_diff = end_mem - start_mem
        
        print(f"Init: {init_time:.3f}s, Update: {update_time:.3f}s, Mem: +{mem_diff:.1f}MB")
        
        results.append({
            "experiment": "resolution_scaling",
            "scan": scan_name,
            "threads": -1,
            "points": len(points),
            "resolution": res,
            "init_time": init_time,
            "update_time": update_time,
            "infer_time": infer_time,
            "update_throughput": len(points) / update_time,
            "infer_throughput": len(points) / infer_time,
            "map_size": bki.get_size(),
            "memory_increase_mb": mem_diff
        })
        
    return results

def get_scan_label_pairs(scan_dir, label_dir):
    """Return list of (scan_path, label_path) for scans that have matching labels."""
    scan_dir = Path(scan_dir)
    label_dir = Path(label_dir)
    pairs = []
    for scan_path in sorted(scan_dir.glob("*.bin")):
        stem = scan_path.stem
        label_path = find_label_file(label_dir, stem)
        if label_path:
            pairs.append((scan_path, Path(label_path)))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Throughput and Scaling Benchmark")
    
    # Data paths (directories)
    parser.add_argument("--scan-dir", default="../example_data/mcd-data/data", help="Directory of .bin scans")
    parser.add_argument("--label-dir", default="../example_data/mcd-data/labels_predicted", help="Directory of label files")
    parser.add_argument("--osm", default="../example_data/mcd-data/kth_day_06_osm_geometries.bin", help="Path to OSM geometries")
    parser.add_argument("--config", default="../configs/mcd_config.yaml", help="Path to YAML config")
    parser.add_argument("--max-scans", type=int, default=None, help="Max scans to use (default: first scan only for throughput)")
    
    # Base BKI Parameters
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
    
    # Output
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
    config_path = (script_dir / args.config).resolve()
    
    if not check_files_exist({
        "Scan Dir": scan_dir,
        "Label Dir": label_dir,
        "OSM": osm_path,
        "Config": config_path
    }):
        return 1

    pairs = get_scan_label_pairs(scan_dir, label_dir)
    if not pairs:
        print(f"❌ No matching scan/label pairs in {scan_dir} and {label_dir}")
        return 1

    max_scans = args.max_scans if args.max_scans is not None else 1
    pairs = pairs[:max_scans]

    all_results = []
    print(f"\n🚀 Starting Throughput Benchmark")
    print(f"   Scans: {len(pairs)} ({', '.join(p[0].name for p in pairs)})")
    print("-" * 60)

    for scan_path, label_path in pairs:
        scan_name = scan_path.stem
        # 1. Thread Scaling
        all_results.extend(run_thread_scaling(scan_path, label_path, osm_path, config_path, args, scan_name))
        # 2. Point Scaling
        all_results.extend(run_point_scaling(scan_path, label_path, osm_path, config_path, args, scan_name))
        # 3. Resolution Scaling
        all_results.extend(run_resolution_scaling(scan_path, label_path, osm_path, config_path, args, scan_name))
    
    # Write Results
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = script_dir / "throughput_results"
        output_dir.mkdir(exist_ok=True)
        output_csv = output_dir / f"throughput_benchmark_{timestamp}.csv"
    else:
        output_csv = Path(args.output)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        
    print(f"\nWriting results to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        fieldnames = [
            "experiment", "scan", "threads", "points", "resolution", 
            "init_time", "update_time", "infer_time", 
            "update_throughput", "infer_throughput", 
            "map_size", "memory_increase_mb"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
        
    print("\nDone.")

if __name__ == "__main__":
    main()
