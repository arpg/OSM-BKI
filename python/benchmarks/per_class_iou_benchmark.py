#!/usr/bin/env python3
"""
Per-Class IoU Benchmark.

Evaluates the semantic segmentation performance broken down by class.
This helps identify which classes benefit most from the BKI process and OSM priors.

Key features:
- Computes Intersection over Union (IoU) for each class.
- Compares baseline (input labels) vs BKI refined labels.
- Aggregates results across multiple scans to provide robust statistics.
"""

import sys
import argparse
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
import time
from collections import defaultdict

# Add parent directory to path to import osm_bki_cpp
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import osm_bki_cpp

# Import benchmark utilities
from benchmark_utils import (
    load_poses, transform_points_to_world, load_scan, load_labels,
    find_label_file, get_frame_number, check_files_exist
)

def compute_confusion_matrix(pred, gt, num_classes, ignore_label=0):
    """Compute confusion matrix for a single scan."""
    mask = (gt != ignore_label)
    pred = pred[mask]
    gt = gt[mask]
    
    # Use bincount for fast confusion matrix computation
    # valid classes are 1..num_classes (assuming 0 is ignore)
    # We map them to 0..num_classes-1 for matrix indexing if needed, 
    # but here we just use the raw values and a dictionary or large array
    
    # A simple way for arbitrary labels is to use a 2D histogram or manual accumulation
    # Since class IDs might be sparse (e.g. SemanticKITTI), we use a dict of dicts or similar
    # But for speed with numpy, we can map unique classes to indices
    
    # Let's stick to the logic in compute_metrics but keep the raw counts
    # intersection[c] = TP, union[c] = TP + FP + FN
    # We need TP, FP, FN for each class to aggregate
    
    classes = np.unique(np.concatenate([pred, gt]))
    stats = {}
    
    for c in classes:
        c = int(c)
        pred_mask = (pred == c)
        gt_mask = (gt == c)
        
        tp = np.sum(pred_mask & gt_mask)
        fp = np.sum(pred_mask & ~gt_mask)
        fn = np.sum(~pred_mask & gt_mask)
        
        stats[c] = {"tp": tp, "fp": fp, "fn": fn}
        
    return stats

def aggregate_iou(total_stats):
    """Compute IoU from aggregated TP, FP, FN counts."""
    ious = {}
    for c, counts in total_stats.items():
        intersection = counts["tp"]
        union = counts["tp"] + counts["fp"] + counts["fn"]
        
        if union > 0:
            ious[c] = intersection / union
        else:
            ious[c] = float('nan')
            
    return ious

def main():
    parser = argparse.ArgumentParser(description="Per-Class IoU Benchmark")
    
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
    
    # BKI Variants to test
    variants = {
        "Baseline": None, # Just evaluate inputs
        "BKI_Only": {"use_semantic_kernel": True, "seed_osm_prior": False, "osm_prior_strength": 0.0, "osm_fallback_in_infer": False},
        "BKI_OSM": {"use_semantic_kernel": True, "seed_osm_prior": args.seed_osm_prior, "osm_prior_strength": args.osm_prior_strength, "osm_fallback_in_infer": not args.disable_osm_fallback},
        "NoSem_BKI": {"use_semantic_kernel": False, "seed_osm_prior": False, "osm_prior_strength": 0.0, "osm_fallback_in_infer": False},
        "NoSem_BKI_OSM": {"use_semantic_kernel": False, "seed_osm_prior": args.seed_osm_prior, "osm_prior_strength": args.osm_prior_strength, "osm_fallback_in_infer": not args.disable_osm_fallback}
    }
    
    # Initialize BKI instances for each variant
    bki_instances = {}
    for name, params in variants.items():
        if name == "Baseline": continue
        
        bki_instances[name] = osm_bki_cpp.PyContinuousBKI(
            osm_path=str(osm_path),
            config_path=str(config_path),
            resolution=args.resolution,
            l_scale=args.l_scale,
            sigma_0=args.sigma_0,
            prior_delta=args.prior_delta,
            height_sigma=args.height_sigma,
            use_semantic_kernel=params["use_semantic_kernel"],
            use_spatial_kernel=True,
            num_threads=-1,
            alpha0=args.alpha0,
            seed_osm_prior=params["seed_osm_prior"],
            osm_prior_strength=params["osm_prior_strength"],
            osm_fallback_in_infer=params["osm_fallback_in_infer"]
        )
    
    # Stats accumulators: variant -> class_id -> {tp, fp, fn}
    all_stats = {name: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}) for name in variants.keys()}
    
    print(f"\n🚀 Starting Per-Class IoU Benchmark")
    print(f"   Total Scans: {len(scan_files)}")
    print(f"   Variants: {list(variants.keys())}")
    print("-" * 60)
    
    for i, scan_path in enumerate(scan_files):
        stem = scan_path.stem
        frame = get_frame_number(stem)
        
        # Load Data
        label_path = find_label_file(label_dir, stem)
        gt_path = find_label_file(gt_dir, stem)
        
        if not label_path or not gt_path:
            continue
            
        points_xyz, _ = load_scan(str(scan_path))
        input_labels = load_labels(label_path)
        gt_labels = load_labels(gt_path)
        
        # Transform
        if poses is not None and frame is not None and frame in poses:
            points_xyz = transform_points_to_world(points_xyz, poses[frame])
        else:
            continue
            
        # Process each variant
        for name, bki in bki_instances.items():
            # Update
            bki.update(input_labels, points_xyz)
            # Infer
            pred_labels = bki.infer(points_xyz)
            
            # Compute stats
            n = min(len(pred_labels), len(gt_labels))
            if n > 0:
                scan_stats = compute_confusion_matrix(pred_labels[:n], gt_labels[:n], num_classes=30)
                for c, s in scan_stats.items():
                    all_stats[name][c]["tp"] += s["tp"]
                    all_stats[name][c]["fp"] += s["fp"]
                    all_stats[name][c]["fn"] += s["fn"]
                    
        # Baseline (Input Labels)
        n = min(len(input_labels), len(gt_labels))
        if n > 0:
            scan_base_stats = compute_confusion_matrix(input_labels[:n], gt_labels[:n], num_classes=30)
            for c, s in scan_base_stats.items():
                all_stats["Baseline"][c]["tp"] += s["tp"]
                all_stats["Baseline"][c]["fp"] += s["fp"]
                all_stats["Baseline"][c]["fn"] += s["fn"]
            
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} scans...", flush=True)
            
    # Compute Final IoUs and Accuracy per variant
    final_metrics = {}
    for name, stats in all_stats.items():
        # mIoU
        ious = aggregate_iou(stats)
        miou = np.nanmean(list(ious.values()))
        
        # Accuracy
        total_tp = sum(s["tp"] for s in stats.values())
        total_count = sum(s["tp"] + s["fn"] for s in stats.values())
        accuracy = total_tp / total_count if total_count > 0 else 0.0
        
        final_metrics[name] = {"miou": miou, "accuracy": accuracy, "ious": ious}
    
    # Prepare Results Table (Per-Class)
    # Collect all unique class IDs across all variants
    all_classes = set()
    for m in final_metrics.values():
        all_classes.update(m["ious"].keys())
    all_classes = sorted(list(all_classes))
    
    results = []
    for c in all_classes:
        row = {"class_id": c}
        base_iou = final_metrics["Baseline"]["ious"].get(c, 0.0)
        row["Baseline"] = base_iou
        
        for name in variants.keys():
            if name == "Baseline": continue
            val = final_metrics[name]["ious"].get(c, 0.0)
            row[name] = val
            row[f"{name}_Imp"] = val - base_iou
            
        results.append(row)
        
    # Write Per-Class Results and Method Summary (separate folders)
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        per_class_dir = script_dir / "per_class_results"
        per_class_dir.mkdir(exist_ok=True)
        output_csv = per_class_dir / f"per_class_iou_{timestamp}.csv"

        summary_dir = script_dir / "method_summary_results"
        summary_dir.mkdir(exist_ok=True)
        summary_csv = summary_dir / f"method_summary_{timestamp}.csv"
    else:
        output_csv = Path(args.output)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary_dir = output_csv.parent / "method_summary_results"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_csv = summary_dir / f"method_summary_{output_csv.stem.split('_')[-1]}.csv"
        
    print(f"\nWriting per-class results to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        # Dynamic fieldnames based on variants
        fieldnames = ["class_id", "Baseline"]
        for name in variants.keys():
            if name == "Baseline": continue
            fieldnames.append(name)
            fieldnames.append(f"{name}_Imp")
            
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Write Summary Results (Method Comparison)
    print(f"Writing method summary to {summary_csv}...")
    with open(summary_csv, 'w', newline='') as f:
        fieldnames = ["Method", "mIoU", "Accuracy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Define order
        ordered_names = ["Baseline", "NoSem_BKI", "NoSem_BKI_OSM", "BKI_Only", "BKI_OSM"]
        # Add any others not in ordered list
        for name in final_metrics.keys():
            if name not in ordered_names:
                ordered_names.append(name)
                
        for name in ordered_names:
            if name in final_metrics:
                writer.writerow({
                    "Method": name,
                    "mIoU": final_metrics[name]["miou"],
                    "Accuracy": final_metrics[name]["accuracy"]
                })
        
    # Print Summary to Console
    print("\nMethod Comparison Summary:")
    print(f"{'Method':<15} {'mIoU':<8} {'Acc':<8}")
    print("-" * 33)
    for name in ordered_names:
        if name in final_metrics:
            m = final_metrics[name]
            print(f"{name:<15} {m['miou']*100:<8.1f} {m['accuracy']*100:<8.1f}")

    print("\nPer-Class IoU Summary (Selected Variants):")
    headers = ["Class", "Base", "BKI", "BKI+OSM"]
    print(f"{headers[0]:<6} {headers[1]:<8} {headers[2]:<8} {headers[3]:<8}")
    print("-" * 40)
    for r in results:
        print(f"{r['class_id']:<6} {r['Baseline']*100:<8.1f} {r['BKI_Only']*100:<8.1f} {r['BKI_OSM']*100:<8.1f}")
        
    print("\nDone.")

if __name__ == "__main__":
    main()
