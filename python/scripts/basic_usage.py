#!/usr/bin/env python3
"""
Basic usage example for Composite BKI C++ library.

This demonstrates how to use the high-performance C++ implementation
for semantic-spatial Bayesian Kernel Inference on LiDAR point clouds.
Paths are read from OSM-S-BKI/python/configs/mcd_config.yaml; if those
paths do not exist, the script falls back to ../example_data/mcd_scan.
"""

import sys
from pathlib import Path

# Ensure composite_bki_cpp is found: prefer python/build/ (if present), then python/
_script_dir = Path(__file__).resolve().parent
_python_dir = _script_dir.parent
_build_dir = _python_dir / "build"
if _build_dir.is_dir() and str(_build_dir) not in sys.path:
    sys.path.insert(0, str(_build_dir))
if str(_python_dir) not in sys.path:
    sys.path.insert(0, str(_python_dir))

import numpy as np
import composite_bki_cpp
import os

try:
    import yaml
except ImportError:
    yaml = None

# Default config path (python/configs/mcd_config.yaml)
def _config_path():
    """Find mcd_config.yaml (python/configs/), whether run from python/ or repo root."""
    p = Path(__file__).resolve().parent
    for _ in range(4):
        c = p / "configs" / "mcd_config.yaml"
        if c.exists():
            return c
        p = p.parent
    return Path(__file__).resolve().parent.parent / "configs" / "mcd_config.yaml"


def _parse_config_root_and_paths(config_path):
    """Read dataset_root_path and path keys from config; return None if not found."""
    cfg = None
    if yaml is not None:
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        except Exception:
            pass
    if not cfg or not isinstance(cfg, dict):
        # Minimal parse without PyYAML: look for dataset_root_path line
        try:
            with open(config_path) as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("dataset_root_path:") and "dataset_root_path" not in (cfg or {}):
                        val = line.split(":", 1)[1].strip()
                        if "#" in val:
                            val = val.split("#", 1)[0].strip()
                        val = val.strip('"').strip("'")
                        if val:
                            cfg = {"dataset_root_path": val}
                        break
        except Exception:
            pass
    if not cfg or not cfg.get("dataset_root_path"):
        return None
    return cfg


def load_paths_from_config():
    """
    Load data paths from mcd_config.yaml. Returns a dict with keys:
    config_path, data_dir, lidar_path, label_path, ground_truth_path, osm_path.
    Uses config paths whenever the config file exists and contains dataset_root_path;
    only falls back to example_data when the config is missing or unreadable.
    """
    config_path = _config_path()
    # example_data is next to python/ (parent of configs/)
    fallback_dir = config_path.parent.parent / "example_data" / "mcd_scan"
    fallback = {
        "config_path": str(config_path),
        "lidar_path": fallback_dir / "0000000011_transformed.bin",
        "label_path": fallback_dir / "0000000011_transformed_noisy.bin",
        "ground_truth_path": fallback_dir / "0000000011_transformed.bin",
        "osm_path": fallback_dir / "kth_day_06_osm_geometries.bin",
        "data_dir": fallback_dir,
    }
    if not config_path.exists():
        return fallback
    cfg = _parse_config_root_and_paths(config_path)
    if not cfg:
        return fallback
    root = Path(cfg.get("dataset_root_path"))
    lidar_data_path = cfg.get("lidar_data_path", "kth_day_06/lidar_bin/data/")
    pred_labels_path = cfg.get("pred_labels_path", "kth_day_06/inferred_labels/cenet_semkitti/")
    gt_labels_path = cfg.get("gt_labels_path", "kth_day_06/gt_labels/")
    osm_file = cfg.get("osm_file", "kth.osm")
    lidar_dir = root / lidar_data_path.rstrip("/")
    pred_label_dir = root / pred_labels_path.rstrip("/")
    gt_label_dir = root / gt_labels_path.rstrip("/")
    osm_path = root / osm_file
    scan_id = cfg.get("example_scan_id", "0000000011")
    lidar_path = lidar_dir / f"{scan_id}.bin"
    if lidar_path.exists():
        pass
    elif lidar_dir.exists():
        bins = sorted(lidar_dir.glob("*.bin"))
        if bins:
            lidar_path = bins[0]
            scan_id = lidar_path.stem
    label_path = pred_label_dir / f"{scan_id}.bin"
    ground_truth_path = gt_label_dir / f"{scan_id}.bin"
    return {
        "config_path": str(config_path),
        "data_dir": lidar_dir,
        "lidar_path": lidar_path,
        "label_path": label_path,
        "ground_truth_path": ground_truth_path,
        "osm_path": osm_path,
    }


def check_files_exist(files_dict):
    """
    Check if required files exist.
    
    Args:
        files_dict: Dictionary mapping description to Path object
        
    Returns:
        bool: True if all files exist, False otherwise
    """
    missing_files = [desc for desc, path in files_dict.items() if not path.exists()]
    
    if missing_files:
        print("⚠ Missing required files:")
        for desc in missing_files:
            print(f"  - {desc}: {files_dict[desc]}")
        print("\nPlease ensure all example data files are available.")
        return False
    return True


def example_basic_refinement():
    """Basic example: Refine noisy semantic labels using OSM priors."""
    print("=" * 70)
    print("Example 1: Basic Label Refinement")
    print("=" * 70)
    
    paths = load_paths_from_config()
    config_path = paths["config_path"]
    data_dir = paths["data_dir"]
    lidar_path = paths["lidar_path"]
    label_path = paths["label_path"]
    ground_truth_path = paths["ground_truth_path"]
    osm_path = paths["osm_path"]
    
    if not check_files_exist({
        "LiDAR data": lidar_path,
        "Noisy labels": label_path,
        "Ground truth": ground_truth_path,
        "OSM map": osm_path
    }):
        return
    
    print(f"📂 Loading data from {data_dir}")
    print(f"  LiDAR: {lidar_path.name}")
    print(f"  Noisy Labels: {label_path.name}")
    print(f"  Ground Truth: {ground_truth_path.name}")
    print(f"  OSM: {osm_path.name}")
    print()
    
    refined_labels = composite_bki_cpp.run_pipeline(
        lidar_path=str(lidar_path),
        label_path=str(label_path),
        osm_path=str(osm_path),
        config_path=config_path,
        ground_truth_path=str(ground_truth_path),
        output_path="refined_labels.labels",
        l_scale=3.0,
        sigma_0=1.0,
        prior_delta=5.0,
        alpha_0=0.01,
        num_threads=-1  # Use all available cores
    )
    
    print()
    print(f"✓ Successfully processed {len(refined_labels)} points")
    print(f"✓ Output saved to: refined_labels.labels")
    print()


def example_custom_parameters():
    """Example with custom BKI parameters."""
    print("=" * 70)
    print("Example 2: Custom Parameters")
    print("=" * 70)
    
    paths = load_paths_from_config()
    config_path = paths["config_path"]
    lidar_path = paths["lidar_path"]
    label_path = paths["label_path"]
    ground_truth_path = paths["ground_truth_path"]
    osm_path = paths["osm_path"]
    
    if not check_files_exist({
        "LiDAR data": lidar_path,
        "Noisy labels": label_path,
        "Ground truth": ground_truth_path,
        "OSM map": osm_path
    }):
        return
    
    print("Testing different parameter combinations...")
    print()
    
    gt_path = str(ground_truth_path) if ground_truth_path.exists() else None
    
    # Test with tighter spatial kernel
    print("Configuration 1: Tighter spatial kernel (l_scale=1.5)")
    refined_1 = composite_bki_cpp.run_pipeline(
        lidar_path=str(lidar_path),
        label_path=str(label_path),
        osm_path=str(osm_path),
        config_path=config_path,
        ground_truth_path=gt_path,
        output_path="refined_tight.labels",
        l_scale=1.5,  # Tighter spatial influence
        alpha_0=0.01
    )
    print(f"  Processed {len(refined_1)} points")
    print()
    
    # Test with wider spatial kernel
    print("Configuration 2: Wider spatial kernel (l_scale=5.0)")
    refined_2 = composite_bki_cpp.run_pipeline(
        lidar_path=str(lidar_path),
        label_path=str(label_path),
        osm_path=str(osm_path),
        config_path=config_path,
        ground_truth_path=gt_path,
        output_path="refined_wide.labels",
        l_scale=5.0,  # Wider spatial influence
        alpha_0=0.01
    )
    print(f"  Processed {len(refined_2)} points")
    print()


def example_advanced_api():
    """Example using the advanced PySemanticBKI class directly."""
    print("=" * 70)
    print("Example 3: Advanced API - Direct Class Usage")
    print("=" * 70)
    
    paths = load_paths_from_config()
    config_path = paths["config_path"]
    lidar_path = paths["lidar_path"]
    label_path = paths["label_path"]
    ground_truth_path = paths["ground_truth_path"]
    osm_path = paths["osm_path"]
    
    if not check_files_exist({
        "LiDAR data": lidar_path,
        "Noisy labels": label_path,
        "Ground truth": ground_truth_path,
        "OSM map": osm_path
    }):
        return
    
    print("Loading data manually...")
    
    # Load LiDAR scan
    scan = np.fromfile(str(lidar_path), dtype=np.float32).reshape((-1, 4))
    points = scan[:, :3].astype(np.float32)
    print(f"  Loaded {len(points)} points")
    
    # Load noisy labels
    labels_raw = np.fromfile(str(label_path), dtype=np.uint32)
    labels = (labels_raw & 0xFFFF).astype(np.uint32)
    print(f"  Loaded {len(labels)} noisy labels")
    
    # Load ground truth
    gt_raw = np.fromfile(str(ground_truth_path), dtype=np.uint32)
    gt_labels = (gt_raw & 0xFFFF).astype(np.uint32)
    print(f"  Loaded {len(gt_labels)} ground truth labels")
    
    print()
    print("Initializing Semantic BKI processor...")
    bki = composite_bki_cpp.PySemanticBKI(
        osm_path=str(osm_path),
        config_path=config_path,
        l_scale=3.0,
        sigma_0=1.0,
        prior_delta=5.0,
        num_threads=-1
    )
    print("  ✓ BKI initialized")
    
    # Process a subset for demonstration
    subset_size = 10000
    print()
    print(f"Processing first {subset_size} points...")
    refined_labels = bki.process_point_cloud(
        points[:subset_size], 
        labels[:subset_size], 
        alpha_0=0.01
    )
    
    # Analyze results
    num_changed = np.sum(refined_labels != labels[:subset_size])
    pct_changed = (num_changed / len(refined_labels)) * 100
    
    print()
    print(f"✓ Processing complete!")
    print(f"  Points processed: {len(refined_labels)}")
    print(f"  Labels changed: {num_changed} ({pct_changed:.2f}%)")
    
    # Show some statistics
    unique_before = len(np.unique(labels[:subset_size]))
    unique_after = len(np.unique(refined_labels))
    print(f"  Unique labels before: {unique_before}")
    print(f"  Unique labels after: {unique_after}")
    
    # Evaluate against ground truth
    print()
    print("Evaluating against ground truth...")
    metrics = bki.evaluate_metrics(refined_labels, gt_labels[:subset_size])
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  mIoU: {metrics['miou']*100:.2f}%")
    
    # Save results
    refined_labels.tofile("refined_advanced.labels")
    print(f"  Saved to: refined_advanced.labels")
    print()


def example_with_evaluation():
    """Example demonstrating the improvement from noisy to refined labels."""
    print("=" * 70)
    print("Example 4: Evaluation - Comparing Noisy vs Refined")
    print("=" * 70)
    
    paths = load_paths_from_config()
    config_path = paths["config_path"]
    lidar_path = paths["lidar_path"]
    noisy_label_path = paths["label_path"]
    gt_path = paths["ground_truth_path"]
    osm_path = paths["osm_path"]
    
    if not check_files_exist({
        "LiDAR data": lidar_path,
        "Noisy labels": noisy_label_path,
        "Ground truth": gt_path,
        "OSM map": osm_path
    }):
        return
    
    print("This example compares:")
    print("  • Noisy input labels (before refinement)")
    print("  • Refined labels (after BKI refinement)")
    print("  • Ground truth labels (target)")
    print()
    
    print("Running pipeline with evaluation...")
    refined_labels = composite_bki_cpp.run_pipeline(
        lidar_path=str(lidar_path),
        label_path=str(noisy_label_path),
        osm_path=str(osm_path),
        config_path=config_path,
        ground_truth_path=str(gt_path),  # Compare against clean labels
        output_path="refined_evaluated.labels"
    )
    
    print()
    print(f"✓ Processed {len(refined_labels)} points with evaluation")
    print("✓ Check console output above for accuracy improvements!")
    print()


def main():
    """Run all examples."""
    print()
    print("=" * 70)
    print("  Composite BKI C++ Library - Usage Examples")
    print("=" * 70)
    print()
    print("This script demonstrates various ways to use the Composite BKI")
    print("C++ library for semantic label refinement of LiDAR point clouds.")
    print()
    
    # Run examples
    examples = [
        ("Basic Refinement", example_basic_refinement),
        ("Custom Parameters", example_custom_parameters),
        ("Advanced API", example_advanced_api),
        ("With Evaluation", example_with_evaluation),
    ]
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue
    
    print("=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  • Modify parameters to see how they affect results")
    print("  • Try with your own data")
    print("  • Create custom config files for your label format")
    print("  • Use the CLI tool: composite-bki --help")
    print()


if __name__ == "__main__":
    main()
