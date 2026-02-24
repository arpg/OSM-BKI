#!/usr/bin/env python3
"""
Benchmark Composite BKI performance across OSM randomization levels.

This benchmark:
1. Randomizes the OSM map at controlled levels (logic in this script)
2. Runs Composite BKI refinement for each randomized map
3. Evaluates against ground truth (using osm_bki metrics logic)
4. Outputs results to CSV
"""

import numpy as np
import sys
import csv
import struct
import yaml
from pathlib import Path
from datetime import datetime

# Add repository paths for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import osm_bki_cpp

# Category mapping: load_osm_geometries returns by class (config osm_class_map order)
_CLASS_TO_CATEGORY = {
    0: "roads",
    1: "roads",   # parking -> merge into roads
    2: "roads",   # sidewalks -> merge into roads
    3: "grasslands",  # vegetation: grasslands, trees, wood
    4: "buildings",
    5: "grasslands",  # fences -> merge into grasslands
}


def _extract_polygons_from_geometries(points, lines):
    """Extract polygons from load_osm_geometries output (points + line indices)."""
    if not points or not lines:
        return []
    pts = np.array(points)[:, :2]  # drop z
    n_pts = len(pts)
    # Build next[i] = j for each line (i,j)
    next_idx = {}
    for (i, j) in lines:
        i, j = int(i), int(j)
        if 0 <= i < n_pts and 0 <= j < n_pts:
            next_idx[i] = j
    polygons = []
    visited = set()
    for start in range(n_pts):
        if start in visited:
            continue
        cycle = []
        cur = start
        while cur is not None and cur not in visited:
            visited.add(cur)
            cycle.append(cur)
            cur = next_idx.get(cur)
        if cycle and next_idx.get(cycle[-1]) == start:
            poly = [(float(pts[i, 0]), float(pts[i, 1])) for i in cycle]
            if len(poly) >= 3:
                polygons.append(poly)
    return polygons


def load_osm_from_xml(osm_path, config_path):
    """
    Load OSM geometries from .osm XML file via C++ loader.
    Returns dict: category -> list of polygon coords [(x,y), ...].
    """
    data = osm_bki_cpp.load_osm_geometries(str(osm_path), str(config_path), z_offset=0.0)
    result = {cat: [] for cat in ["buildings", "roads", "grasslands", "trees", "wood"]}
    for idx, geom in enumerate(data):
        cat = _CLASS_TO_CATEGORY.get(idx, "grasslands")
        if cat not in result:
            continue
        pts = geom["points"]
        lines = geom["lines"]
        polys = _extract_polygons_from_geometries(pts, lines)
        if cat == "grasslands":
            result["grasslands"].extend(polys)
        else:
            result[cat].extend(polys)
    return result


def load_osm_bin(bin_file):
    """
    Load OSM geometries from binary file.
    
    Returns:
        Dictionary with categories and their polygon coordinates
    """
    if not Path(bin_file).exists():
        raise FileNotFoundError(f"OSM bin file not found: {bin_file}")
    
    data = {}
    categories = ["buildings", "roads", "grasslands", "trees", "wood"]
    
    with open(bin_file, "rb") as f:
        for cat in categories:
            try:
                num_items_bytes = f.read(4)
                if not num_items_bytes:
                    break
                num_items = struct.unpack("I", num_items_bytes)[0]
                
                items = []
                for _ in range(num_items):
                    n_pts_bytes = f.read(4)
                    if not n_pts_bytes:
                        break
                    n_pts = struct.unpack("I", n_pts_bytes)[0]
                    
                    bytes_data = f.read(n_pts * 2 * 4)
                    if len(bytes_data) < n_pts * 2 * 4:
                        break
                    floats = struct.unpack(f"{n_pts * 2}f", bytes_data)
                    poly_coords = list(zip(floats[0::2], floats[1::2]))
                    items.append(poly_coords)
                
                data[cat] = items
            except struct.error as e:
                print(f"Warning: Failed to load {cat}: {e}")
                break
    
    return data


def _get_osm_categories(config_path):
    """Load osm_categories from config YAML (order required by C++ loader)."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("osm_categories", ["buildings", "roads", "grasslands", "trees", "wood"])


def save_osm_bin(data, output_file, config_path=None):
    """
    Save OSM geometries to binary file (format compatible with C++ loadOSMBinary).
    
    Args:
        data: Dictionary with categories and polygon coordinates
        output_file: Path to output .bin file
        config_path: Optional config path for osm_categories order
    """
    if config_path and Path(config_path).exists():
        categories = _get_osm_categories(config_path)
    else:
        categories = ["buildings", "roads", "grasslands", "trees", "wood"]
    
    with open(output_file, "wb") as f:
        for cat in categories:
            items = data.get(cat, [])
            
            # Write number of items
            f.write(struct.pack("I", len(items)))
            
            # Write each polygon
            for poly_coords in items:
                # Write number of points
                f.write(struct.pack("I", len(poly_coords)))
                
                # Write coordinates
                for x, y in poly_coords:
                    f.write(struct.pack("f", x))
                    f.write(struct.pack("f", y))
    
    print(f"Saved randomized OSM data to {output_file}")


def add_coordinate_noise(data, noise_level, noise_std_meters=1.0):
    """
    Add Gaussian noise to all coordinates.
    
    Args:
        data: OSM data dictionary
        noise_level: Probability of adding noise to each point (0.0 to 1.0)
        noise_std_meters: Standard deviation of noise in meters
    
    Returns:
        Noisy OSM data dictionary
    """
    noisy_data = {}
    total_points = 0
    noisy_points = 0
    
    for cat, items in data.items():
        noisy_items = []
        for poly_coords in items:
            noisy_poly = []
            for x, y in poly_coords:
                total_points += 1
                if np.random.random() < noise_level:
                    # Add Gaussian noise
                    noise_x = np.random.normal(0, noise_std_meters)
                    noise_y = np.random.normal(0, noise_std_meters)
                    noisy_poly.append((x + noise_x, y + noise_y))
                    noisy_points += 1
                else:
                    noisy_poly.append((x, y))
            noisy_items.append(noisy_poly)
        noisy_data[cat] = noisy_items
    
    print(f"Added noise to {noisy_points}/{total_points} points ({noisy_points/total_points*100:.1f}%)")
    return noisy_data


def remove_geometries(data, removal_rate):
    """
    Randomly remove entire geometries.
    
    Args:
        data: OSM data dictionary
        removal_rate: Probability of removing each geometry (0.0 to 1.0)
    
    Returns:
        OSM data with some geometries removed
    """
    filtered_data = {}
    total_removed = 0
    total_original = 0
    
    for cat, items in data.items():
        filtered_items = []
        for poly_coords in items:
            total_original += 1
            if np.random.random() >= removal_rate:
                filtered_items.append(poly_coords)
            else:
                total_removed += 1
        filtered_data[cat] = filtered_items
    
    print(f"Removed {total_removed}/{total_original} geometries ({total_removed/total_original*100:.1f}%)")
    return filtered_data


def simplify_geometries(data, simplification_rate, min_points=3):
    """
    Randomly remove vertices from polygons.
    
    Args:
        data: OSM data dictionary
        simplification_rate: Probability of removing each vertex (0.0 to 1.0)
        min_points: Minimum points to keep per polygon
    
    Returns:
        Simplified OSM data
    """
    simplified_data = {}
    total_removed = 0
    total_original = 0
    
    for cat, items in data.items():
        simplified_items = []
        for poly_coords in items:
            total_original += len(poly_coords)
            
            # Always keep first and last points (closed polygon)
            simplified_poly = [poly_coords[0]]
            
            # Randomly keep middle points
            for i in range(1, len(poly_coords) - 1):
                if np.random.random() >= simplification_rate:
                    simplified_poly.append(poly_coords[i])
                else:
                    total_removed += 1
            
            # Add last point
            if len(poly_coords) > 1:
                simplified_poly.append(poly_coords[-1])
            
            # Ensure minimum points
            if len(simplified_poly) >= min_points:
                simplified_items.append(simplified_poly)
            else:
                simplified_items.append(poly_coords)  # Keep original if too simplified
        
        simplified_data[cat] = simplified_items
    
    print(f"Removed {total_removed}/{total_original} vertices ({total_removed/total_original*100:.1f}%)")
    return simplified_data


def scale_geometries(data, scale_factor_range=(0.8, 1.2), apply_prob=1.0):
    """
    Randomly scale geometries around their centroids.
    
    Args:
        data: OSM data dictionary
        scale_factor_range: (min, max) scale factors
        apply_prob: Probability of scaling each geometry
    
    Returns:
        Scaled OSM data
    """
    scaled_data = {}
    num_scaled = 0
    total_geoms = 0
    
    for cat, items in data.items():
        scaled_items = []
        for poly_coords in items:
            total_geoms += 1
            
            if np.random.random() < apply_prob:
                # Calculate centroid
                coords_array = np.array(poly_coords)
                centroid = coords_array.mean(axis=0)
                
                # Random scale factor
                scale = np.random.uniform(*scale_factor_range)
                
                # Scale around centroid
                scaled_coords = []
                for x, y in poly_coords:
                    dx = x - centroid[0]
                    dy = y - centroid[1]
                    new_x = centroid[0] + dx * scale
                    new_y = centroid[1] + dy * scale
                    scaled_coords.append((new_x, new_y))
                
                scaled_items.append(scaled_coords)
                num_scaled += 1
            else:
                scaled_items.append(poly_coords)
        
        scaled_data[cat] = scaled_items
    
    print(f"Scaled {num_scaled}/{total_geoms} geometries ({num_scaled/total_geoms*100:.1f}%)")
    return scaled_data


def shift_geometries(data, max_shift_meters=5.0, apply_prob=1.0):
    """
    Randomly shift geometries in x and y directions.
    
    Args:
        data: OSM data dictionary
        max_shift_meters: Maximum shift distance in meters
        apply_prob: Probability of shifting each geometry
    
    Returns:
        Shifted OSM data
    """
    shifted_data = {}
    num_shifted = 0
    total_geoms = 0
    
    for cat, items in data.items():
        shifted_items = []
        for poly_coords in items:
            total_geoms += 1
            
            if np.random.random() < apply_prob:
                # Random shift
                shift_x = np.random.uniform(-max_shift_meters, max_shift_meters)
                shift_y = np.random.uniform(-max_shift_meters, max_shift_meters)
                
                # Apply shift
                shifted_coords = [(x + shift_x, y + shift_y) for x, y in poly_coords]
                shifted_items.append(shifted_coords)
                num_shifted += 1
            else:
                shifted_items.append(poly_coords)
        
        shifted_data[cat] = shifted_items
    
    print(f"Shifted {num_shifted}/{total_geoms} geometries ({num_shifted/total_geoms*100:.1f}%)")
    return shifted_data


def get_osm_statistics(data):
    """Get statistics about the OSM data."""
    stats = {}
    total_geoms = 0
    total_points = 0
    
    for cat, items in data.items():
        num_geoms = len(items)
        num_points = sum(len(poly) for poly in items)
        stats[cat] = {"geometries": num_geoms, "points": num_points}
        total_geoms += num_geoms
        total_points += num_points
    
    stats["total"] = {"geometries": total_geoms, "points": total_points}
    return stats


def print_statistics(data, title="OSM Statistics"):
    """Print statistics about the OSM data."""
    print(f"\n{title}")
    print("=" * 60)
    
    stats = get_osm_statistics(data)
    
    for cat in ["buildings", "roads", "grasslands", "trees", "wood"]:
        if cat in stats:
            s = stats[cat]
            print(f"  {cat:12s}: {s['geometries']:4d} geometries, {s['points']:6d} points")
    
    print("-" * 60)
    total = stats["total"]
    print(f"  {'TOTAL':12s}: {total['geometries']:4d} geometries, {total['points']:6d} points")
    print("=" * 60)


def randomize_osm(
    input_file,
    output_file,
    config_path=None,
    noise_level=0.0,
    noise_std=1.0,
    removal_rate=0.0,
    simplification_rate=0.0,
    scale_prob=0.0,
    scale_range=(0.8, 1.2),
    shift_prob=0.0,
    max_shift=5.0,
    seed=None,
):
    """
    Apply multiple randomization techniques to OSM data.
    
    Args:
        input_file: Input OSM file (.osm XML or .bin)
        output_file: Output OSM .bin file
        config_path: Config YAML (required for .osm input)
        noise_level: Probability of adding noise to coordinates (0.0-1.0)
        noise_std: Standard deviation of coordinate noise in meters
        removal_rate: Probability of removing entire geometries (0.0-1.0)
        simplification_rate: Probability of removing vertices (0.0-1.0)
        scale_prob: Probability of scaling geometries (0.0-1.0)
        scale_range: (min, max) scale factors for geometries
        shift_prob: Probability of shifting geometries (0.0-1.0)
        max_shift: Maximum shift distance in meters
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
        print(f"Random seed: {seed}")
    
    input_path = Path(input_file)
    print(f"Loading OSM data from {input_file}...")
    if input_path.suffix.lower() == ".osm":
        if not config_path:
            raise ValueError("config_path required when loading .osm file")
        data = load_osm_from_xml(input_file, config_path)
    else:
        data = load_osm_bin(input_file)
    
    print_statistics(data, "Original OSM Data")
    
    # Apply randomizations in sequence
    print("\nApplying randomizations...")
    
    if removal_rate > 0:
        print(f"\n1. Removing geometries (rate={removal_rate})...")
        data = remove_geometries(data, removal_rate)
    
    if simplification_rate > 0:
        print(f"\n2. Simplifying geometries (rate={simplification_rate})...")
        data = simplify_geometries(data, simplification_rate)
    
    if scale_prob > 0:
        print(f"\n3. Scaling geometries (prob={scale_prob}, range={scale_range})...")
        data = scale_geometries(data, scale_range, scale_prob)
    
    if shift_prob > 0:
        print(f"\n4. Shifting geometries (prob={shift_prob}, max={max_shift}m)...")
        data = shift_geometries(data, max_shift, shift_prob)
    
    if noise_level > 0:
        print(f"\n5. Adding coordinate noise (level={noise_level}, std={noise_std}m)...")
        data = add_coordinate_noise(data, noise_level, noise_std)
    
    print_statistics(data, "Randomized OSM Data")
    
    # Save output
    print(f"\nSaving to {output_file}...")
    save_osm_bin(data, output_file, config_path)
    
    return data


def check_files_exist(file_dict):
    """Check if required files exist."""
    missing = [name for name, path in file_dict.items() if not path.exists()]

    if missing:
        print("❌ Missing required files:")
        for name in missing:
            print(f"  - {name}: {file_dict[name]}")
        return False
    return True


def calculate_metrics(pred_labels, gt_labels):
    """
    Calculate accuracy and mIoU (matches osm_bki::compute_metrics exactly).
    """
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
        "accuracy": accuracy,
        "miou": miou,
    }


def compute_randomization_params(level, args):
    """Convert a 0-1 level into concrete randomization parameters."""
    level = float(level)
    level = max(0.0, min(1.0, level))

    noise_level = level * args.max_noise_level
    removal_rate = level * args.max_removal_rate
    simplification_rate = level * args.max_simplification_rate
    scale_prob = level * args.max_scale_prob
    shift_prob = level * args.max_shift_prob
    max_shift = level * args.max_shift

    scale_delta = level * args.scale_delta
    scale_range = (max(0.01, 1.0 - scale_delta), 1.0 + scale_delta)

    return {
        "noise_level": noise_level,
        "noise_std": args.noise_std,
        "removal_rate": removal_rate,
        "simplification_rate": simplification_rate,
        "scale_prob": scale_prob,
        "scale_range": scale_range,
        "shift_prob": shift_prob,
        "max_shift": max_shift,
    }


def run_single_benchmark(
    lidar_path,
    labels_path,
    gt_labels,
    osm_path,
    config_path,
):
    """
    Run a single benchmark iteration with a given OSM map.
    """
    labels_raw = np.fromfile(labels_path, dtype=np.uint32)
    labels = (labels_raw & 0xFFFF).astype(np.uint32)

    metrics_before = calculate_metrics(labels, gt_labels)

    # Using PyContinuousBKI directly
    bki = osm_bki_cpp.PyContinuousBKI(
        osm_path=str(osm_path),
        config_path=str(config_path),
        resolution=1,
        l_scale=3.0,
        sigma_0=1.0,
        prior_delta=5.0,
        height_sigma=0.3,
        use_semantic_kernel=True,
        use_spatial_kernel=False, # Matches original script (use_spatial_kernel=False)
        num_threads=-1,
        alpha0=0.01,
        seed_osm_prior=False,
        osm_prior_strength=0.0
    )

    # Load points
    points = np.fromfile(str(lidar_path), dtype=np.float32).reshape((-1, 4))[:, :3]
    
    # Load labels and mask
    labels_raw = np.fromfile(str(labels_path), dtype=np.uint32)
    labels_semantic = (labels_raw & 0xFFFF).astype(np.uint32)
    
    # Update
    bki.update(labels_semantic, points)
    
    # Infer
    refined_labels = bki.infer(points)
    refined_labels = np.array(refined_labels, dtype=np.uint32)

    metrics_after = calculate_metrics(refined_labels, gt_labels)

    return {
        "before": metrics_before,
        "after": metrics_after,
    }


def run_benchmark(
    lidar_path,
    labels_path,
    gt_labels_path,
    osm_path,
    config_path,
    randomization_levels,
    output_csv,
    num_runs=3,
    seed=None,
    randomization_dir=None,
    args=None,
):
    """
    Run benchmark across different OSM randomization levels.
    """
    print("=" * 80)
    print("Composite BKI OSM Randomization Benchmark")
    print("=" * 80)
    print()

    if not check_files_exist({
        "LiDAR data": Path(lidar_path),
        "Input labels": Path(labels_path),
        "Ground truth labels": Path(gt_labels_path),
        "OSM geometries": Path(osm_path),
        "Config": Path(config_path),
    }):
        raise FileNotFoundError("Required files missing")

    print("Loading data...")
    lidar_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 4))
    gt_raw = np.fromfile(gt_labels_path, dtype=np.uint32)
    gt_labels = (gt_raw & 0xFFFF).astype(np.uint32)

    print(f"  LiDAR points: {len(lidar_data)}")
    print(f"  Input labels: {Path(labels_path).name}")
    print(f"  Ground truth labels: {len(gt_labels)}")
    print()

    if randomization_dir is None:
        randomization_dir = Path(output_csv).parent / "randomized_osm"
    randomization_dir.mkdir(exist_ok=True)
    print(f"Saving randomized OSM files to: {randomization_dir}")
    print()

    results = []
    for idx, level in enumerate(randomization_levels):
        level_tag = f"{level:.2f}".replace(".", "p")
        params = compute_randomization_params(level, args)

        print(f"Testing randomization level {level}...")
        print(
            "  Params: "
            f"noise={params['noise_level']:.3f}, "
            f"removal={params['removal_rate']:.3f}, "
            f"simplify={params['simplification_rate']:.3f}, "
            f"scale_prob={params['scale_prob']:.3f}, "
            f"scale_range={params['scale_range']}, "
            f"shift_prob={params['shift_prob']:.3f}, "
            f"max_shift={params['max_shift']:.2f}m"
        )

        run_results = []
        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}...", end=" ", flush=True)

            run_seed = None
            if seed is not None:
                run_seed = int(seed) + idx * 1000 + run

            randomized_osm_path = randomization_dir / f"osm_rand_{level_tag}_run{run}.bin"

            randomize_osm(
                input_file=osm_path,
                output_file=randomized_osm_path,
                config_path=config_path,
                noise_level=params["noise_level"],
                noise_std=params["noise_std"],
                removal_rate=params["removal_rate"],
                simplification_rate=params["simplification_rate"],
                scale_prob=params["scale_prob"],
                scale_range=params["scale_range"],
                shift_prob=params["shift_prob"],
                max_shift=params["max_shift"],
                seed=run_seed,
            )

            metrics = run_single_benchmark(
                lidar_path=lidar_path,
                labels_path=labels_path,
                gt_labels=gt_labels,
                osm_path=randomized_osm_path,
                config_path=config_path,
            )

            run_results.append(metrics)

            print(
                f"Before: Acc={metrics['before']['accuracy']*100:.2f}%, "
                f"mIoU={metrics['before']['miou']*100:.2f}% | "
                f"After: Acc={metrics['after']['accuracy']*100:.2f}%, "
                f"mIoU={metrics['after']['miou']*100:.2f}%"
            )

        accuracies_before = [r["before"]["accuracy"] for r in run_results]
        mious_before = [r["before"]["miou"] for r in run_results]
        accuracies_after = [r["after"]["accuracy"] for r in run_results]
        mious_after = [r["after"]["miou"] for r in run_results]

        results.append({
            "randomization_level": level,
            "accuracy_before": np.mean(accuracies_before),
            "miou_before": np.mean(mious_before),
            "accuracy_after": np.mean(accuracies_after),
            "miou_after": np.mean(mious_after),
            "accuracy_improvement": np.mean(accuracies_after) - np.mean(accuracies_before),
            "miou_improvement": np.mean(mious_after) - np.mean(mious_before),
            "accuracy_before_std": np.std(accuracies_before),
            "miou_before_std": np.std(mious_before),
            "accuracy_after_std": np.std(accuracies_after),
            "miou_after_std": np.std(mious_after),
        })

        result = results[-1]
        print(
            f"  Average BEFORE - Accuracy: {result['accuracy_before']*100:.2f}% "
            f"(±{result['accuracy_before_std']*100:.2f}%), "
            f"mIoU: {result['miou_before']*100:.2f}% "
            f"(±{result['miou_before_std']*100:.2f}%)"
        )
        print(
            f"  Average AFTER  - Accuracy: {result['accuracy_after']*100:.2f}% "
            f"(±{result['accuracy_after_std']*100:.2f}%), "
            f"mIoU: {result['miou_after']*100:.2f}% "
            f"(±{result['miou_after_std']*100:.2f}%)"
        )
        print(
            f"  Improvement    - Accuracy: {result['accuracy_improvement']*100:+.2f}%, "
            f"mIoU: {result['miou_improvement']*100:+.2f}%"
        )
        print()

    print(f"Writing results to {output_csv}...")
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Randomization_Level",
            "Accuracy_Before", "mIoU_Before",
            "Accuracy_After", "mIoU_After",
            "Accuracy_Improvement", "mIoU_Improvement",
            "Accuracy_Before_Std", "mIoU_Before_Std",
            "Accuracy_After_Std", "mIoU_After_Std",
        ])

        for result in results:
            writer.writerow([
                result["randomization_level"],
                f"{result['accuracy_before']*100:.4f}",
                f"{result['miou_before']*100:.4f}",
                f"{result['accuracy_after']*100:.4f}",
                f"{result['miou_after']*100:.4f}",
                f"{result['accuracy_improvement']*100:.4f}",
                f"{result['miou_improvement']*100:.4f}",
                f"{result['accuracy_before_std']*100:.4f}",
                f"{result['miou_before_std']*100:.4f}",
                f"{result['accuracy_after_std']*100:.4f}",
                f"{result['miou_after_std']*100:.4f}",
            ])

    print()
    print("=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)
    print()
    print("Results Summary:")
    print(f"{'Level':<8} {'Acc Before':<12} {'mIoU Before':<13} "
          f"{'Acc After':<12} {'mIoU After':<13} {'Acc Δ':<10} {'mIoU Δ':<10}")
    print(f"{'(0-1)':<8} {'(%)':<12} {'(%)':<13} {'(%)':<12} {'(%)':<13} "
          f"{'(%)':<10} {'(%)':<10}")
    print("-" * 88)
    for result in results:
        print(
            f"{result['randomization_level']:<8.2f} "
            f"{result['accuracy_before']*100:<12.2f} "
            f"{result['miou_before']*100:<13.2f} "
            f"{result['accuracy_after']*100:<12.2f} "
            f"{result['miou_after']*100:<13.2f} "
            f"{result['accuracy_improvement']*100:<+10.2f} "
            f"{result['miou_improvement']*100:<+10.2f}"
        )
    print()
    print(f"📊 Results saved to: {output_csv}")
    print(f"🗺️  Randomized OSM files saved to: {randomization_dir}/")
    print()


def main():
    """Main benchmark execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark Composite BKI across OSM randomization levels"
    )

    parser.add_argument(
        "--lidar",
        type=str,
        default="../example_data/mcd-data/data/0000000011.bin",
        help="Path to LiDAR point cloud (.bin)",
    )

    parser.add_argument(
        "--labels",
        type=str,
        default="../example_data/mcd-data/labels_predicted/0000000011.bin",
        help="Path to input labels used for refinement",
    )

    parser.add_argument(
        "--gt-labels",
        type=str,
        default="../example_data/mcd-data/labels_groundtruth/0000000011.bin",
        help="Path to ground truth labels",
    )

    parser.add_argument(
        "--osm",
        type=str,
        default="../../example_data/kth.osm",
        help="Path to OSM file (.osm XML or .bin)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="../configs/mcd_config.yaml",
        help="Path to configuration YAML",
    )

    parser.add_argument(
        "--randomization-levels",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Randomization levels to test (0.0 to 1.0)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: auto-generated with timestamp)",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per randomization level",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed base for reproducible OSM randomization",
    )

    parser.add_argument(
        "--max-noise-level",
        type=float,
        default=0.2,
        help="Max coordinate noise probability at level=1.0",
    )

    parser.add_argument(
        "--noise-std",
        type=float,
        default=1.0,
        help="Coordinate noise standard deviation (meters)",
    )

    parser.add_argument(
        "--max-removal-rate",
        type=float,
        default=0.1,
        help="Max geometry removal rate at level=1.0",
    )

    parser.add_argument(
        "--max-simplification-rate",
        type=float,
        default=0.2,
        help="Max vertex removal rate at level=1.0",
    )

    parser.add_argument(
        "--max-scale-prob",
        type=float,
        default=0.3,
        help="Max geometry scaling probability at level=1.0",
    )

    parser.add_argument(
        "--scale-delta",
        type=float,
        default=0.2,
        help="Max scale delta (1±delta) at level=1.0",
    )

    parser.add_argument(
        "--max-shift-prob",
        type=float,
        default=0.3,
        help="Max geometry shift probability at level=1.0",
    )

    parser.add_argument(
        "--max-shift",
        type=float,
        default=5.0,
        help="Max geometry shift distance (meters) at level=1.0",
    )

    args = parser.parse_args()

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = Path(__file__).parent / f"osm_randomization_{timestamp}.csv"
    else:
        output_csv = Path(args.output)

    script_dir = Path(__file__).parent
    lidar_path = (script_dir / args.lidar).resolve()
    labels_path = (script_dir / args.labels).resolve()
    gt_labels_path = (script_dir / args.gt_labels).resolve()
    osm_path = (script_dir / args.osm).resolve()
    config_path = (script_dir / args.config).resolve()

    run_benchmark(
        lidar_path=lidar_path,
        labels_path=labels_path,
        gt_labels_path=gt_labels_path,
        osm_path=osm_path,
        config_path=config_path,
        randomization_levels=args.randomization_levels,
        output_csv=output_csv,
        num_runs=args.runs,
        seed=args.seed,
        args=args,
    )


if __name__ == "__main__":
    main()
