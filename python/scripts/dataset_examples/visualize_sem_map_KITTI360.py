#!/usr/bin/env python3
"""
Visualize a semantic map from a KITTI-360 sequence using Open3D.

Loads LiDAR scans and semantic labels, transforms to world frame using poses,
and displays the accumulated point cloud colored by semantic class.
Optionally overlays OSM building/road outlines and the drive path.

With --use-multiclass, loads multiclass confidence scores instead of one-hot
labels, computes accuracy vs GT, and reports uncertainty analysis
(before/after removing uncertain points, Spearman correlation, AUROC).

    Usage:
        python visualize_sem_map_KITTI360.py
        python visualize_sem_map_KITTI360.py --sequence 2 --scan-skip 5 --downsample 0.2
        python visualize_sem_map_KITTI360.py --with-osm --with-path
        python visualize_sem_map_KITTI360.py --with-osm --osm-all --osm-thickness 3
"""

import argparse
import glob
import os
import sys

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr
import matplotlib.pyplot as plt

from utils import *
from label_mappings import build_to_common_lut, apply_common_lut, common_labels_to_colors, IGNORE_LABELS

def load_label_config(config_path):
    """Load learning_map and learning_map_inv from a label YAML config."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    learning_map = {int(k): int(v) for k, v in cfg.get("learning_map", {}).items()}
    learning_map_inv = {int(k): int(v) for k, v in cfg.get("learning_map_inv", {}).items()}
    return {"learning_map": learning_map, "learning_map_inv": learning_map_inv}


def map_labels_to_class_indices(labels, learning_map):
    """Map raw label IDs to class indices via learning_map. Unknown labels map to 0."""
    maxkey = max(learning_map.keys(), default=0)
    lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, value in learning_map.items():
        try:
            lut[int(key)] = int(value)
        except IndexError:
            pass
    safe = np.clip(labels.astype(np.int64), 0, len(lut) - 1)
    return lut[safe]


def read_poses(poses_path):
    """
    Read KITTI-360 pose file: frame_index followed by 16 floats (4x4) or 12 floats (3x4).
    Skips empty lines and lines that don't parse (e.g. header). Returns dict[frame_index -> 4x4 numpy array].
    """
    poses = {}
    with open(poses_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 13:
                continue
            try:
                frame_index = int(parts[0])
                values = np.array(parts[1:], dtype=float)
            except (ValueError, TypeError):
                continue
            if len(values) == 16:
                poses[frame_index] = values.reshape(4, 4)
            elif len(values) == 12:
                m = values.reshape(3, 4)
                pose_4x4 = np.eye(4)
                pose_4x4[:3, :] = m
                poses[frame_index] = pose_4x4
            # else: skip line (wrong length)
    return poses


def get_velodyne_poses(kitti360_root, sequence_dir):
    """Load Velodyne poses from root/<seq>/velodyne_poses.txt."""
    velodyne_file = os.path.join(kitti360_root, sequence_dir, "velodyne_poses.txt")
    if not os.path.exists(velodyne_file):
        raise FileNotFoundError(f"velodyne_poses.txt not found: {velodyne_file}")
    poses = read_poses(velodyne_file)
    if not poses:
        raise ValueError(f"velodyne_poses.txt is empty or has no valid lines: {velodyne_file}")
    return poses


def transform_points_to_world(points_xyz, pose_4x4):
    """Transform (N, 3) points from LiDAR frame to world using 4x4 pose (lidar-to-world)."""
    ones = np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)
    xyz_h = np.hstack([points_xyz, ones])  # (N, 4)
    return (xyz_h @ pose_4x4.T)[:, :3]



def get_sequence_paths(kitti360_root, sequence_index):
    """Return paths for a given sequence index (e.g. 0 -> 2013_05_28_drive_0000_sync)."""
    seq_dir = f"2013_05_28_drive_{sequence_index:04d}_sync"
    raw_pc_dir = os.path.join(kitti360_root, seq_dir, "velodyne_points", "data")
    gt_labels_dir = os.path.join(kitti360_root, seq_dir, "gt_labels")
    return {
        "sequence_dir": seq_dir,
        "raw_pc_dir": raw_pc_dir,
        "label_dir": gt_labels_dir,
    }


def find_frame_range(label_dir):
    """Return (min_frame, max_frame) from existing .bin files in label_dir."""
    pattern = os.path.join(label_dir, "*.bin")
    files = glob.glob(pattern)
    if not files:
        return None, None
    numbers = []
    for p in files:
        base = os.path.basename(p)
        try:
            numbers.append(int(os.path.splitext(base)[0]))
        except ValueError:
            continue
    if not numbers:
        return None, None
    return min(numbers), max(numbers)


def build_semantic_map(
    kitti360_root,
    sequence_index=0,
    gt_common_lut=None,
    scan_skip=1,
    voxel_size=0.15,
    max_range=None,
    max_points=None,
):
    """
    Load scans and labels for one sequence, transform to world, and return (points, colors).
    scan_skip: use every Nth scan (1 = all scans). voxel_size: voxel size in m for downsampling (0 = disable).
    If max_points is set, stop once total points reach that limit (truncating the last scan if needed).
    """
    paths = get_sequence_paths(kitti360_root, sequence_index)
    seq_dir = paths["sequence_dir"]
    raw_pc_dir = paths["raw_pc_dir"]
    label_dir = paths["label_dir"]

    print(f"Label directory: {label_dir}")
    if not os.path.isdir(raw_pc_dir):
        raise FileNotFoundError(f"Raw LiDAR directory not found: {raw_pc_dir}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Semantic labels directory not found: {label_dir}")

    velodyne_poses = get_velodyne_poses(kitti360_root, seq_dir)
    if not velodyne_poses:
        raise ValueError("No poses loaded.")

    min_f, max_f = find_frame_range(label_dir)
    if min_f is None:
        raise ValueError(f"No label .bin files in {label_dir}")

    frame_indices = sorted(velodyne_poses.keys())
    frame_indices = [f for f in frame_indices if min_f <= f <= max_f]
    if scan_skip > 1:
        frame_indices = frame_indices[::scan_skip]

    all_xyz = []
    all_colors = []

    for frame_id in frame_indices:
        pose = velodyne_poses.get(frame_id)
        if pose is None:
            continue

        pc_path = os.path.join(raw_pc_dir, f"{frame_id:010d}.bin")
        label_path = os.path.join(label_dir, f"{frame_id:010d}.bin")
        if not os.path.isfile(pc_path) or not os.path.isfile(label_path):
            continue

        try:
            pc = read_bin_file(pc_path, dtype=np.float32, shape=(-1, 4))
            xyz = pc[:, :3]
            # KITTI-360 semantic labels are stored as int16
            labels = read_bin_file(label_path, dtype=np.uint32, shape=(-1)).astype(np.int32)
        except Exception as e:
            print(f"  Skip frame {frame_id}: {e}", file=sys.stderr)
            continue

        if len(labels) != len(xyz):
            # Align: truncate labels or pad with 0 (unlabeled) so we can still visualize
            if len(labels) > len(xyz):
                labels = labels[: len(xyz)]
            else:
                labels = np.resize(labels, len(xyz))
        world_xyz = transform_points_to_world(xyz, pose)

        if max_range is not None and max_range > 0:
            dist = np.linalg.norm(world_xyz - pose[:3, 3], axis=1)
            mask = dist <= max_range
            world_xyz = world_xyz[mask]
            labels = labels[mask]

        colors = common_labels_to_colors(apply_common_lut(labels, gt_common_lut))

        if voxel_size is not None and voxel_size > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(world_xyz.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            world_xyz = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)

        current_total = sum(len(x) for x in all_xyz)
        if max_points is not None and current_total >= max_points:
            break
        if max_points is not None and current_total + len(world_xyz) > max_points:
            need = max_points - current_total
            world_xyz = world_xyz[:need]
            colors = colors[:need]
        all_xyz.append(world_xyz)
        all_colors.append(colors)

    if not all_xyz:
        return None, None

    points = np.vstack(all_xyz)
    colors = np.vstack(all_colors)
    return points, colors


def build_multiclass_map(
    kitti360_root,
    sequence_index=0,
    labels_key="kitti360",
    learning_map_inv=None,
    gt_common_lut=None,
    inf_common_lut=None,
    scan_skip=1,
    voxel_size=0.15,
    max_range=None,
    max_points=None,
    view_variance=False,
):
    """
    Load scans, GT labels, and multiclass confidence scores for one sequence.
    GT labels are mapped to common taxonomy via gt_common_lut (KITTI360 raw IDs).
    Inferred labels are mapped via learning_map_inv then inf_common_lut.
    Returns dict with points, gt_labels, inf_labels (all in common taxonomy IDs),
    inf_variance, and optionally inf_uncertainty.
    """
    seq_dir = f"2013_05_28_drive_{sequence_index:04d}_sync"
    paths = get_sequence_paths(kitti360_root, sequence_index)
    raw_pc_dir = paths["raw_pc_dir"]

    infer_subdir = INFERRED_SUBDIRS.get(labels_key, f"cenet_{labels_key}")
    gt_label_dir = os.path.join(kitti360_root, seq_dir, "gt_labels")
    multiclass_dir = os.path.join(kitti360_root, seq_dir, "inferred_labels", infer_subdir, "multiclass_confidence_scores")

    if not os.path.isdir(raw_pc_dir):
        raise FileNotFoundError(f"Raw LiDAR directory not found: {raw_pc_dir}")
    if not os.path.isdir(gt_label_dir):
        raise FileNotFoundError(f"GT labels directory not found: {gt_label_dir}")
    if not os.path.isdir(multiclass_dir):
        raise FileNotFoundError(f"Multiclass confidence scores directory not found: {multiclass_dir}")

    print(f"GT label directory: {gt_label_dir}")
    print(f"Multiclass directory: {multiclass_dir}")

    velodyne_poses = get_velodyne_poses(kitti360_root, seq_dir)
    if not velodyne_poses:
        raise ValueError("No poses loaded.")

    min_f, max_f = find_frame_range(gt_label_dir)
    if min_f is None:
        raise ValueError(f"No label .bin files in {gt_label_dir}")

    frame_indices = sorted(velodyne_poses.keys())
    frame_indices = [f for f in frame_indices if min_f <= f <= max_f]
    if scan_skip > 1:
        frame_indices = frame_indices[::scan_skip]

    all_points = []
    all_gt_labels = []
    all_inf_labels = []
    all_inf_variance = []
    all_inf_uncertainty = []

    for frame_id in frame_indices:
        pose = velodyne_poses.get(frame_id)
        if pose is None:
            continue

        pc_path = os.path.join(raw_pc_dir, f"{frame_id:010d}.bin")
        gt_path = os.path.join(gt_label_dir, f"{frame_id:010d}.bin")
        mc_path = os.path.join(multiclass_dir, f"{frame_id:010d}.bin")
        if not os.path.isfile(pc_path) or not os.path.isfile(gt_path) or not os.path.isfile(mc_path):
            continue

        try:
            pc = read_bin_file(pc_path, dtype=np.float32, shape=(-1, 4))
            xyz = pc[:, :3]
            gt_lbl = read_bin_file(gt_path, dtype=np.uint32, shape=(-1)).astype(np.int32)
            raw_probs = read_bin_file(mc_path, dtype=np.float16)
            n_points = len(xyz)
            n_classes = len(raw_probs) // n_points
            if len(gt_lbl) != n_points or len(raw_probs) != n_points * n_classes:
                continue
            multiclass_probs = raw_probs.reshape(n_points, n_classes)
        except Exception as e:
            print(f"  Skip frame {frame_id}: {e}", file=sys.stderr)
            continue

        # Map GT raw labels directly to common taxonomy
        gt_lbl_mapped = apply_common_lut(gt_lbl, gt_common_lut) if gt_common_lut is not None else gt_lbl

        # Map inferred class indices to label IDs, then to common taxonomy
        class_indices = np.argmax(multiclass_probs, axis=1)
        inf_lbl_raw = map_class_indices_to_labels(class_indices, learning_map_inv) if learning_map_inv else class_indices
        inf_lbl = apply_common_lut(inf_lbl_raw, inf_common_lut) if inf_common_lut is not None else inf_lbl_raw

        variances = np.var(multiclass_probs.astype(np.float32), axis=1)
        if view_variance:
            max_var = (n_classes - 1) / (n_classes ** 2) if n_classes > 1 else 1.0
            scaled = np.clip(variances / max_var, 0.0, 1.0)
            uncertainty = 1.0 - scaled
        else:
            uncertainty = None

        world_xyz = transform_points_to_world(xyz, pose)

        if max_range is not None and max_range > 0:
            dist = np.linalg.norm(world_xyz - pose[:3, 3], axis=1)
            mask = dist <= max_range
            world_xyz = world_xyz[mask]
            gt_lbl_mapped = gt_lbl_mapped[mask]
            inf_lbl = inf_lbl[mask]
            variances = variances[mask]
            if uncertainty is not None:
                uncertainty = uncertainty[mask]

        if voxel_size is not None and voxel_size > 0 and len(world_xyz) > 0:
            scan_pcd = o3d.geometry.PointCloud()
            scan_pcd.points = o3d.utility.Vector3dVector(world_xyz.astype(np.float64))
            scan_pcd = scan_pcd.voxel_down_sample(voxel_size=voxel_size)
            world_ds = np.asarray(scan_pcd.points)
            if len(world_ds) == 0:
                continue
            tree_scan = cKDTree(world_xyz)
            _, idx = tree_scan.query(world_ds, k=1)
            gt_lbl_mapped = gt_lbl_mapped[idx]
            inf_lbl = inf_lbl[idx]
            variances = variances[idx]
            if uncertainty is not None:
                uncertainty = uncertainty[idx]
            world_xyz = world_ds

        current_total = sum(len(x) for x in all_points)
        if max_points is not None and current_total >= max_points:
            break
        if max_points is not None and current_total + len(world_xyz) > max_points:
            need = max_points - current_total
            world_xyz = world_xyz[:need]
            gt_lbl_mapped = gt_lbl_mapped[:need]
            inf_lbl = inf_lbl[:need]
            variances = variances[:need]
            if uncertainty is not None:
                uncertainty = uncertainty[:need]

        all_points.append(world_xyz.astype(np.float64))
        all_gt_labels.append(gt_lbl_mapped)
        all_inf_labels.append(inf_lbl)
        all_inf_variance.append(variances)
        if view_variance and uncertainty is not None:
            all_inf_uncertainty.append(uncertainty)

    if not all_points:
        return None

    return {
        "points": np.vstack(all_points),
        "gt_labels": np.concatenate(all_gt_labels),
        "inf_labels": np.concatenate(all_inf_labels),
        "inf_variance": np.concatenate(all_inf_variance),
        "inf_uncertainty": np.concatenate(all_inf_uncertainty) if all_inf_uncertainty else None,
        "n_classes": n_classes,
    }


def run_accuracy_analysis(result):
    """Run accuracy analysis, uncertainty calibration, and print reports. Returns (correct, uncertainty_all) over ALL points (including ignored)."""
    gt_labels = result["gt_labels"]
    inf_labels = result["inf_labels"]
    inf_variance = result["inf_variance"]
    n_classes = result["n_classes"]

    # Mask to exclude ignored labels from accuracy metrics
    ignore_set = set(IGNORE_LABELS)
    valid = np.array([g not in ignore_set for g in gt_labels], dtype=bool)
    n_ignored = int(np.sum(~valid))
    print(f"Ignoring {n_ignored} points with GT label in {IGNORE_LABELS}")

    correct_all = (inf_labels == gt_labels)
    correct = correct_all[valid]
    n_correct = int(np.sum(correct))
    n_total = len(correct)
    print(f"Accuracy: {n_correct}/{n_total} ({100.0 * n_correct / n_total:.2f}%)")

    max_var = (n_classes - 1) / (n_classes ** 2) if n_classes > 1 else 1.0
    scaled_var = np.clip(inf_variance / max_var, 0.0, 1.0)
    uncertainty_all = 1.0 - scaled_var
    uncertainty = uncertainty_all[valid]
    median_uncertainty = float(np.median(uncertainty))
    print(f"Median uncertainty: {median_uncertainty:.6f}")

    low_mask = (uncertainty <= median_uncertainty)
    n_kept = int(np.sum(low_mask))
    n_removed = n_total - n_kept
    n_correct_kept = int(np.sum(correct[low_mask]))
    accuracy_kept = 100.0 * n_correct_kept / n_kept if n_kept > 0 else 0.0
    print(f"Keeping points with uncertainty <= median: n={n_kept} (removed {n_removed})")
    print(f"Accuracy after removing high-uncertainty points: {n_correct_kept}/{n_kept} ({accuracy_kept:.2f}%)")

    incorrect = np.asarray(~correct, dtype=np.float64)
    rho, p_val = spearmanr(uncertainty, incorrect)
    print(f"\n--- Uncertainty calibration ---")
    print(f"Spearman(uncertainty, incorrect): rho = {rho:.4f} (p = {p_val:.2e})")

    n_incorrect = int(np.sum(incorrect))
    n_correct_count = n_total - n_incorrect
    if n_incorrect > 0 and n_correct_count > 0:
        ranks = rankdata(uncertainty)
        S = np.sum(ranks[~correct])
        auroc = (S - n_incorrect * (n_incorrect + 1) / 2) / (n_incorrect * n_correct_count)
        print(f"AUROC(uncertainty -> predicts error): {auroc:.4f}")

    n_bins = 10
    bin_edges = np.percentile(uncertainty, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-9
    print(f"\nAccuracy by uncertainty bin (lower bin = more confident):")
    bin_accs = []
    bin_centers = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        b = (uncertainty >= lo) & (uncertainty < hi)
        n_b = int(np.sum(b))
        if n_b > 0:
            acc_b = 100.0 * np.sum(correct[b]) / n_b
            bin_accs.append(acc_b)
            bin_centers.append((lo + hi) / 2)
            print(f"  bin {i+1:2d} [unc {lo:.3f}-{hi:.3f}]: acc = {acc_b:.1f}%  (n={n_b})")
    if bin_accs:
        fig_cal, ax_cal = plt.subplots(figsize=(7, 4))
        ax_cal.plot(bin_centers, bin_accs, "o-", color="steelblue", linewidth=2, markersize=8)
        ax_cal.set_xlabel("Mean uncertainty (bin)")
        ax_cal.set_ylabel("Accuracy (%)")
        ax_cal.set_title("Accuracy by uncertainty bin")
        ax_cal.set_ylim(0, 105)
        ax_cal.grid(True, alpha=0.3)
        fig_cal.tight_layout()
        plt.show(block=True)
        plt.close(fig_cal)

    var_valid = inf_variance[valid]
    avg_var = float(np.mean(var_valid))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(var_valid, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(avg_var, color="coral", linewidth=2, label=f"Mean = {avg_var:.4f}")
    ax.set_xlabel("Variance of class probabilities")
    ax.set_ylabel("Count")
    ax.set_title(f"Variance distribution (n={n_total})")
    ax.legend()
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)

    return correct_all, uncertainty_all


# ---------------------------------------------------------------------------
# OSM overlay helpers (KITTI360-specific)
# ---------------------------------------------------------------------------

def get_osm_path(kitti360_root, sequence_index):
    """Return path to OSM file: root/<seq>/map_XXXX.osm."""
    seq_dir = f"2013_05_28_drive_{sequence_index:04d}_sync"
    return os.path.join(kitti360_root, seq_dir, f"map_{sequence_index:04d}.osm")


def get_path_from_velodyne_poses(kitti360_root, sequence_index):
    """
    Build path from velodyne_poses.txt in world frame, shifted so the first pose is at (0, 0, 0).
    Returns (path_xyz (N,3), first_pose_xyz (3,)) or (None, None) if no poses.
    """
    seq_dir = f"2013_05_28_drive_{sequence_index:04d}_sync"
    try:
        poses = get_velodyne_poses(kitti360_root, seq_dir)
    except (FileNotFoundError, OSError):
        return None, None
    if not poses:
        return None, None
    frame_ids = sorted(poses.keys())
    positions = np.array([poses[fid][:3, 3] for fid in frame_ids], dtype=np.float64)
    first_pose = positions[0].copy()
    path_xyz = positions - first_pose
    return path_xyz, first_pose


def main():
    parser = argparse.ArgumentParser(
        description="Visualize KITTI-360 semantic map with Open3D. Supports one-hot labels (default) "
                    "or multiclass confidence scores with accuracy analysis."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.join(OSMBKI_ROOT, "example_data", "kitti360"),
    )
    parser.add_argument(
        "--sequence",
        type=int,
        default=9,
        help=f"Sequence index, e.g. 0 -> 2013_05_28_drive_0000_sync (default: 9)",
    )
    parser.add_argument(
        "--scan-skip",
        type=int,
        default=1,
        help="Use every Nth scan (1=all, 2=every other, etc.) (default: 1)",
    )
    parser.add_argument(
        "--downsample",
        type=float,
        default=1.0,
        help="Voxel size in meters for voxel downsampling (default: 1.0, use 0 to disable)",
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=None,
        help="Max distance from sensor per scan in meters (default: none)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Stop accumulating once total points reach this limit (default: none)",
    )
    parser.add_argument(
        "--inferred-labels",
        type=str,
        default="semkitti",
        choices=list(INFERRED_LABEL_CONFIGS.keys()),
        help="Inference network label set (default: kitti360). Options: %(choices)s",
    )
    # Multiclass / accuracy arguments
    parser.add_argument("--use-multiclass", action="store_true",
                        help="Load multiclass confidence scores instead of one-hot labels; enables accuracy analysis")
    parser.add_argument("--variance", action="store_true",
                        help="Color by variance of class probabilities (Viridis: yellow=uncertain, dark=confident)")
    parser.add_argument("--view", type=str, choices=["all", "correct", "incorrect"], default="all",
                        help="When using multiclass: show all / correct / incorrect points (default: all)")
    # OSM overlay arguments
    parser.add_argument("--with-osm", action="store_true", help="Overlay OSM map on the semantic map")
    parser.add_argument("--osm-thickness", type=float, default=2.0, help="OSM line thickness in meters (default: 2.0)")
    parser.add_argument("--osm-all", action="store_true", help="Show all OSM features; default is buildings only")
    parser.add_argument("--with-path", action="store_true", help="Show drive path from velodyne_poses.txt")
    parser.add_argument("--path-thickness", type=float, default=1.5, help="Path line thickness in meters (default: 1.5)")
    args = parser.parse_args()

    voxel_size = args.downsample if args.downsample > 0 else None

    # Build common taxonomy LUTs
    gt_common_lut = build_to_common_lut("kitti360")
    inf_common_lut = build_to_common_lut(args.inferred_labels)

    print(f"KITTI360 root: {args.root}")
    print(f"Sequence index: {args.sequence}")

    geoms = []
    first_pose_for_osm = None

    # -- Velodyne path (also needed to align OSM) -------------------------
    path_xyz, first_pose_for_osm = get_path_from_velodyne_poses(args.root, args.sequence)
    path_z_default = 0.0
    if path_xyz is not None and len(path_xyz) >= 2:
        path_z_default = float(np.median(path_xyz[:, 2]))
        if args.with_path:
            path_geom = create_path_geometry(path_xyz, color=(0.0, 0.8, 0.0), thickness=args.path_thickness)
            if path_geom is not None:
                geoms.append(path_geom)
                print(f"Path: {len(path_xyz)} poses from velodyne_poses.txt")
    else:
        if args.with_path or args.with_osm:
            print("Warning: no velodyne poses or too few frames for path/OSM alignment.", file=sys.stderr)

    # -- OSM overlay ------------------------------------------------------
    if args.with_osm:
        osm_file = get_osm_path(args.root, args.sequence)
        if os.path.isfile(osm_file):
            print(f"Loading OSM: {osm_file}")
            loader = OSMLoader(osm_file, origin_latlon=KITTI360_ORIGIN_LATLON)
            osm_geoms = loader.get_geometries(
                z_offset=path_z_default, thickness=args.osm_thickness, buildings_only=not args.osm_all,
            )
            if osm_geoms:
                if first_pose_for_osm is not None:
                    trans = np.array([-first_pose_for_osm[0], -first_pose_for_osm[1], 0.0])
                    for mesh in osm_geoms:
                        mesh.translate(trans)
                    print("OSM: shifted by -first_pose to align with semantic map")
                geoms.extend(osm_geoms)
                kind = "building" if not args.osm_all else "geometry"
                print(f"OSM: {len(osm_geoms)} {kind} groups")
        else:
            print(f"OSM file not found: {osm_file}", file=sys.stderr)

    # -- Build map --------------------------------------------------------
    if args.use_multiclass:
        # Load label config for learning_map / learning_map_inv
        config_path = INFERRED_LABEL_CONFIGS.get(args.inferred_labels)
        if config_path is None or not os.path.isfile(config_path):
            print(f"ERROR: Label config not found for '{args.inferred_labels}': {config_path}", file=sys.stderr)
            sys.exit(1)
        label_cfg = load_label_config(config_path)

        print("Building multiclass map...")
        result = build_multiclass_map(
            args.root,
            sequence_index=args.sequence,
            labels_key=args.inferred_labels,
            learning_map_inv=label_cfg["learning_map_inv"],
            gt_common_lut=gt_common_lut,
            inf_common_lut=inf_common_lut,
            scan_skip=args.scan_skip,
            voxel_size=voxel_size,
            max_range=args.max_range,
            max_points=args.max_points,
            view_variance=args.variance,
        )

        if result is None:
            print("No points loaded. Check paths and sequence.", file=sys.stderr)
            sys.exit(1)

        points = result["points"]
        inf_labels = result["inf_labels"]
        inf_uncertainty = result["inf_uncertainty"]
        n_total = len(points)
        print(f"Total points: {n_total}")

        # Accuracy analysis
        correct, uncertainty_all = run_accuracy_analysis(result)

        # Filter by view mode
        if args.view == "correct":
            mask = correct
        elif args.view == "incorrect":
            mask = ~correct
        else:
            mask = np.ones(n_total, dtype=bool)
        n_show = int(np.sum(mask))
        print(f"Showing {n_show} points (view={args.view})")

        pts = points[mask]

        # Color by variance or by common taxonomy class
        if args.variance and inf_uncertainty is not None:
            colors = scalar_to_viridis_rgb(inf_uncertainty[mask], normalize_range=False)
        else:
            colors = common_labels_to_colors(inf_labels[mask])

        if first_pose_for_osm is not None and (args.with_osm or args.with_path):
            pts = pts - first_pose_for_osm.astype(pts.dtype)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geoms.insert(0, pcd)

    else:
        # Default: one-hot label mode
        print("Building semantic map...")
        points, colors = build_semantic_map(
            args.root,
            sequence_index=args.sequence,
            gt_common_lut=gt_common_lut,
            scan_skip=args.scan_skip,
            voxel_size=voxel_size,
            max_range=args.max_range,
            max_points=args.max_points,
        )

        if points is not None:
            if first_pose_for_osm is not None and (args.with_osm or args.with_path):
                points = points - first_pose_for_osm.astype(points.dtype)

            print(f"Total points: {len(points)}")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            geoms.insert(0, pcd)
        else:
            print("No semantic points loaded. Check paths and sequence.", file=sys.stderr)

    if not geoms:
        print("Nothing to display.", file=sys.stderr)
        sys.exit(1)

    print("Opening Open3D viewer (mouse: rotate, shift+mouse: pan, wheel: zoom, Q/ESC: quit).")
    o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()
