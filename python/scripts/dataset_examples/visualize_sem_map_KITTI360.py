#!/usr/bin/env python3
"""
Visualize a semantic map from a KITTI-360 sequence using Open3D.

Loads LiDAR scans and semantic labels, transforms to world frame using poses,
and displays the accumulated point cloud colored by semantic class.
Optionally overlays OSM building/road outlines and the drive path.

With --use-multiclass, loads multiclass confidence scores instead of one-hot
labels, computes accuracy vs GT, and reports uncertainty analysis.
"""

import argparse
import glob
import os
import sys

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from utils import *
from label_mappings import build_to_common_lut, apply_common_lut, common_labels_to_colors

# ---------------------------------------------------------------------------
# KITTI-360 pose I/O
# ---------------------------------------------------------------------------

def read_poses(poses_path):
    """Read KITTI-360 pose file. Returns dict[frame_index -> 4x4 numpy array]."""
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
    return poses


def get_velodyne_poses(kitti360_root, sequence_dir):
    """Load Velodyne poses from root/<seq>/velodyne_poses.txt."""
    velodyne_file = os.path.join(kitti360_root, sequence_dir, "velodyne_poses.txt")
    if not os.path.exists(velodyne_file):
        raise FileNotFoundError(f"velodyne_poses.txt not found: {velodyne_file}")
    poses = read_poses(velodyne_file)
    if not poses:
        raise ValueError(f"No valid poses in {velodyne_file}")
    return poses


# ---------------------------------------------------------------------------
# Point transformation
# ---------------------------------------------------------------------------

def transform_points_to_world(points_xyz, pose_4x4):
    """Transform (N, 3) points from LiDAR frame to world using 4x4 pose."""
    ones = np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)
    xyz_h = np.hstack([points_xyz, ones])
    return (xyz_h @ pose_4x4.T)[:, :3]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_sequence_dir(sequence_index):
    return f"2013_05_28_drive_{sequence_index:04d}_sync"


def get_sequence_paths(kitti360_root, sequence_index):
    """Return paths for a given sequence index."""
    seq_dir = get_sequence_dir(sequence_index)
    return {
        "sequence_dir": seq_dir,
        "raw_pc_dir": os.path.join(kitti360_root, seq_dir, "velodyne_points", "data"),
        "label_dir": os.path.join(kitti360_root, seq_dir, "gt_labels"),
    }


def get_osm_path(kitti360_root, sequence_index):
    """Return path to OSM file: root/<seq>/map_XXXX.osm."""
    seq_dir = get_sequence_dir(sequence_index)
    return os.path.join(kitti360_root, seq_dir, f"map_{sequence_index:04d}.osm")


def find_frame_range(label_dir):
    """Return (min_frame, max_frame) from .bin files in label_dir."""
    files = glob.glob(os.path.join(label_dir, "*.bin"))
    if not files:
        return None, None
    numbers = []
    for p in files:
        try:
            numbers.append(int(os.path.splitext(os.path.basename(p))[0]))
        except ValueError:
            continue
    if not numbers:
        return None, None
    return min(numbers), max(numbers)


def get_path_from_velodyne_poses(kitti360_root, sequence_index):
    """
    Build drive path from velodyne_poses.txt, shifted so the first pose is at origin.
    Returns (path_xyz (N,3), first_pose_xyz (3,)) or (None, None).
    """
    seq_dir = get_sequence_dir(sequence_index)
    try:
        poses = get_velodyne_poses(kitti360_root, seq_dir)
    except (FileNotFoundError, OSError):
        return None, None
    frame_ids = sorted(poses.keys())
    positions = np.array([poses[fid][:3, 3] for fid in frame_ids], dtype=np.float64)
    first_pose = positions[0].copy()
    return positions - first_pose, first_pose


# ---------------------------------------------------------------------------
# Map builders
# ---------------------------------------------------------------------------

def get_frame_indices(velodyne_poses, label_dir, downsample_factor, max_scans):
    """Return filtered and subsampled frame indices."""
    min_f, max_f = find_frame_range(label_dir)
    if min_f is None:
        raise ValueError(f"No label .bin files in {label_dir}")
    indices = sorted(velodyne_poses.keys())
    indices = [f for f in indices if min_f <= f <= max_f]
    if downsample_factor > 1:
        indices = indices[::downsample_factor]
    if max_scans:
        indices = indices[:max_scans]
    return indices


def build_semantic_map(kitti360_root, sequence_index=0, gt_common_lut=None,
                       downsample_factor=1, voxel_size=0.15, max_distance=None,
                       max_scans=None):
    """Load scans and GT labels, transform to world, return (points, colors)."""
    paths = get_sequence_paths(kitti360_root, sequence_index)
    seq_dir, raw_pc_dir, label_dir = paths["sequence_dir"], paths["raw_pc_dir"], paths["label_dir"]

    if not os.path.isdir(raw_pc_dir):
        raise FileNotFoundError(f"Raw LiDAR directory not found: {raw_pc_dir}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Labels directory not found: {label_dir}")

    velodyne_poses = get_velodyne_poses(kitti360_root, seq_dir)
    frame_indices = get_frame_indices(velodyne_poses, label_dir, downsample_factor, max_scans)

    all_xyz, all_colors = [], []
    for frame_id in frame_indices:
        pose = velodyne_poses.get(frame_id)
        if pose is None:
            continue

        pc_path = os.path.join(raw_pc_dir, f"{frame_id:010d}.bin")
        label_path = os.path.join(label_dir, f"{frame_id:010d}.bin")
        if not os.path.isfile(pc_path) or not os.path.isfile(label_path):
            continue

        try:
            xyz = read_bin_file(pc_path, dtype=np.float32, shape=(-1, 4))[:, :3]
            labels = read_bin_file(label_path, dtype=np.uint32, shape=(-1,)).astype(np.int32)
        except Exception as e:
            print(f"  Skip frame {frame_id}: {e}", file=sys.stderr)
            continue

        if len(labels) != len(xyz):
            labels = labels[:len(xyz)] if len(labels) > len(xyz) else np.resize(labels, len(xyz))

        world_xyz = transform_points_to_world(xyz, pose)
        if max_distance and max_distance > 0:
            mask = np.linalg.norm(world_xyz - pose[:3, 3], axis=1) <= max_distance
            world_xyz, labels = world_xyz[mask], labels[mask]

        colors = common_labels_to_colors(apply_common_lut(labels, gt_common_lut))

        if voxel_size and voxel_size > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(world_xyz.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            world_xyz, colors = np.asarray(pcd.points), np.asarray(pcd.colors)

        all_xyz.append(world_xyz)
        all_colors.append(colors)

    if not all_xyz:
        return None, None
    return np.vstack(all_xyz), np.vstack(all_colors)


def build_multiclass_map(kitti360_root, sequence_index=0, labels_key="kitti360",
                         learning_map_inv=None, gt_common_lut=None, inf_common_lut=None,
                         downsample_factor=1, voxel_size=0.15, max_distance=None,
                         max_scans=None, view_variance=False):
    """
    Load scans, GT labels, and multiclass confidence scores.
    Returns result dict with points, gt_labels, inf_labels, inf_variance,
    inf_uncertainty, n_classes.
    """
    seq_dir = get_sequence_dir(sequence_index)
    paths = get_sequence_paths(kitti360_root, sequence_index)
    raw_pc_dir = paths["raw_pc_dir"]
    infer_subdir = INFERRED_SUBDIRS.get(labels_key, f"cenet_{labels_key}")
    gt_label_dir = os.path.join(kitti360_root, seq_dir, "gt_labels")
    multiclass_dir = os.path.join(
        kitti360_root, seq_dir, "inferred_labels", infer_subdir, "multiclass_confidence_scores",
    )

    for d, desc in [(raw_pc_dir, "Raw LiDAR"), (gt_label_dir, "GT labels"), (multiclass_dir, "Multiclass")]:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"{desc} directory not found: {d}")

    print(f"GT label directory: {gt_label_dir}")
    print(f"Multiclass directory: {multiclass_dir}")

    velodyne_poses = get_velodyne_poses(kitti360_root, seq_dir)
    frame_indices = get_frame_indices(velodyne_poses, gt_label_dir, downsample_factor, max_scans)

    all_points, all_gt, all_inf, all_var, all_unc = [], [], [], [], []
    n_classes = 0

    for frame_id in frame_indices:
        pose = velodyne_poses.get(frame_id)
        if pose is None:
            continue

        pc_path = os.path.join(raw_pc_dir, f"{frame_id:010d}.bin")
        gt_path = os.path.join(gt_label_dir, f"{frame_id:010d}.bin")
        mc_path = os.path.join(multiclass_dir, f"{frame_id:010d}.bin")
        if not all(os.path.isfile(p) for p in [pc_path, gt_path, mc_path]):
            continue

        try:
            xyz = read_bin_file(pc_path, dtype=np.float32, shape=(-1, 4))[:, :3]
            gt_lbl = read_bin_file(gt_path, dtype=np.uint32, shape=(-1,)).astype(np.int32)
            raw_probs = read_bin_file(mc_path, dtype=np.float16)
            n_points = len(xyz)
            n_classes = len(raw_probs) // n_points
            if len(gt_lbl) != n_points or len(raw_probs) != n_points * n_classes:
                continue
            probs = raw_probs.reshape(n_points, n_classes)
        except Exception as e:
            print(f"  Skip frame {frame_id}: {e}", file=sys.stderr)
            continue

        gt_mapped = apply_common_lut(gt_lbl, gt_common_lut) if gt_common_lut is not None else gt_lbl
        class_idx = np.argmax(probs, axis=1)
        inf_raw = map_class_indices_to_labels(class_idx, learning_map_inv) if learning_map_inv else class_idx
        inf_mapped = apply_common_lut(inf_raw, inf_common_lut) if inf_common_lut is not None else inf_raw

        variances = np.var(probs.astype(np.float32), axis=1)
        if view_variance:
            max_var = (n_classes - 1) / (n_classes ** 2) if n_classes > 1 else 1.0
            uncertainty = 1.0 - np.clip(variances / max_var, 0.0, 1.0)
        else:
            uncertainty = None

        world_xyz = transform_points_to_world(xyz, pose)
        if max_distance and max_distance > 0:
            mask = np.linalg.norm(world_xyz - pose[:3, 3], axis=1) <= max_distance
            world_xyz, gt_mapped, inf_mapped, variances = (
                world_xyz[mask], gt_mapped[mask], inf_mapped[mask], variances[mask],
            )
            if uncertainty is not None:
                uncertainty = uncertainty[mask]

        if voxel_size and voxel_size > 0 and len(world_xyz) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(world_xyz.astype(np.float64))
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            world_ds = np.asarray(pcd.points)
            if len(world_ds) == 0:
                continue
            _, idx = cKDTree(world_xyz).query(world_ds, k=1)
            gt_mapped, inf_mapped, variances = gt_mapped[idx], inf_mapped[idx], variances[idx]
            if uncertainty is not None:
                uncertainty = uncertainty[idx]
            world_xyz = world_ds

        all_points.append(world_xyz.astype(np.float64))
        all_gt.append(gt_mapped)
        all_inf.append(inf_mapped)
        all_var.append(variances)
        if view_variance and uncertainty is not None:
            all_unc.append(uncertainty)

    if not all_points:
        return None
    return {
        "points": np.vstack(all_points),
        "gt_labels": np.concatenate(all_gt),
        "inf_labels": np.concatenate(all_inf),
        "inf_variance": np.concatenate(all_var),
        "inf_uncertainty": np.concatenate(all_unc) if all_unc else None,
        "n_classes": n_classes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize KITTI-360 semantic map with Open3D.",
    )
    parser.add_argument("--dataset-path", type=str,
                        default=os.path.join(OSMBKI_ROOT, "example_data", "kitti360"),
                        help="Root directory of the KITTI-360 dataset")
    parser.add_argument("--sequence", type=int, default=9,
                        help="Sequence index, e.g. 9 -> 2013_05_28_drive_0009_sync (default: 9)")
    parser.add_argument("--downsample-factor", type=int, default=1,
                        help="Process every Nth scan (default: 1)")
    parser.add_argument("--voxel-size", type=float, default=1.0,
                        help="Voxel size in meters for downsampling (default: 1.0, 0 to disable)")
    parser.add_argument("--max-distance", type=float, default=None,
                        help="Max distance from sensor per scan in meters")
    parser.add_argument("--max-scans", type=int, default=None,
                        help="Max number of scans to process")
    parser.add_argument("--inferred-labels", type=str, default="semkitti",
                        choices=list(INFERRED_LABEL_CONFIGS.keys()),
                        help="Inference network label set (default: semkitti)")
    parser.add_argument("--use-multiclass", action="store_true",
                        help="Load multiclass confidence scores; enables accuracy analysis")
    parser.add_argument("--variance", action="store_true",
                        help="Color by variance (Viridis: yellow=uncertain, dark=confident)")
    parser.add_argument("--max-uncertainty", type=float, default=None,
                        help="Keep points with uncertainty <= threshold in [0,1] (multiclass only)")
    parser.add_argument("--view", type=str, choices=["all", "correct", "incorrect"], default="all",
                        help="Filter displayed points (default: all)")
    parser.add_argument("--filter-by-confusion", action="store_true",
                        help="Use confusion matrix precision to filter uncertain predictions per class")
    parser.add_argument("--with-osm", action="store_true", help="Overlay OSM map")
    parser.add_argument("--osm-thickness", type=float, default=2.0, help="OSM line thickness in meters")
    parser.add_argument("--osm-all", action="store_true", help="Show all OSM features (default: buildings only)")
    parser.add_argument("--with-path", action="store_true", help="Show drive path")
    parser.add_argument("--path-thickness", type=float, default=1.5, help="Path line thickness in meters")
    args = parser.parse_args()

    voxel_size = args.voxel_size if args.voxel_size > 0 else None
    gt_common_lut = build_to_common_lut("kitti360")
    inf_common_lut = build_to_common_lut(args.inferred_labels)

    print(f"Dataset path: {args.dataset_path}")
    print(f"Sequence: {args.sequence}")

    geoms = []

    # -- Drive path (also needed to align OSM) -----------------------------
    path_xyz, first_pose = get_path_from_velodyne_poses(args.dataset_path, args.sequence)
    path_z = 0.0
    if path_xyz is not None and len(path_xyz) >= 2:
        path_z = float(np.median(path_xyz[:, 2]))
        if args.with_path:
            g = create_path_geometry(path_xyz, thickness=args.path_thickness)
            if g:
                geoms.append(g)
                print(f"Path: {len(path_xyz)} poses")
    elif args.with_path or args.with_osm:
        print("Warning: insufficient poses for path/OSM.", file=sys.stderr)

    # -- OSM overlay -------------------------------------------------------
    if args.with_osm:
        osm_file = get_osm_path(args.dataset_path, args.sequence)
        if os.path.isfile(osm_file):
            print(f"Loading OSM: {osm_file}")
            loader = OSMLoader(osm_file, origin_latlon=KITTI360_ORIGIN_LATLON)
            osm_geoms = loader.get_geometries(
                z_offset=path_z, thickness=args.osm_thickness, buildings_only=not args.osm_all,
            )
            if osm_geoms:
                if first_pose is not None:
                    trans = np.array([-first_pose[0], -first_pose[1], 0.0])
                    for m in osm_geoms:
                        m.translate(trans)
                geoms.extend(osm_geoms)
                print(f"OSM: {len(osm_geoms)} groups")
        else:
            print(f"OSM file not found: {osm_file}", file=sys.stderr)

    # -- Build map ---------------------------------------------------------
    if args.use_multiclass:
        label_cfg = load_label_config(INFERRED_LABEL_CONFIGS[args.inferred_labels])
        print("Building multiclass map...")
        result = build_multiclass_map(
            args.dataset_path, sequence_index=args.sequence,
            labels_key=args.inferred_labels,
            learning_map_inv=label_cfg["learning_map_inv"],
            gt_common_lut=gt_common_lut, inf_common_lut=inf_common_lut,
            downsample_factor=args.downsample_factor, voxel_size=voxel_size,
            max_distance=args.max_distance, max_scans=args.max_scans,
            view_variance=args.variance,
        )
        if result is None:
            print("No points loaded.", file=sys.stderr)
            sys.exit(1)

        points = result["points"]
        print(f"Total points: {len(points)}")

        correct, uncertainty_all, _, _ = run_accuracy_analysis(result)

        mask = np.ones(len(points), dtype=bool)
        if args.filter_by_confusion:
            mask &= filter_by_confusion_matrix(result, uncertainty_all)
        if args.max_uncertainty is not None:
            mask &= (uncertainty_all <= float(args.max_uncertainty))

        if args.view == "correct":
            mask &= correct
        elif args.view == "incorrect":
            mask &= ~correct
        print(f"Showing {int(np.sum(mask))} points (view={args.view})")

        pts = points[mask]
        if args.variance and result["inf_uncertainty"] is not None:
            colors = scalar_to_viridis_rgb(result["inf_uncertainty"][mask], normalize_range=False)
        else:
            colors = common_labels_to_colors(result["inf_labels"][mask])

        if first_pose is not None and (args.with_osm or args.with_path):
            pts = pts - first_pose.astype(pts.dtype)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geoms.insert(0, pcd)

    else:
        print("Building semantic map...")
        points, colors = build_semantic_map(
            args.dataset_path, sequence_index=args.sequence,
            gt_common_lut=gt_common_lut,
            downsample_factor=args.downsample_factor, voxel_size=voxel_size,
            max_distance=args.max_distance, max_scans=args.max_scans,
        )
        if points is not None:
            if first_pose is not None and (args.with_osm or args.with_path):
                points = points - first_pose.astype(points.dtype)
            print(f"Total points: {len(points)}")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            geoms.insert(0, pcd)
        else:
            print("No points loaded.", file=sys.stderr)

    if not geoms:
        print("Nothing to display.", file=sys.stderr)
        sys.exit(1)

    o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()
