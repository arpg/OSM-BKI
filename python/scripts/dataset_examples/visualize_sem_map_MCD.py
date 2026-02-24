#!/usr/bin/env python3
"""
Visualize a semantic map from an MCD sequence using Open3D.

Loads LiDAR scans and semantic labels, transforms to world frame using poses,
and displays the accumulated point cloud colored by semantic class.
Optionally overlays OSM building/road outlines and the drive path.

With --use-multiclass, loads multiclass confidence scores instead of one-hot
labels, computes accuracy vs GT, and reports uncertainty analysis.
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from utils import *
from label_mappings import build_to_common_lut, apply_common_lut, common_labels_to_colors

# ---------------------------------------------------------------------------
# MCD-specific I/O
# ---------------------------------------------------------------------------

def load_body_to_lidar_tf(calib_path, sensor_key="os_sensor"):
    """Load the body-to-LiDAR 4x4 transform from a calibration YAML."""
    with open(calib_path, "r") as f:
        cfg = yaml.safe_load(f)
    return np.array(cfg["body"][sensor_key]["T"], dtype=np.float64)


def load_poses(poses_file):
    """
    Load poses from CSV with columns: num, t, x, y, z, qx, qy, qz, qw.
    Returns dict mapping scan number (int) to [x, y, z, qx, qy, qz, qw].
    """
    df = pd.read_csv(poses_file)
    poses = {}
    for _, row in df.iterrows():
        num = int(row["num"])
        poses[num] = [
            float(row["x"]), float(row["y"]), float(row["z"]),
            float(row["qx"]), float(row["qy"]), float(row["qz"]), float(row["qw"]),
        ]
    print(f"Loaded {len(poses)} poses from {poses_file}")
    return poses


# ---------------------------------------------------------------------------
# Point transformation
# ---------------------------------------------------------------------------

def transform_points_to_world(points_xyz, position, quaternion, body_to_lidar_tf=None):
    """Transform (N, 3) points from lidar frame to world frame using pose."""
    body_to_world = np.eye(4)
    body_to_world[:3, :3] = R.from_quat(quaternion).as_matrix()
    body_to_world[:3, 3] = position

    if body_to_lidar_tf is not None:
        transform = body_to_world @ np.linalg.inv(body_to_lidar_tf)
    else:
        transform = body_to_world

    pts_h = np.hstack([points_xyz, np.ones((len(points_xyz), 1), dtype=points_xyz.dtype)])
    return (transform @ pts_h.T).T[:, :3]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_mcd_osm_path(dataset_path, osm_filename="kth.osm"):
    """Return path to OSM file for the MCD dataset."""
    return os.path.join(dataset_path, osm_filename)


def build_path_from_poses(poses):
    """Extract world-frame path positions from MCD poses. Returns (N,3) or None."""
    if not poses:
        return None
    positions = [[poses[n][0], poses[n][1], poses[n][2]] for n in sorted(poses.keys())]
    if len(positions) < 2:
        return None
    return np.array(positions, dtype=np.float64)


# ---------------------------------------------------------------------------
# Map builders
# ---------------------------------------------------------------------------

def get_sorted_pose_indices(poses, downsample_factor, max_scans):
    """Return filtered and subsampled pose indices."""
    indices = sorted(poses.keys())
    if downsample_factor > 1:
        indices = indices[::downsample_factor]
    if max_scans:
        indices = indices[:max_scans]
    return indices


def build_semantic_map_mcd(dataset_path, seq_name, body_to_lidar_tf,
                           downsample_factor=1, voxel_size=0.1, max_distance=None,
                           max_scans=None):
    """Build a one-hot GT semantic map. Returns (points, colors) or (None, None)."""
    root_path = os.path.join(dataset_path, seq_name)
    data_dir = os.path.join(root_path, "lidar_bin/data")
    poses_file = os.path.join(root_path, "pose_inW.csv")
    gt_labels_dir = os.path.join(root_path, "gt_labels")
    gt_common_lut = build_to_common_lut("mcd")

    for path, desc in [(data_dir, "Data"), (poses_file, "Poses"), (gt_labels_dir, "GT labels")]:
        if not os.path.exists(path):
            print(f"ERROR: {desc} not found: {path}", file=sys.stderr)
            return None, None

    poses = load_poses(poses_file)
    if not poses:
        print("ERROR: No poses loaded.", file=sys.stderr)
        return None, None

    sorted_indices = get_sorted_pose_indices(poses, downsample_factor, max_scans)
    all_points, all_colors = [], []

    for pose_num in tqdm(sorted_indices, desc="Loading scans", unit="scan"):
        pose_data = poses[pose_num]
        position, quaternion = pose_data[0:3], pose_data[3:7]
        bin_file = f"{pose_num:010d}.bin"
        bin_path = os.path.join(data_dir, bin_file)
        gt_path = os.path.join(gt_labels_dir, bin_file)
        if not os.path.exists(bin_path) or not os.path.exists(gt_path):
            continue

        try:
            xyz = read_bin_file(bin_path, dtype=np.float32, shape=(-1, 4))[:, :3]
            gt_lbl = read_bin_file(gt_path, dtype=np.int32, shape=(-1,))
            if len(gt_lbl) != len(xyz):
                continue
        except Exception as e:
            tqdm.write(f"  Error {bin_file}: {e}")
            continue

        world = transform_points_to_world(xyz, position, quaternion, body_to_lidar_tf)
        if max_distance and max_distance > 0:
            mask = np.linalg.norm(world - position, axis=1) <= max_distance
            world, gt_lbl = world[mask], gt_lbl[mask]

        colors = common_labels_to_colors(apply_common_lut(gt_lbl, gt_common_lut))

        if voxel_size and voxel_size > 0 and len(world) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(world.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            world, colors = np.asarray(pcd.points), np.asarray(pcd.colors)

        all_points.append(world.astype(np.float64))
        all_colors.append(colors)

    if not all_points:
        return None, None
    return np.vstack(all_points), np.vstack(all_colors)


def build_multiclass_map_mcd(dataset_path, seq_name, body_to_lidar_tf,
                             labels_key="semkitti", learning_map_inv=None,
                             downsample_factor=1, voxel_size=0.1, max_distance=None,
                             max_scans=None, view_variance=False):
    """
    Load scans, GT labels, and multiclass confidence scores.
    Returns result dict with points, gt_labels, inf_labels, inf_variance,
    inf_uncertainty, n_classes -- or None.
    """
    infer_subdir = INFERRED_SUBDIRS.get(labels_key, f"cenet_{labels_key}")
    root_path = os.path.join(dataset_path, seq_name)
    data_dir = os.path.join(root_path, "lidar_bin/data")
    poses_file = os.path.join(root_path, "pose_inW.csv")
    gt_labels_dir = os.path.join(root_path, "gt_labels")
    multiclass_dir = os.path.join(
        root_path, "inferred_labels", infer_subdir, "multiclass_confidence_scores",
    )

    gt_common_lut = build_to_common_lut("mcd")
    inf_common_lut = build_to_common_lut(labels_key)

    for path, desc in [
        (data_dir, "Data"), (poses_file, "Poses"),
        (gt_labels_dir, "GT labels"), (multiclass_dir, "Multiclass"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: {desc} not found: {path}", file=sys.stderr)
            return None

    print(f"GT label directory: {gt_labels_dir}")
    print(f"Multiclass directory: {multiclass_dir}")

    poses = load_poses(poses_file)
    if not poses:
        print("ERROR: No poses loaded.", file=sys.stderr)
        return None

    sorted_indices = get_sorted_pose_indices(poses, downsample_factor, max_scans)

    all_points, all_gt, all_inf, all_var, all_unc = [], [], [], [], []
    n_classes = 0

    for pose_num in tqdm(sorted_indices, desc="Loading scans", unit="scan"):
        pose_data = poses[pose_num]
        position, quaternion = pose_data[0:3], pose_data[3:7]
        bin_file = f"{pose_num:010d}.bin"
        bin_path = os.path.join(data_dir, bin_file)
        gt_path = os.path.join(gt_labels_dir, bin_file)
        mc_path = os.path.join(multiclass_dir, bin_file)
        if not all(os.path.exists(p) for p in [bin_path, gt_path, mc_path]):
            continue

        try:
            xyz = read_bin_file(bin_path, dtype=np.float32, shape=(-1, 4))[:, :3]
            gt_lbl = read_bin_file(gt_path, dtype=np.int32, shape=(-1,))
            raw_probs = read_bin_file(mc_path, dtype=np.float16)
            n_points = len(xyz)
            n_classes = len(raw_probs) // n_points
            if len(gt_lbl) != n_points or len(raw_probs) != n_points * n_classes:
                continue
            probs = raw_probs.reshape(n_points, n_classes)
        except Exception as e:
            tqdm.write(f"  Error {bin_file}: {e}")
            continue

        gt_mapped = apply_common_lut(gt_lbl, gt_common_lut)
        class_idx = np.argmax(probs, axis=1)
        inf_raw = map_class_indices_to_labels(class_idx, learning_map_inv) if learning_map_inv else class_idx
        inf_mapped = apply_common_lut(inf_raw, inf_common_lut)

        variances = np.var(probs.astype(np.float32), axis=1)
        if view_variance:
            max_var = (n_classes - 1) / (n_classes ** 2) if n_classes > 1 else 1.0
            uncertainty = 1.0 - np.clip(variances / max_var, 0.0, 1.0)
        else:
            uncertainty = None

        world = transform_points_to_world(xyz, position, quaternion, body_to_lidar_tf)
        if max_distance and max_distance > 0:
            mask = np.linalg.norm(world - position, axis=1) <= max_distance
            world, gt_mapped, inf_mapped, variances = (
                world[mask], gt_mapped[mask], inf_mapped[mask], variances[mask],
            )
            if uncertainty is not None:
                uncertainty = uncertainty[mask]

        if voxel_size and voxel_size > 0 and len(world) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(world.astype(np.float64))
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            world_ds = np.asarray(pcd.points)
            if len(world_ds) == 0:
                continue
            _, idx = cKDTree(world).query(world_ds, k=1)
            gt_mapped, inf_mapped, variances = gt_mapped[idx], inf_mapped[idx], variances[idx]
            if uncertainty is not None:
                uncertainty = uncertainty[idx]
            world = world_ds

        all_points.append(world.astype(np.float64))
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
# Visualization
# ---------------------------------------------------------------------------

def add_osm_and_path(geoms, dataset_path, poses, args):
    """Add OSM overlay and/or drive path to the geometry list. Returns path z-offset."""
    path_z = 0.0

    if args.with_path or args.with_osm:
        path_xyz = build_path_from_poses(poses)
        if path_xyz is not None and len(path_xyz) >= 2:
            path_z = float(np.median(path_xyz[:, 2]))
            if args.with_path:
                g = create_path_geometry(path_xyz, thickness=args.path_thickness)
                if g:
                    geoms.append(g)
                    print(f"Path: {len(path_xyz)} poses")
        elif args.with_path:
            print("Warning: not enough poses for path.", file=sys.stderr)

    if args.with_osm:
        osm_file = get_mcd_osm_path(dataset_path)
        if os.path.isfile(osm_file):
            print(f"Loading OSM: {osm_file}")
            loader = OSMLoader(osm_file, origin_latlon=MCD_ORIGIN_LATLON)
            osm_geoms = loader.get_geometries(
                z_offset=path_z, thickness=args.osm_thickness, buildings_only=not args.osm_all,
            )
            if osm_geoms:
                geoms.extend(osm_geoms)
                print(f"OSM: {len(osm_geoms)} groups")
        else:
            print(f"OSM file not found: {osm_file}", file=sys.stderr)

    return path_z


def show_point_cloud(pcd, extra_geoms):
    """Display point cloud with optional extra geometries."""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    for geom in extra_geoms:
        vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    vis.run()
    vis.destroy_window()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize MCD semantic map with Open3D.",
    )
    parser.add_argument("--dataset-path", type=str,
                        default=os.path.join(OSMBKI_ROOT, "example_data", "mcd"),
                        help="Root directory of the MCD dataset")
    parser.add_argument("--seq", type=str, nargs="+", default=["kth_day_09"],
                        help="Sequence name(s) (default: kth_day_09)")
    parser.add_argument("--downsample-factor", type=int, default=1,
                        help="Process every Nth scan (default: 1)")
    parser.add_argument("--voxel-size", type=float, default=0.5,
                        help="Voxel size in meters for downsampling (default: 0.5)")
    parser.add_argument("--max-distance", type=float, default=200.0,
                        help="Max distance from sensor per scan in meters (default: 200.0)")
    parser.add_argument("--max-scans", type=int, default=10000,
                        help="Max number of scans to process (default: 10000)")
    parser.add_argument("--calib", type=str, default=None,
                        help="Path to calibration YAML (default: <dataset-path>/hhs_calib.yaml)")
    parser.add_argument("--inferred-labels", type=str, default="semkitti",
                        choices=list(INFERRED_LABEL_CONFIGS.keys()),
                        help="Inference network label set (default: semkitti)")
    parser.add_argument("--use-multiclass", action="store_true",
                        help="Load multiclass confidence scores; enables accuracy analysis")
    parser.add_argument("--variance", action="store_true",
                        help="Color by variance (Viridis: yellow=uncertain, dark=confident)")
    parser.add_argument("--view", type=str, choices=["all", "correct", "incorrect"], default="all",
                        help="Filter displayed points (default: all)")
    parser.add_argument("--with-osm", action="store_true", help="Overlay OSM map")
    parser.add_argument("--osm-thickness", type=float, default=2.0, help="OSM line thickness in meters")
    parser.add_argument("--osm-all", action="store_true", help="Show all OSM features (default: buildings only)")
    parser.add_argument("--with-path", action="store_true", help="Show drive path")
    parser.add_argument("--path-thickness", type=float, default=1.5, help="Path line thickness in meters")
    args = parser.parse_args()

    max_scans = args.max_scans if args.max_scans else None

    calib_path = args.calib or os.path.join(args.dataset_path, "hhs_calib.yaml")
    if not os.path.exists(calib_path):
        print(f"ERROR: Calibration file not found: {calib_path}", file=sys.stderr)
        sys.exit(1)
    body_to_lidar_tf = load_body_to_lidar_tf(calib_path)
    print(f"Loaded body-to-LiDAR transform from {calib_path}")

    if args.use_multiclass:
        label_cfg = load_label_config(INFERRED_LABEL_CONFIGS[args.inferred_labels])

        for seq_name in args.seq:
            result = build_multiclass_map_mcd(
                args.dataset_path, seq_name, body_to_lidar_tf,
                labels_key=args.inferred_labels,
                learning_map_inv=label_cfg["learning_map_inv"],
                downsample_factor=args.downsample_factor, voxel_size=args.voxel_size,
                max_distance=args.max_distance, max_scans=max_scans,
                view_variance=args.variance,
            )
            if result is None:
                print(f"No points loaded for {seq_name}.", file=sys.stderr)
                continue

            points = result["points"]
            print(f"Total points: {len(points)}")

            correct, _ = run_accuracy_analysis(result)

            if args.view == "correct":
                mask = correct
            elif args.view == "incorrect":
                mask = ~correct
            else:
                mask = np.ones(len(points), dtype=bool)
            print(f"Showing {int(np.sum(mask))} points (view={args.view})")

            pts = points[mask]
            if args.variance and result["inf_uncertainty"] is not None:
                colors = scalar_to_viridis_rgb(result["inf_uncertainty"][mask], normalize_range=False)
            else:
                colors = common_labels_to_colors(result["inf_labels"][mask])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            root_path = os.path.join(args.dataset_path, seq_name)
            poses = load_poses(os.path.join(root_path, "pose_inW.csv"))
            extra_geoms = []
            add_osm_and_path(extra_geoms, args.dataset_path, poses, args)
            show_point_cloud(pcd, extra_geoms)
    else:
        for seq_name in args.seq:
            points, colors = build_semantic_map_mcd(
                args.dataset_path, seq_name, body_to_lidar_tf,
                downsample_factor=args.downsample_factor, voxel_size=args.voxel_size,
                max_distance=args.max_distance, max_scans=max_scans,
            )
            if points is None:
                print(f"No points loaded for {seq_name}.", file=sys.stderr)
                continue

            print(f"Total points: {len(points)}")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            root_path = os.path.join(args.dataset_path, seq_name)
            poses = load_poses(os.path.join(root_path, "pose_inW.csv"))
            extra_geoms = []
            add_osm_and_path(extra_geoms, args.dataset_path, poses, args)
            show_point_cloud(pcd, extra_geoms)
