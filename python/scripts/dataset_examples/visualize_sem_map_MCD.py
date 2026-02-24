#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr
import matplotlib.pyplot as plt
import open3d as o3d
import yaml
from tqdm import tqdm

from utils import *
from label_mappings import build_to_common_lut, apply_common_lut, common_labels_to_colors, IGNORE_LABELS

DEFAULT_CONFIG_PATH = INFERRED_LABEL_CONFIGS["mcd"]

def load_body_to_lidar_tf(calib_path, sensor_key="os_sensor"):
    """Load the body-to-LiDAR 4x4 transform from a calibration YAML (body -> sensor_key -> T)."""
    with open(calib_path, "r") as f:
        cfg = yaml.safe_load(f)
    T = cfg["body"][sensor_key]["T"]
    return np.array(T, dtype=np.float64)

def load_label_config(config_path):
    """
    Load label definitions (names, color_map, learning_map_inv) from MCD YAML config.
    Returns dict with labels, color_map_rgb (id -> (r,g,b) 0-255), learning_map_inv.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    labels = cfg.get("labels", {})
    color_map_bgr = cfg.get("color_map", {})
    learning_map_inv = cfg.get("learning_map_inv", {})
    color_map_rgb = {}
    for k, bgr in color_map_bgr.items():
        color_map_rgb[int(k)] = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
    return {
        "labels": {int(k): v for k, v in labels.items()},
        "color_map_rgb": color_map_rgb,
        "learning_map_inv": {int(k): int(v) for k, v in learning_map_inv.items()},
    }


def get_label_id_to_class_index(learning_map_inv):
    """Map semantic label ID -> class index for indexing multiclass_probs."""
    return {int(v): int(k) for k, v in learning_map_inv.items()}


def _apply_value_curve(t, value_floor=0.0, gamma=1.0):
    """
    Map values in [0, 1] to output brightness for better visibility.
    value_floor: minimum output (e.g. 0.15 so low confidence isn't pure black).
    gamma: < 1 brightens mid-tones, > 1 darkens them.
    """
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    if gamma != 1.0:
        t = np.power(t, gamma)
    if value_floor > 0:
        t = value_floor + (1.0 - value_floor) * t
    return t


def labels_to_colors(labels, label_id_to_color, confidences=None, value_floor=0.15, gamma=1.0):
    """
    Convert semantic label IDs to RGB colors. If confidences is given, modulate brightness
    (lower confidence = darker). Normalizes confidence range per batch; optional floor/gamma
    improve visibility (avoid pure black, better mid-tone contrast).
    """
    n = len(labels)
    colors = np.zeros((n, 3), dtype=np.float32)
    if confidences is None:
        confidences = np.ones(n, dtype=np.float32)
    else:
        confidences = np.asarray(confidences, dtype=np.float32).reshape(-1)
    max_c = np.max(confidences)
    min_c = np.min(confidences)
    rng = max_c - min_c
    if rng <= 0:
        # Uniform confidences (e.g. confidences=None): full brightness
        t = np.ones(n, dtype=np.float32)
    else:
        t = (confidences - min_c) / rng
        t = _apply_value_curve(t, value_floor=value_floor, gamma=gamma)
    for i, label_id in enumerate(labels):
        lid = int(label_id)
        if lid in label_id_to_color:
            base = np.array(label_id_to_color[lid], dtype=np.float32) / 255.0
            colors[i] = base * t[i]
        else:
            colors[i] = [0.5, 0.5, 0.5]
    return colors


def single_label_confidence_to_colors(confidences, label_id, label_id_to_color,
                                      normalize_range=True, value_floor=0.12, gamma=0.65,
                                      grayscale=False):
    """
    Color by one label's confidence.
    - grayscale=False: high = label color, low = dark (with value_floor).
    - grayscale=True: high = white, low = black (use value_floor=0 for full range). Scene background gray is set in the viewer.
    - normalize_range, value_floor, gamma: as in labels_to_colors.
    """
    colors = np.zeros((len(confidences), 3), dtype=np.float32)
    c = np.asarray(confidences, dtype=np.float32).reshape(-1)
    if normalize_range:
        c_min, c_max = np.min(c), np.max(c)
        rng = c_max - c_min
        if rng <= 0:
            rng = 1.0
        c = (c - c_min) / rng
    t = _apply_value_curve(c, value_floor=value_floor, gamma=gamma)
    if grayscale:
        # White (high) to black (low); use value_floor=0 for full range
        colors[:] = t[:, np.newaxis]
    else:
        if label_id not in label_id_to_color:
            return colors
        base = np.array(label_id_to_color[label_id], dtype=np.float32) / 255.0
        colors[:] = base * t[:, np.newaxis]
    return colors


def _parse_single_label(single_label_arg, label_config):
    """Parse --single-label: accept label name or id. Return (label_id, label_name) or (None, None)."""
    labels_by_id = label_config["labels"]
    labels_by_name = {v: k for k, v in labels_by_id.items()}
    try:
        lid = int(single_label_arg)
        if lid in labels_by_id:
            return lid, labels_by_id[lid]
        return None, None
    except ValueError:
        pass
    for name, lid in labels_by_name.items():
        if name.lower() == single_label_arg.lower():
            return lid, name
    return None, None


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


def transform_points_to_world(points_xyz, position, quaternion, body_to_lidar_tf=None):
    """
    Transform points from lidar frame to world frame using pose.
    
    Args:
        points_xyz: (N, 3) array of points in lidar frame
        position: [x, y, z] translation (body frame position in world)
        quaternion: [qx, qy, qz, qw] rotation quaternion (body frame orientation in world)
        body_to_lidar_tf: Optional 4x4 transformation matrix from body to lidar frame
    
    Returns:
        world_points: (N, 3) array of points in world frame
    """
    # Create rotation matrix from quaternion (body frame orientation)
    body_rotation_matrix = R.from_quat(quaternion).as_matrix()
    
    # Create 4x4 transformation matrix for body frame in world
    body_to_world = np.eye(4)
    body_to_world[:3, :3] = body_rotation_matrix
    body_to_world[:3, 3] = position
    
    # If body_to_lidar transform is provided, compose the transformations
    # world_to_lidar = world_to_body * body_to_lidar
    # So: lidar_to_world = (body_to_lidar)^-1 * body_to_world
    if body_to_lidar_tf is not None:
        # Transform from body to lidar, then from body to world
        # T_lidar_to_world = T_body_to_world * T_lidar_to_body
        # T_lidar_to_body = inv(T_body_to_lidar)
        lidar_to_body = np.linalg.inv(body_to_lidar_tf)
        transform_matrix = body_to_world @ lidar_to_body
    else:
        transform_matrix = body_to_world
    
    # Transform points to world coordinates
    points_homogeneous = np.hstack(
        [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)]
    )
    world_points = (transform_matrix @ points_homogeneous.T).T
    world_points_xyz = world_points[:, :3]
    
    return world_points_xyz


# ---------------------------------------------------------------------------
# OSM overlay helpers (MCD-specific)
# ---------------------------------------------------------------------------

def get_mcd_osm_path(dataset_path, osm_filename="kth.osm"):
    """Return path to OSM file for the MCD dataset."""
    return os.path.join(dataset_path, osm_filename)


def build_path_from_poses(poses):
    """Extract world-frame path positions from MCD poses. Returns (N,3) array or None."""
    if not poses:
        return None
    positions = []
    for num in sorted(poses.keys()):
        p = poses[num]
        positions.append([p[0], p[1], p[2]])
    if len(positions) < 2:
        return None
    return np.array(positions, dtype=np.float64)


def build_semantic_map_mcd(dataset_path, seq_name, body_to_lidar_tf,
                           max_scans=None, downsample_factor=1, voxel_size=0.1, max_distance=None,
                           with_osm=False, osm_thickness=2.0, osm_all=False,
                           with_path=False, path_thickness=1.5):
    """
    Build a one-hot GT semantic map for MCD: load GT labels, transform to world, color by common taxonomy.
    """
    root_path = os.path.join(dataset_path, seq_name)
    data_dir = os.path.join(root_path, "lidar_bin/data")
    poses_file = os.path.join(root_path, "pose_inW.csv")
    gt_labels_dir = os.path.join(root_path, "gt_labels")

    gt_common_lut = build_to_common_lut("mcd")

    if not os.path.exists(data_dir) or not os.path.exists(poses_file):
        print("ERROR: Data directory or poses file not found.")
        return
    if not os.path.exists(gt_labels_dir):
        print(f"ERROR: GT labels directory not found: {gt_labels_dir}")
        return

    poses = load_poses(poses_file)
    if not poses:
        print("ERROR: No poses loaded.")
        return

    sorted_pose_indices = sorted(poses.keys())
    if downsample_factor > 1:
        sorted_pose_indices = sorted_pose_indices[::downsample_factor]
    if max_scans:
        sorted_pose_indices = sorted_pose_indices[:max_scans]

    all_points = []
    all_colors = []

    for pose_num in tqdm(sorted_pose_indices, desc="Loading scans", unit="scan"):
        pose_data = poses[pose_num]
        position = pose_data[0:3]
        quaternion = pose_data[3:7]
        bin_file = f"{pose_num:010d}.bin"
        bin_path = os.path.join(data_dir, bin_file)
        gt_path = os.path.join(gt_labels_dir, bin_file)
        if not os.path.exists(bin_path) or not os.path.exists(gt_path):
            continue
        try:
            points = read_bin_file(bin_path, dtype=np.float32, shape=(-1, 4))
            points_xyz = points[:, :3]
            gt_lbl_raw = read_bin_file(gt_path, dtype=np.int32, shape=(-1))
            if len(gt_lbl_raw) != len(points_xyz):
                continue
        except Exception as e:
            tqdm.write(f"  Error {bin_file}: {e}")
            continue

        world = transform_points_to_world(points_xyz, position, quaternion, body_to_lidar_tf)
        if max_distance is not None and max_distance > 0:
            dist_mask = np.linalg.norm(world - position, axis=1) <= max_distance
            world = world[dist_mask]
            gt_lbl_raw = gt_lbl_raw[dist_mask]

        colors = common_labels_to_colors(apply_common_lut(gt_lbl_raw, gt_common_lut))

        if voxel_size is not None and voxel_size > 0 and len(world) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(world.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            world = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)

        all_points.append(world.astype(np.float64))
        all_colors.append(colors)

    if not all_points:
        print("ERROR: No points accumulated.")
        return

    points = np.vstack(all_points)
    colors = np.vstack(all_colors)
    print(f"Total points: {len(points)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    extra_geoms = []
    path_z_default = 0.0

    if with_path or with_osm:
        path_xyz = build_path_from_poses(poses)
        if path_xyz is not None and len(path_xyz) >= 2:
            path_z_default = float(np.median(path_xyz[:, 2]))
            if with_path:
                path_geom = create_path_geometry(path_xyz, color=(0.0, 0.8, 0.0), thickness=path_thickness)
                if path_geom is not None:
                    extra_geoms.append(path_geom)
                    print(f"Path: {len(path_xyz)} poses")
        else:
            if with_path:
                print("Warning: not enough poses for path geometry.", file=sys.stderr)

    if with_osm:
        osm_file = get_mcd_osm_path(dataset_path)
        if os.path.isfile(osm_file):
            print(f"Loading OSM: {osm_file}")
            loader = OSMLoader(osm_file, origin_latlon=MCD_ORIGIN_LATLON)
            osm_geoms = loader.get_geometries(
                z_offset=path_z_default, thickness=osm_thickness, buildings_only=not osm_all,
            )
            if osm_geoms:
                extra_geoms.extend(osm_geoms)
                kind = "building" if not osm_all else "geometry"
                print(f"OSM: {len(osm_geoms)} {kind} groups")
        else:
            print(f"OSM file not found: {osm_file}", file=sys.stderr)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    for geom in extra_geoms:
        vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    vis.run()
    vis.destroy_window()


def compare_gt_inferred_map(dataset_path, seq_name, label_config, body_to_lidar_tf,
                            labels_key="semkitti", inf_learning_map_inv=None,
                            max_scans=None, downsample_factor=1, voxel_size=0.1, max_distance=None,
                            view_mode="all", view_variance=False,
                            with_osm=False, osm_thickness=2.0, osm_all=False,
                            with_path=False, path_thickness=1.5):
    """
    Build GT map (world points + gt labels) and inferred map (world points + dominant class).
    GT labels are MCD raw IDs; inferred labels come from the network specified by labels_key.
    """
    infer_subdir = INFERRED_SUBDIRS.get(labels_key, f"cenet_{labels_key}")
    root_path = os.path.join(dataset_path, seq_name)
    data_dir = os.path.join(root_path, "lidar_bin/data")
    poses_file = os.path.join(root_path, "pose_inW.csv")
    gt_labels_dir = os.path.join(root_path, "gt_labels")
    multiclass_dir = os.path.join(root_path, "inferred_labels", infer_subdir, "multiclass_confidence_scores")

    learning_map_inv = inf_learning_map_inv if inf_learning_map_inv is not None else label_config["learning_map_inv"]
    gt_common_lut = build_to_common_lut("mcd")
    inf_common_lut = build_to_common_lut(labels_key)

    if not os.path.exists(data_dir) or not os.path.exists(poses_file):
        print("ERROR: Data directory or poses file not found.")
        return
    if not os.path.exists(gt_labels_dir):
        print(f"ERROR: GT labels directory not found: {gt_labels_dir}")
        return
    if not os.path.exists(multiclass_dir):
        print(f"ERROR: Multiclass directory not found: {multiclass_dir}")
        return

    poses = load_poses(poses_file)
    if not poses:
        print("ERROR: No poses loaded.")
        return

    sorted_pose_indices = sorted(poses.keys())
    if downsample_factor > 1:
        sorted_pose_indices = sorted_pose_indices[::downsample_factor]
    if max_scans:
        sorted_pose_indices = sorted_pose_indices[:max_scans]

    all_gt_points = []
    all_gt_labels = []
    all_inf_points = []
    all_inf_labels = []
    all_inf_variance = []
    all_inf_uncertainty = []

    for pose_num in tqdm(sorted_pose_indices, desc="Loading scans", unit="scan"):
        pose_data = poses[pose_num]
        position = pose_data[0:3]
        quaternion = pose_data[3:7]
        bin_file = f"{pose_num:010d}.bin"
        bin_path = os.path.join(data_dir, bin_file)
        gt_path = os.path.join(gt_labels_dir, bin_file)
        multiclass_path = os.path.join(multiclass_dir, bin_file)
        if not os.path.exists(bin_path) or not os.path.exists(gt_path) or not os.path.exists(multiclass_path):
            continue
        try:
            points = read_bin_file(bin_path, dtype=np.float32, shape=(-1, 4))
            points_xyz = points[:, :3]
            gt_lbl_raw = read_bin_file(gt_path, dtype=np.int32, shape=(-1))
            raw_probs = read_bin_file(multiclass_path, dtype=np.float16)
            n_points = len(points_xyz)
            n_classes = len(raw_probs) // n_points
            if len(gt_lbl_raw) != n_points or len(raw_probs) != n_points * n_classes:
                continue
            multiclass_probs = raw_probs.reshape(n_points, n_classes)
            gt_lbl = apply_common_lut(gt_lbl_raw, gt_common_lut)
            class_indices = np.argmax(multiclass_probs, axis=1)
            inf_lbl = apply_common_lut(
                map_class_indices_to_labels(class_indices, learning_map_inv), inf_common_lut,
            )
            # Variance of class probs (always store for average-variance report)
            variances = np.var(multiclass_probs.astype(np.float32), axis=1)
            # Uncertainty for optional variance coloring (yellow=uncertain, dark=confident)
            if view_variance:
                max_var = (n_classes - 1) / (n_classes ** 2) if n_classes > 1 else 1.0
                scaled = np.clip(variances / max_var, 0.0, 1.0)
                uncertainty = 1.0 - scaled
            else:
                uncertainty = None
        except Exception as e:
            tqdm.write(f"  Error {bin_file}: {e}")
            continue

        world = transform_points_to_world(points_xyz, position, quaternion, body_to_lidar_tf)
        if max_distance is not None and max_distance > 0:
            dist_mask = np.linalg.norm(world - position, axis=1) <= max_distance
            world = world[dist_mask]
            gt_lbl = gt_lbl[dist_mask]
            inf_lbl = inf_lbl[dist_mask]
            variances = variances[dist_mask]
            if uncertainty is not None:
                uncertainty = uncertainty[dist_mask]
        if voxel_size is not None and voxel_size > 0 and len(world) > 0:
            scan_pcd = o3d.geometry.PointCloud()
            scan_pcd.points = o3d.utility.Vector3dVector(world.astype(np.float64))
            scan_pcd = scan_pcd.voxel_down_sample(voxel_size=voxel_size)
            world_ds = np.asarray(scan_pcd.points)
            if len(world_ds) == 0:
                continue
            tree_scan = cKDTree(world)
            _, idx = tree_scan.query(world_ds, k=1)
            gt_lbl = gt_lbl[idx]
            inf_lbl = inf_lbl[idx]
            variances = variances[idx]
            if uncertainty is not None:
                uncertainty = uncertainty[idx]
            world = world_ds
        all_gt_points.append(world.astype(np.float64))
        all_gt_labels.append(gt_lbl)
        all_inf_points.append(world.astype(np.float64))
        all_inf_labels.append(inf_lbl)
        all_inf_variance.append(variances)
        if view_variance:
            all_inf_uncertainty.append(uncertainty)

    if not all_gt_points:
        print("ERROR: No points accumulated.")
        return
    gt_points = np.vstack(all_gt_points)
    gt_labels = np.concatenate(all_gt_labels)
    inf_points = np.vstack(all_inf_points)
    inf_labels = np.concatenate(all_inf_labels)
    inf_variance = np.concatenate(all_inf_variance)
    inf_uncertainty = np.concatenate(all_inf_uncertainty) if all_inf_uncertainty else None
    assert len(gt_points) == len(gt_labels) == len(inf_points) == len(inf_labels) == len(inf_variance)

    print("Building KD-tree on GT map...")
    tree = cKDTree(gt_points)
    print("Querying nearest GT neighbor for each inferred point...")
    _, nn_idx = tree.query(inf_points, k=1)
    gt_label_matched = gt_labels[nn_idx]

    # Mask to exclude ignored labels from accuracy metrics
    ignore_set = set(IGNORE_LABELS)
    valid = np.array([g not in ignore_set for g in gt_label_matched], dtype=bool)
    n_ignored = int(np.sum(~valid))
    print(f"Ignoring {n_ignored} points with GT label in {IGNORE_LABELS}")

    correct_all = (inf_labels == gt_label_matched)
    correct = correct_all[valid]
    n_correct = int(np.sum(correct))
    n_total = len(correct)
    print(f"Accuracy: {n_correct}/{n_total} ({100.0 * n_correct / n_total:.2f}%)")

    # Uncertainty from variance (0=confident, 1=uncertain) for all points
    n_classes = len(learning_map_inv)
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
    print(f"Keeping points with uncertainty <= median: n={n_kept} (removed {n_removed} high-uncertainty points)")
    print(f"Accuracy after removing high-uncertainty points: {n_correct_kept}/{n_kept} ({accuracy_kept:.2f}%)")

    # --- Uncertainty calibration: is the model's uncertainty predictive of errors? ---
    incorrect = np.asarray(~correct, dtype=np.float64)
    rho, p_val = spearmanr(uncertainty, incorrect)
    print(f"\n--- Uncertainty calibration ---")
    print(f"Spearman(uncertainty, incorrect): rho = {rho:.4f} (p = {p_val:.2e})")

    # AUROC: can uncertainty discriminate correct vs incorrect? (1 = perfect, 0.5 = random)
    n_incorrect = int(np.sum(incorrect))
    n_correct_count = n_total - n_incorrect
    if n_incorrect > 0 and n_correct_count > 0:
        ranks = rankdata(uncertainty)
        S = np.sum(ranks[~correct])
        auroc = (S - n_incorrect * (n_incorrect + 1) / 2) / (n_incorrect * n_correct_count)
        print(f"AUROC(uncertainty -> predicts error): {auroc:.4f}")
    else:
        auroc = None

    # Accuracy by uncertainty bin (well-calibrated = accuracy decreases as uncertainty increases)
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

    if view_mode == "correct":
        mask = correct_all
    elif view_mode == "incorrect":
        mask = ~correct_all
    else:
        mask = np.ones(len(correct_all), dtype=bool)
    n_show = int(np.sum(mask))
    print(f"Showing {n_show} points (view={view_mode})")
    avg_var = float(np.mean(inf_variance[mask]))
    print(f"Average variance ({view_mode}): {avg_var:.6f}")

    # Plot variance distribution for the selected subset
    var_subset = inf_variance[mask]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(var_subset, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(avg_var, color="coral", linewidth=2, label=f"Mean = {avg_var:.4f}")
    ax.set_xlabel("Variance of class probabilities")
    ax.set_ylabel("Count")
    ax.set_title(f"Variance distribution (view={view_mode}, n={n_show})")
    ax.legend()
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)

    pts = inf_points[mask]
    lbl = inf_labels[mask]
    if view_variance and inf_uncertainty is not None:
        unc = inf_uncertainty[mask]
        colors = scalar_to_viridis_rgb(unc, normalize_range=False)
    else:
        colors = common_labels_to_colors(lbl)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    extra_geoms = []
    path_xyz = None
    path_z_default = 0.0

    # -- Drive path (also used to set OSM z-offset) -----------------------
    if with_path or with_osm:
        path_xyz = build_path_from_poses(poses)
        if path_xyz is not None and len(path_xyz) >= 2:
            path_z_default = float(np.median(path_xyz[:, 2]))
            if with_path:
                path_geom = create_path_geometry(path_xyz, color=(0.0, 0.8, 0.0), thickness=path_thickness)
                if path_geom is not None:
                    extra_geoms.append(path_geom)
                    print(f"Path: {len(path_xyz)} poses")
        else:
            if with_path:
                print("Warning: not enough poses for path geometry.", file=sys.stderr)

    # -- OSM overlay ------------------------------------------------------
    if with_osm:
        osm_file = get_mcd_osm_path(dataset_path)
        if os.path.isfile(osm_file):
            print(f"Loading OSM: {osm_file}")
            loader = OSMLoader(osm_file, origin_latlon=MCD_ORIGIN_LATLON)
            osm_geoms = loader.get_geometries(
                z_offset=path_z_default, thickness=osm_thickness, buildings_only=not osm_all,
            )
            if osm_geoms:
                extra_geoms.extend(osm_geoms)
                kind = "building" if not osm_all else "geometry"
                print(f"OSM: {len(osm_geoms)} {kind} groups")
        else:
            print(f"OSM file not found: {osm_file}", file=sys.stderr)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    for geom in extra_geoms:
        vis.add_geometry(geom)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize MCD semantic map with Open3D. Default: one-hot GT labels. "
                    "With --use-multiclass: compare GT vs inferred, show accuracy and uncertainty analysis."
    )
    parser.add_argument("--dataset-path", type=str,
                        default=os.path.join(OSMBKI_ROOT, "example_data", "mcd"))
    parser.add_argument("--seq", type=str, nargs="+", default=["kth_day_09"], help="Sequence name(s)")
    parser.add_argument("--max-scans", type=int, default=10000, help="Max scans to process (0 or None for all)")
    parser.add_argument("--downsample-factor", type=int, default=1, help="Process every Nth scan")
    parser.add_argument("--voxel-size", type=float, default=0.5, help="Voxel size in meters (per-scan downsampling)")
    parser.add_argument("--max-distance", type=float, default=200.0, help="Max distance from pose to keep points (m)")
    parser.add_argument("--config", type=str, default=None, help="Path to MCD label config YAML (default: labels_mcd.yaml)")
    parser.add_argument("--calib", type=str, default=None,
                        help=f"Path to calibration YAML (default: <dataset-path>/hhs_calib.yaml)")
    parser.add_argument("--inferred-labels", type=str, choices=list(INFERRED_LABEL_CONFIGS.keys()), default="semkitti",
                        help="Inference network label set (default: semkitti)")
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
    parser.add_argument("--with-path", action="store_true", help="Show drive path from poses")
    parser.add_argument("--path-thickness", type=float, default=1.5, help="Path line thickness in meters (default: 1.5)")
    args = parser.parse_args()

    max_scans = args.max_scans if args.max_scans else None

    calib_path = args.calib or os.path.join(args.dataset_path, "hhs_calib.yaml")
    if not os.path.exists(calib_path):
        print(f"ERROR: Calibration file not found: {calib_path}")
        sys.exit(1)
    body_to_lidar_tf = load_body_to_lidar_tf(calib_path)
    print(f"Loaded body-to-LiDAR transform from {calib_path}")

    if args.use_multiclass:
        config_path = args.config or DEFAULT_CONFIG_PATH
        if not os.path.exists(config_path):
            print(f"ERROR: Config not found: {config_path}")
            sys.exit(1)
        label_config = load_label_config(config_path)

        inf_label_config_path = INFERRED_LABEL_CONFIGS[args.inferred_labels]
        inf_label_config = load_label_config(inf_label_config_path)
        inf_learning_map_inv = inf_label_config["learning_map_inv"]

        for seq_name in args.seq:
            compare_gt_inferred_map(
                args.dataset_path,
                seq_name,
                label_config,
                body_to_lidar_tf,
                labels_key=args.inferred_labels,
                inf_learning_map_inv=inf_learning_map_inv,
                max_scans=max_scans,
                downsample_factor=args.downsample_factor,
                voxel_size=args.voxel_size,
                max_distance=args.max_distance,
                view_mode=args.view,
                view_variance=args.variance,
                with_osm=args.with_osm,
                osm_thickness=args.osm_thickness,
                osm_all=args.osm_all,
                with_path=args.with_path,
                path_thickness=args.path_thickness,
            )
    else:
        for seq_name in args.seq:
            build_semantic_map_mcd(
                args.dataset_path,
                seq_name,
                body_to_lidar_tf,
                max_scans=max_scans,
                downsample_factor=args.downsample_factor,
                voxel_size=args.voxel_size,
                max_distance=args.max_distance,
                with_osm=args.with_osm,
                osm_thickness=args.osm_thickness,
                osm_all=args.osm_all,
                with_path=args.with_path,
                path_thickness=args.path_thickness,
            )