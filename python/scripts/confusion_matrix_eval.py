#!/usr/bin/env python3
"""
Confusion Matrix Evaluation: Semantic Method vs Ground Truth.

Trains the BKI semantic pipeline on scan data and evaluates predicted labels
against ground truth on held-out test scans. Produces a confusion matrix figure.

Usage (matches run_cmd.sh parameters):
    python python/scripts/confusion_matrix_eval.py \\
        --config cpp/osm_bki/configs/mcd_config.yaml \\
        --osm example_data/kth.osm \\
        --calib example_data/hhs_calib.yaml \\
        --scan-dir /path/to/lidar_bin/data/ \\
        --label-dir /path/to/inferred_labels/ \\
        --gt-dir /path/to/gt_labels/ \\
        --pose /path/to/pose_inW.csv \\
        --max-scans 100 --offset 1 --test-fraction 1 \\
        --prior-delta 0.5 --osm-prior-strength 0.01 \\
        --resolution 1.0 --l-scale 3.0 \\
        --output confusion_matrix.png
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

import osm_bki_cpp

from script_utils import ConfigReader, (
    load_body_to_lidar, load_poses_csv, transform_points_to_world,
    load_scan, load_labels, find_label_file, get_frame_number,
)


def compute_confusion_matrix_manual(pred, gt, labels):
    """Compute confusion matrix without sklearn. pred/gt: 1D arrays; labels: sorted class IDs."""
    pred = np.asarray(pred, dtype=np.int64)
    gt = np.asarray(gt, dtype=np.int64)
    label_to_idx = {c: i for i, c in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)
    for p, g in zip(pred.flat, gt.flat):
        if p in label_to_idx and g in label_to_idx:
            cm[label_to_idx[g], label_to_idx[p]] += 1
    return cm


def plot_confusion_matrix(cm, class_names, output_path=None, normalize=None, title=None):
    """
    Plot confusion matrix as heatmap.

    cm: (n_classes, n_classes) array [rows=GT, cols=pred]
    class_names: list of str for axis labels
    normalize: 'true' (rows), 'pred' (cols), or None (raw counts)
    """
    if normalize == 'true':
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm.astype(float) / row_sums, 0)
        fmt = '.2f'
        cbar_label = 'Fraction (by GT)'
    elif normalize == 'pred':
        col_sums = cm.sum(axis=0, keepdims=True)
        cm_plot = np.where(col_sums > 0, cm.astype(float) / col_sums, 0)
        fmt = '.2f'
        cbar_label = 'Fraction (by Pred)'
    else:
        cm_plot = cm  # keep as int for 'd' format
        fmt = 'd'
        cbar_label = 'Count'

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': cbar_label},
                linewidths=0.5, linecolor='gray')

    ax.set_xlabel('Predicted (Semantic Method)')
    ax.set_ylabel('Ground Truth')
    ax.set_title(title or 'Confusion Matrix: Semantic Method vs Ground Truth')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate semantic BKI vs ground truth and produce confusion matrix figure."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config (e.g. mcd_config.yaml)")
    parser.add_argument("--osm", required=True, help="Path to OSM geometries (.bin or .osm)")
    parser.add_argument("--calib", required=True, help="Path to hhs_calib.yaml (body→LiDAR)")
    parser.add_argument("--scan-dir", required=True, help="Directory of .bin scans")
    parser.add_argument("--label-dir", required=True, help="Directory of predicted labels (.label/.bin)")
    parser.add_argument("--gt-dir", required=True, help="Directory of ground-truth labels")
    parser.add_argument("--pose", required=True, help="Pose CSV (num,t,x,y,z,qx,qy,qz,qw)")
    parser.add_argument("--max-scans", type=int, default=None, help="Cap on number of scans")
    parser.add_argument("--offset", type=int, default=1,
                        help="Train on every Nth scan; test on in-between (N>=1)")
    parser.add_argument("--test-fraction", type=float, default=1.0,
                        help="Fraction of test scans to evaluate (0 < f <= 1)")
    parser.add_argument("--prior-delta", type=float, default=0.5, help="BKI prior_delta")
    parser.add_argument("--osm-prior-strength", type=float, default=0.01,
                        help="OSM prior strength (>0 enables seeding)")
    parser.add_argument("--resolution", type=float, default=1.0, help="BKI resolution")
    parser.add_argument("--l-scale", type=float, default=3.0, help="BKI l_scale")
    parser.add_argument("--sigma-0", type=float, default=1.0, help="BKI sigma_0")
    parser.add_argument("--alpha0", type=float, default=1.0, help="BKI alpha0")
    parser.add_argument("--disable-osm-fallback", action="store_true",
                        help="Disable OSM fallback during inference")
    parser.add_argument("--init-rel-pos", type=float, nargs=3, default=None,
                        metavar=('X', 'Y', 'Z'),
                        help="World-frame initial position (init_rel_pos_day_06)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output figure path (e.g. confusion_matrix.png)")
    parser.add_argument("--normalize", choices=['true', 'pred'], default=None,
                        help="Normalize matrix: 'true' by GT rows, 'pred' by pred cols")
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir)
    label_dir = Path(args.label_dir)
    gt_dir = Path(args.gt_dir)

    if not scan_dir.exists():
        print(f"ERROR: Scan dir not found: {scan_dir}", file=sys.stderr)
        return 1
    if not label_dir.exists():
        print(f"ERROR: Label dir not found: {label_dir}", file=sys.stderr)
        return 1
    if not gt_dir.exists():
        print(f"ERROR: GT dir not found: {gt_dir}", file=sys.stderr)
        return 1
    if not os.path.exists(args.pose):
        print(f"ERROR: Pose file not found: {args.pose}", file=sys.stderr)
        return 1
    if not os.path.exists(args.config):
        print(f"ERROR: Config not found: {args.config}", file=sys.stderr)
        return 1
    if not os.path.exists(args.osm):
        print(f"ERROR: OSM file not found: {args.osm}", file=sys.stderr)
        return 1

    scan_files = sorted(scan_dir.glob("*.bin"))
    if not scan_files:
        print(f"No .bin scans in {scan_dir}", file=sys.stderr)
        return 1

    if args.max_scans is not None and args.max_scans > 0:
        scan_files = scan_files[:args.max_scans]
    if args.offset < 1:
        print("ERROR: --offset must be >= 1", file=sys.stderr)
        return 1
    if args.test_fraction <= 0.0 or args.test_fraction > 1.0:
        print("ERROR: --test-fraction must be in (0, 1]", file=sys.stderr)
        return 1

    n_total = len(scan_files)
    if args.offset == 1:
        train_files = scan_files
        candidate_test_files = scan_files
    else:
        train_files = [f for i, f in enumerate(scan_files) if i % args.offset == 0]
        candidate_test_files = [f for i, f in enumerate(scan_files) if i % args.offset != 0]

    if candidate_test_files and args.test_fraction < 1.0:
        n_candidates = len(candidate_test_files)
        n_keep = max(1, int(np.ceil(n_candidates * args.test_fraction)))
        keep_idxs = np.linspace(0, n_candidates - 1, n_keep, dtype=int)
        test_files = [candidate_test_files[i] for i in keep_idxs]
    else:
        test_files = candidate_test_files

    print(f"Scans: {n_total} total, train: {len(train_files)}, test: {len(test_files)}")

    poses = load_poses_csv(args.pose)
    body_to_lidar = load_body_to_lidar(args.calib)
    init_rel_pos = np.array(args.init_rel_pos, dtype=np.float64) if args.init_rel_pos else None

    # Train BKI
    bki = osm_bki_cpp.PyContinuousBKI(
        osm_path=args.osm,
        config_path=args.config,
        resolution=args.resolution,
        l_scale=args.l_scale,
        sigma_0=args.sigma_0,
        prior_delta=args.prior_delta,
        use_semantic_kernel=True,
        use_spatial_kernel=True,
        alpha0=args.alpha0,
        osm_prior_strength=args.osm_prior_strength,
        osm_fallback_in_infer=not args.disable_osm_fallback,
    )

    for scan_path in train_files:
        stem = scan_path.stem
        frame = get_frame_number(stem)
        label_path = find_label_file(label_dir, stem)
        if not label_path:
            continue
        points_xyz, _ = load_scan(str(scan_path))
        labels = load_labels(label_path)
        if len(labels) != len(points_xyz):
            min_len = min(len(labels), len(points_xyz))
            points_xyz = points_xyz[:min_len]
            labels = labels[:min_len]
        if frame is not None and frame in poses:
            points_xyz = transform_points_to_world(
                points_xyz, poses[frame], body_to_lidar, init_rel_pos)
        bki.update(labels, points_xyz)

    print(f"Trained on {len(train_files)} scans. Map size: {bki.get_size()} voxels")

    # Evaluate on test scans: collect pred and gt
    all_pred = []
    all_gt = []
    n_eval = 0
    for scan_path in test_files:
        stem = scan_path.stem
        frame = get_frame_number(stem)
        gt_path = find_label_file(gt_dir, stem)
        if not gt_path:
            continue
        points_xyz, _ = load_scan(str(scan_path))
        gt = load_labels(gt_path)
        if frame is not None and frame in poses:
            points_xyz = transform_points_to_world(
                points_xyz, poses[frame], body_to_lidar, init_rel_pos)
        pred = bki.infer(points_xyz)
        n = min(len(pred), len(gt))
        if n > 0:
            all_pred.append(pred[:n])
            all_gt.append(gt[:n])
            n_eval += 1

    if not all_pred:
        print("ERROR: No test scans had matching GT. Check --gt-dir and scan stems.", file=sys.stderr)
        return 1

    pred_flat = np.concatenate(all_pred)
    gt_flat = np.concatenate(all_gt)

    # Determine class set: union of pred and gt (optionally filter ignore)
    ignore_label = 0
    mask = gt_flat != ignore_label
    pred_m = pred_flat[mask]
    gt_m = gt_flat[mask]
    all_classes = sorted(set(np.unique(pred_m).tolist()) | set(np.unique(gt_m).tolist()))
    if not all_classes:
        print("ERROR: No valid (non-ignored) points for confusion matrix.", file=sys.stderr)
        return 1

    cm = sklearn_confusion_matrix(gt_m, pred_m, labels=all_classes)

    # Load label names
    label_names = ConfigReader(args.config).label_names
    class_names = [label_names.get(c, str(c)) for c in all_classes]

    # Plot
    output_path = args.output or "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, output_path=output_path,
                         normalize=args.normalize)

    # Print summary metrics
    accuracy = np.sum(np.diag(cm)) / (cm.sum() + 1e-9)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    print(f"Evaluated {n_eval} test scans, {len(pred_m):,} points")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
