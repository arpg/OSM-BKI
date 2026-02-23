#!/usr/bin/env python3
"""
Train a continuous BKI map on the first half of scans and evaluate on the second half.

Uses the same LiDAR->Body->World transform as build_voxel_map.py when a pose file
is provided, so all scans are in a common world frame. No pre-voxelization;
points are transformed and passed directly to the BKI.
"""

import os
import sys
import argparse
import numpy as np
import osm_bki_cpp
from pathlib import Path

from script_utils import (
    load_body_to_lidar, load_poses_csv, transform_points_to_world,
    load_scan, load_labels, find_label_file, get_frame_number, ConfigReader
)


def compute_metrics(pred, gt, ignore_label=0):
    """Accuracy and mIoU; optionally ignore a label (e.g. 0) in GT."""
    pred = np.asarray(pred, dtype=np.uint32)
    gt = np.asarray(gt, dtype=np.uint32)
    if pred.shape != gt.shape:
        return {"accuracy": 0.0, "miou": 0.0, "class_ious": {}}
    mask = gt != ignore_label
    if not np.any(mask):
        return {"accuracy": 0.0, "miou": 0.0, "class_ious": {}}
    pred_m = pred[mask]
    gt_m = gt[mask]
    accuracy = np.mean(pred_m == gt_m)
    classes = np.unique(np.concatenate([pred_m, gt_m]))
    ious = {}
    for c in classes:
        inter = np.sum((pred_m == c) & (gt_m == c))
        union = np.sum((pred_m == c) | (gt_m == c))
        if union > 0:
            ious[int(c)] = inter / union
    miou = np.mean(list(ious.values())) if ious else 0.0
    return {"accuracy": accuracy, "miou": miou, "class_ious": ious}


def main():
    parser = argparse.ArgumentParser(
        description="Train continuous BKI with offset-based scan subsampling and evaluate on held-out in-between scans."
    )
    parser.add_argument("--scan-dir", required=True, help="Directory of .bin scans")
    parser.add_argument("--label-dir", required=True, help="Directory of prediction labels (.label or .bin)")
    parser.add_argument("--osm", required=True, help="Path to OSM geometries (.bin or .osm XML)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--pose", default=None, help="CSV poses (num,t,x,y,z,qx,qy,qz,qw) for world-frame transform (optional)")
    parser.add_argument("--calib", default="example_data/hhs_calib.yaml", help="Path to hhs_calib.yaml (body→LiDAR calibration, body/os_sensor/T)")
    parser.add_argument("--gt-dir", default=None, help="Optional: directory of ground-truth labels for test half metrics")
    parser.add_argument("--output-dir", default=None, help="Optional: save refined labels for test scans here")
    parser.add_argument("--map-state", default=None, help="Optional: path to save trained map state")
    parser.add_argument("--offset", type=int, default=1, help="Train on every Nth scan; use in-between scans for testing (N>=1)")
    parser.add_argument("--test-fraction", type=float, default=1.0, help="Fraction of selected test scans to evaluate (0 < f <= 1)")
    parser.add_argument("--max-scans", type=int, default=None, help="Optional cap on number of scans to use from the sorted list")
    parser.add_argument("--resolution", type=float, default=1.0, help="BKI resolution")
    parser.add_argument("--l-scale", type=float, default=3.0, help="BKI l_scale")
    parser.add_argument("--sigma-0", type=float, default=1.0, help="BKI sigma_0")
    parser.add_argument("--prior-delta", type=float, default=0.5, help="BKI prior_delta")
    parser.add_argument("--height-sigma", type=float, default=5.0, help="BKI height_sigma (controls vertical spread of OSM prior)")
    parser.add_argument("--alpha0", type=float, default=1.0, help="BKI alpha0")
    parser.add_argument("--seed-osm-prior", type=bool, default=False, help="Enable OSM prior seeding")
    parser.add_argument("--osm-prior-strength", type=float, default=0.0, help="OSM prior strength")
    parser.add_argument("--disable-osm-fallback", type=bool, default=False, help="Disable OSM fallback during inference")
    parser.add_argument("--init_rel_pos", type=float, nargs=3, metavar=('X', 'Y', 'Z'), default=None,
                        help="World-frame initial position to subtract from poses "
                             "(overrides init_rel_pos_day_06 in config YAML)")
    parser.add_argument("--osm_origin_lat", type=float, default=None,
                        help="Override OSM projection origin latitude (read from config YAML if not given)")
    parser.add_argument("--osm_origin_lon", type=float, default=None,
                        help="Override OSM projection origin longitude (read from config YAML if not given)")
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir)
    label_dir = Path(args.label_dir)
    scan_files = sorted(scan_dir.glob("*.bin"))
    if not scan_files:
        print(f"No .bin scans in {scan_dir}", file=sys.stderr)
        return 1

    if args.max_scans is not None:
        if args.max_scans <= 0:
            print("ERROR: --max-scans must be > 0", file=sys.stderr)
            return 1
        if args.max_scans < len(scan_files):
            scan_files = scan_files[:args.max_scans]

    if args.offset < 1:
        print("ERROR: --offset must be >= 1", file=sys.stderr)
        return 1
    if args.test_fraction <= 0.0 or args.test_fraction > 1.0:
        print("ERROR: --test-fraction must be in (0, 1]", file=sys.stderr)
        return 1

    n_total = len(scan_files)
    if args.offset == 1:
        # Backward-compatible behavior: all scans contribute to map and are evaluated.
        train_files = scan_files
        candidate_test_files = scan_files
        print(f"Scans: {n_total} total -> offset=1, using all scans for both training and testing")
    else:
        train_files = [f for i, f in enumerate(scan_files) if i % args.offset == 0]
        candidate_test_files = [f for i, f in enumerate(scan_files) if i % args.offset != 0]
        print(
            f"Scans: {n_total} total -> offset={args.offset}, "
            f"training on {len(train_files)} scans (every {args.offset}th), "
            f"testing candidate set has {len(candidate_test_files)} in-between scans"
        )

    # Optionally downsample test scans using deterministic uniform index spacing.
    if candidate_test_files and args.test_fraction < 1.0:
        n_candidates = len(candidate_test_files)
        n_keep = max(1, int(np.ceil(n_candidates * args.test_fraction)))
        keep_idxs = np.linspace(0, n_candidates - 1, n_keep, dtype=int)
        test_files = [candidate_test_files[i] for i in keep_idxs]
    else:
        test_files = candidate_test_files

    print(
        f"Final test split: {len(test_files)} / {len(candidate_test_files)} "
        f"candidate scans (test_fraction={args.test_fraction:.3f})"
    )
    if n_total > 0:
        pct_total = 100.0 * len(test_files) / n_total
    else:
        pct_total = 0.0
    if len(candidate_test_files) > 0:
        pct_candidates = 100.0 * len(test_files) / len(candidate_test_files)
    else:
        pct_candidates = 0.0
    print(
        f"Testing will run on {len(test_files)} scans "
        f"({pct_total:.1f}% of total, {pct_candidates:.1f}% of candidate test set)."
    )

    # Resolve init_rel_pos: CLI overrides config YAML's init_rel_pos_day_06
    init_rel_pos = None
    if args.init_rel_pos is not None:
        init_rel_pos = np.array(args.init_rel_pos, dtype=np.float64)
    else:
        init_rel_pos = ConfigReader(args.config).init_rel_pos
        if init_rel_pos is not None:
            init_rel_pos = init_rel_pos.astype(np.float64)

    if init_rel_pos is not None:
        print(f"init_rel_pos = [{init_rel_pos[0]:.4f}, {init_rel_pos[1]:.4f}, {init_rel_pos[2]:.4f}]")
    else:
        print("No init_rel_pos; poses used as-is (no world-frame re-zeroing).")

    poses = None
    body_to_lidar = None
    if args.pose:
        if not os.path.exists(args.pose):
            print(f"Pose file not found: {args.pose}", file=sys.stderr)
            return 1
        if not os.path.exists(args.calib):
            print(f"Calibration file not found: {args.calib}", file=sys.stderr)
            return 1
        poses = load_poses_csv(args.pose)
        body_to_lidar = load_body_to_lidar(args.calib)
        print(f"Loaded {len(poses)} poses; transforming scans to world frame using {args.calib}")
    else:
        print("No --pose provided; using scan coordinates as-is (sensor frame).")

    if not os.path.exists(args.osm):
        print("OSM file must exist", file=sys.stderr)
        return 1
    if not (args.osm.endswith(".bin") or args.osm.endswith(".osm")):
        print("OSM file must be .bin (binary) or .osm (XML format)", file=sys.stderr)
        return 1
    if not os.path.exists(args.config):
        print("Config file not found", file=sys.stderr)
        return 1

    def train_bki(use_semantic_kernel):
        bki = osm_bki_cpp.PyContinuousBKI(
            osm_path=args.osm,
            config_path=args.config,
            resolution=args.resolution,
            l_scale=args.l_scale,
            sigma_0=args.sigma_0,
            prior_delta=args.prior_delta,
            height_sigma=args.height_sigma,
            use_semantic_kernel=use_semantic_kernel,
            use_spatial_kernel=True,
            alpha0=args.alpha0,
            seed_osm_prior=args.seed_osm_prior,
            osm_prior_strength=args.osm_prior_strength,
            osm_fallback_in_infer=not args.disable_osm_fallback
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
            if poses is not None and frame is not None and frame in poses:
                points_xyz = transform_points_to_world(points_xyz, poses[frame], body_to_lidar, init_rel_pos)
            bki.update(labels, points_xyz)
        return bki

    # Build maps: with semantic kernel, without semantic kernel, and without spatial kernel
    print("\n--- Training (first half) ---")
    print("  Training with semantic kernel...")
    bki_sem = train_bki(use_semantic_kernel=True)
    print(f"  Map size (with sem):  {bki_sem.get_size()} voxels")
    
    print("  Training without semantic kernel...")
    bki_nosem = train_bki(use_semantic_kernel=False)
    print(f"  Map size (without sem): {bki_nosem.get_size()} voxels")

    print("  Training without BOTH kernels (no spatial, no semantic)...")
    # To disable spatial kernel, we need to modify train_bki or create a new instance
    # Let's modify train_bki to accept use_spatial_kernel
    def train_bki_custom(use_semantic_kernel, use_spatial_kernel):
        bki = osm_bki_cpp.PyContinuousBKI(
            osm_path=args.osm,
            config_path=args.config,
            resolution=args.resolution,
            l_scale=args.l_scale,
            sigma_0=args.sigma_0,
            prior_delta=args.prior_delta,
            height_sigma=args.height_sigma,
            use_semantic_kernel=use_semantic_kernel,
            use_spatial_kernel=use_spatial_kernel,
            alpha0=args.alpha0,
            seed_osm_prior=args.seed_osm_prior,
            osm_prior_strength=args.osm_prior_strength,
            osm_fallback_in_infer=not args.disable_osm_fallback
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
            if poses is not None and frame is not None and frame in poses:
                points_xyz = transform_points_to_world(points_xyz, poses[frame], body_to_lidar, init_rel_pos)
            bki.update(labels, points_xyz)
        return bki

    bki_nokernels = train_bki_custom(use_semantic_kernel=False, use_spatial_kernel=False)
    print(f"  Map size (no kernels): {bki_nokernels.get_size()} voxels")

    if args.map_state:
        base, ext = os.path.splitext(args.map_state)
        os.makedirs(os.path.dirname(args.map_state) or ".", exist_ok=True)
        bki_sem.save(base + "_sem" + (ext or ""))
        bki_nosem.save(base + "_nosem" + (ext or ""))
        bki_nokernels.save(base + "_nokernels" + (ext or ""))
        print(f"  Saved maps to {base}_sem / {base}_nosem / {base}_nokernels")

    # Evaluate on held-out scans selected by offset
    print("\n--- Testing (offset holdout scans) ---")
    all_metrics_sem = []
    all_metrics_nosem = []
    all_metrics_nokernels = []
    all_metrics_baseline = []
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for scan_path in test_files:
        stem = scan_path.stem
        frame = get_frame_number(stem)
        points_xyz, _ = load_scan(str(scan_path))
        if poses is not None and frame is not None and frame in poses:
            points_xyz = transform_points_to_world(points_xyz, poses[frame], body_to_lidar, init_rel_pos)

        pred_sem = bki_sem.infer(points_xyz)
        pred_nosem = bki_nosem.infer(points_xyz)
        pred_nokernels = bki_nokernels.infer(points_xyz)

        if args.output_dir:
            pred_sem.astype(np.uint32).tofile(str(Path(args.output_dir) / f"{stem}_refined_sem.label"))
            pred_nosem.astype(np.uint32).tofile(str(Path(args.output_dir) / f"{stem}_refined_nosem.label"))
            pred_nokernels.astype(np.uint32).tofile(str(Path(args.output_dir) / f"{stem}_refined_nokernels.label"))

        if args.gt_dir:
            gt_path = find_label_file(args.gt_dir, stem)
            if gt_path:
                gt = load_labels(gt_path)
                
                # Load baseline (input) labels
                input_path = find_label_file(label_dir, stem)
                input_labels = load_labels(input_path) if input_path else None

                n = min(len(gt), len(pred_sem), len(pred_nosem), len(pred_nokernels))
                if input_labels is not None:
                    n = min(n, len(input_labels))
                
                if n > 0:
                    all_metrics_sem.append(compute_metrics(pred_sem[:n], gt[:n]))
                    all_metrics_nosem.append(compute_metrics(pred_nosem[:n], gt[:n]))
                    all_metrics_nokernels.append(compute_metrics(pred_nokernels[:n], gt[:n]))
                    if input_labels is not None:
                        all_metrics_baseline.append(compute_metrics(input_labels[:n], gt[:n]))

    if all_metrics_sem:
        acc_sem = np.mean([m["accuracy"] for m in all_metrics_sem])
        miou_sem = np.mean([m["miou"] for m in all_metrics_sem])
        acc_nosem = np.mean([m["accuracy"] for m in all_metrics_nosem])
        miou_nosem = np.mean([m["miou"] for m in all_metrics_nosem])
        acc_nokernels = np.mean([m["accuracy"] for m in all_metrics_nokernels])
        miou_nokernels = np.mean([m["miou"] for m in all_metrics_nokernels])
        
        print(f"  Test scans with GT: {len(all_metrics_sem)}")
        if len(test_files) > 0:
            pct_gt = 100.0 * len(all_metrics_sem) / len(test_files)
        else:
            pct_gt = 0.0
        print(
            f"  Evaluated {len(all_metrics_sem)} / {len(test_files)} selected test scans "
            f"({pct_gt:.1f}% had matching GT)."
        )
        if all_metrics_baseline:
            acc_base = np.mean([m["accuracy"] for m in all_metrics_baseline])
            miou_base = np.mean([m["miou"] for m in all_metrics_baseline])
            print("  Baseline (Input):        Accuracy {:.4f}  mIoU {:.4f}".format(acc_base, miou_base))
        
        print("  With semantic kernel:    Accuracy {:.4f}  mIoU {:.4f}".format(acc_sem, miou_sem))
        print("  Without semantic kernel: Accuracy {:.4f}  mIoU {:.4f}".format(acc_nosem, miou_nosem))
        print("  Without ANY kernels:     Accuracy {:.4f}  mIoU {:.4f}".format(acc_nokernels, miou_nokernels))
    else:
        if not test_files:
            print("  No test scans selected by offset. Increase --offset to create holdout scans.")
        elif args.gt_dir:
            print("  No test scans had matching GT labels.")
        else:
            print("  No --gt-dir provided; skipping metrics.")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
