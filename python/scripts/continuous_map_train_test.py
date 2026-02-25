#!/usr/bin/env python3
"""
Train a continuous BKI map on training scans and evaluate on held-out scans.

Five methods are compared:
  sem+osm     : semantic kernel + OSM prior seeding + OSM fallback
  sem         : semantic kernel only (no OSM prior)
  sbki+osm    : S-BKI (spatial kernel only) + OSM prior seeding + OSM fallback
  sbki        : S-BKI (spatial kernel only, no OSM prior)
  nokernels   : no kernels (label pass-through via BKI mean)
Plus a 'baseline' result from the raw input prediction labels.

All outputs are written to --output-dir (default: $RUN_RESULTS_DIR or ./run_results/):
  {output_dir}/{method}.bki             - saved map state per method
  {output_dir}/labels/{method}/         - per-scan predicted label files
  {output_dir}/accuracy_per_class.csv   - per-class recall for every method
  {output_dir}/iou_per_class.csv        - per-class IoU for every method
  {output_dir}/confusion_matrix_{method}.png
"""

import csv
import os
import sys
import argparse
import numpy as np
import osm_bki_cpp
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from script_utils import (
    load_body_to_lidar,
    load_poses_csv, transform_points_to_world,
    load_poses_mat4, transform_points_to_world_mat4,
    load_scan, load_labels, find_label_file, get_frame_number, ConfigReader,
)
from label_utils import (
    detect_dataset, build_to_common_lut, apply_common_lut,
    IGNORE_LABELS, COMMON_LABELS, N_COMMON,
)

# ---------------------------------------------------------------------------
# Method definitions
# ---------------------------------------------------------------------------

METHODS = [
    {
        "name": "sem+osm",
        "use_semantic_kernel": True,
        "use_spatial_kernel": True,
        "seed_osm_prior": True,
        "osm_fallback": True,
    },
    {
        "name": "sem",
        "use_semantic_kernel": True,
        "use_spatial_kernel": True,
        "seed_osm_prior": False,
        "osm_fallback": False,
    },
    {
        "name": "sbki+osm",
        "use_semantic_kernel": False,
        "use_spatial_kernel": True,
        "seed_osm_prior": True,
        "osm_fallback": True,
    },
    {
        "name": "sbki",
        "use_semantic_kernel": False,
        "use_spatial_kernel": True,
        "seed_osm_prior": False,
        "osm_fallback": False,
    },
    {
        "name": "nokernels",
        "use_semantic_kernel": False,
        "use_spatial_kernel": False,
        "seed_osm_prior": False,
        "osm_fallback": False,
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_cm(pred, gt, n_classes, ignore_label=0):
    """Return (n_classes × n_classes) confusion matrix, ignoring ignore_label in GT."""
    pred = np.asarray(pred, dtype=np.int64)
    gt = np.asarray(gt, dtype=np.int64)
    mask = gt != ignore_label
    classes = list(range(n_classes))
    if not np.any(mask):
        return np.zeros((n_classes, n_classes), dtype=np.int64)
    return sklearn_confusion_matrix(gt[mask], pred[mask], labels=classes)


def metrics_from_cm(cm):
    """
    Derive per-class accuracy (recall), per-class IoU, overall accuracy and mIoU
    from an aggregated confusion matrix.

    Returns dict with keys:
      per_class_accuracy : ndarray (n_classes,)
      per_class_iou      : ndarray (n_classes,)
      accuracy           : float
      miou               : float  (mean over valid classes)
    """
    n = cm.shape[0]
    per_class_acc = np.zeros(n)
    per_class_iou = np.zeros(n)

    for c in range(n):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        row_sum = tp + fn
        union = tp + fp + fn
        per_class_acc[c] = tp / row_sum if row_sum > 0 else float("nan")
        per_class_iou[c] = tp / union if union > 0 else float("nan")

    total_correct = np.diag(cm).sum()
    total_pts = cm.sum()
    accuracy = total_correct / total_pts if total_pts > 0 else 0.0

    valid_ious = per_class_iou[~np.isnan(per_class_iou)]
    miou = float(np.mean(valid_ious)) if len(valid_ious) > 0 else 0.0

    return {
        "per_class_accuracy": per_class_acc,
        "per_class_iou": per_class_iou,
        "accuracy": accuracy,
        "miou": miou,
    }


def plot_and_save_confusion_matrix(cm, class_names, path, title=None, normalize=True):
    """Save a confusion matrix heatmap to *path*."""
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)
        fmt = ".2f"
        cbar_label = "Recall (fraction)"
    else:
        cm_plot = cm
        fmt = "d"
        cbar_label = "Count"

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_plot, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar_kws={"label": cbar_label},
        linewidths=0.5, linecolor="gray",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title or "Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_per_class_csv(path, class_names, method_names, data_rows):
    """
    Save a CSV where rows = classes, columns = [class_name, method1, method2, ...].

    data_rows : list of length n_classes, each entry is a list of per-method values.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class"] + method_names)
        for name, values in zip(class_names, data_rows):
            row = [name] + [f"{v:.4f}" if not np.isnan(v) else "nan" for v in values]
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_inputs(args) -> int:
    """
    Check that all file/directory paths supplied via CLI are valid before any
    expensive work begins.  Collects *all* problems and prints them together so
    the user can fix everything in one edit.

    Returns the number of errors found (0 = OK).
    """
    errors = []

    def require_file(path, label):
        if not path:
            return
        if not os.path.isfile(path):
            errors.append(f"  {label}: file not found: {path}")

    def require_dir(path, label):
        if not path:
            return
        if not os.path.isdir(path):
            errors.append(f"  {label}: directory not found: {path}")

    def require_dir_nonempty(path, label, glob="*"):
        if not path:
            return
        p = Path(path)
        if not p.is_dir():
            errors.append(f"  {label}: directory not found: {path}")
            return
        if not any(p.glob(glob)):
            errors.append(f"  {label}: directory is empty (no '{glob}' files): {path}")

    # Required files
    require_file(args.osm,    "--osm")
    require_file(args.config, "--config")

    # Required directories (must exist and have content)
    require_dir_nonempty(args.scan_dir,  "--scan-dir",  "*.bin")
    require_dir(args.label_dir, "--label-dir")
    require_dir(args.gt_dir,    "--gt-dir")

    # Optional pose file
    if args.pose:
        require_file(args.pose, "--pose")

    # Optional calibration file (only relevant for quat pose format)
    if args.calib and getattr(args, "pose_format", "quat") != "mat4":
        require_file(args.calib, "--calib")

    if errors:
        print("ERROR: invalid inputs — aborting before any processing:\n" +
              "\n".join(errors), file=sys.stderr)
    return len(errors)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    default_output_dir = os.environ.get("RUN_RESULTS_DIR", "./run_results")

    parser = argparse.ArgumentParser(
        description="Train continuous BKI and evaluate five methods against GT labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--scan-dir",   required=True, help="Directory of .bin scans")
    parser.add_argument("--label-dir",  required=True, help="Directory of prediction labels (.label or .bin)")
    parser.add_argument("--osm",        required=True, help="Path to OSM geometries (.bin or .osm XML)")
    parser.add_argument("--config",     required=True, help="Path to YAML config")
    parser.add_argument("--gt-dir",     required=True, help="Directory of ground-truth labels")
    parser.add_argument("--output-dir", default=default_output_dir,
                        help=f"Directory to write all results (default: $RUN_RESULTS_DIR or {default_output_dir})")
    parser.add_argument("--pose",  default=None, help="Pose file path")
    parser.add_argument("--pose-format", choices=["quat", "mat4"], default="quat",
                        help="Pose file format: 'quat' (default, CSV num,t,x,y,z,qx,qy,qz,qw) "
                             "or 'mat4' (KITTI-360 velodyne_poses.txt, frame + 3x4/4x4 matrix)")
    parser.add_argument("--calib", default=None,
                        help="Path to body→LiDAR calibration YAML (hhs_calib.yaml). "
                             "If omitted, an identity transform is used (poses are already in LiDAR frame).")
    parser.add_argument("--dataset-pred-type", choices=["mcd", "semkitti", "kitti360"], default=None,
                        help="Dataset type for prediction labels (overrides autodetect)")
    parser.add_argument("--dataset-gt-type",   choices=["mcd", "semkitti", "kitti360"], default=None,
                        help="Dataset type for GT labels (overrides autodetect)")
    parser.add_argument("--offset",        type=int,   default=1,   help="Train on every Nth scan (N>=1)")
    parser.add_argument("--test-fraction", type=float, default=1.0, help="Fraction of test scans to evaluate (0,1]")
    parser.add_argument("--max-scans",     type=int,   default=None, help="Cap on number of scans")
    parser.add_argument("--resolution",    type=float, default=1.0)
    parser.add_argument("--l-scale",       type=float, default=3.0)
    parser.add_argument("--sigma-0",       type=float, default=1.0)
    parser.add_argument("--prior-delta",   type=float, default=0.5)
    parser.add_argument("--height-sigma",  type=float, default=5.0)
    parser.add_argument("--alpha0",        type=float, default=1.0)
    parser.add_argument("--osm-prior-strength", type=float, default=0.01,
                        help="OSM prior strength used for all OSM-prior methods")
    parser.add_argument("--disable-osm-fallback", action="store_true",
                        help="Disable OSM fallback during inference for OSM-prior methods")
    parser.add_argument("--init_rel_pos", type=float, nargs=3, metavar=("X", "Y", "Z"), default=None)
    parser.add_argument("--osm_origin_lat", type=float, default=None)
    parser.add_argument("--osm_origin_lon", type=float, default=None)
    args = parser.parse_args()

    if validate_inputs(args):
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # -------------------------------------------------------------------------
    # Scan list
    # -------------------------------------------------------------------------
    scan_dir  = Path(args.scan_dir)
    label_dir = Path(args.label_dir)
    scan_files = sorted(scan_dir.glob("*.bin"))

    if args.max_scans is not None:
        if args.max_scans <= 0:
            print("ERROR: --max-scans must be > 0", file=sys.stderr)
            return 1
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
        print(f"Scans: {n_total} total -> offset=1, all scans for train+test")
    else:
        train_files = [f for i, f in enumerate(scan_files) if i % args.offset == 0]
        candidate_test_files = [f for i, f in enumerate(scan_files) if i % args.offset != 0]
        print(
            f"Scans: {n_total} total -> offset={args.offset}, "
            f"train={len(train_files)}, test candidates={len(candidate_test_files)}"
        )

    if candidate_test_files and args.test_fraction < 1.0:
        n_keep = max(1, int(np.ceil(len(candidate_test_files) * args.test_fraction)))
        idxs = np.linspace(0, len(candidate_test_files) - 1, n_keep, dtype=int)
        test_files = [candidate_test_files[i] for i in idxs]
    else:
        test_files = candidate_test_files

    print(f"Test scans: {len(test_files)} / {len(candidate_test_files)} candidates")

    # -------------------------------------------------------------------------
    # Autodetect dataset types + build LUTs
    # -------------------------------------------------------------------------
    _first_pred = find_label_file(label_dir, scan_files[0].stem)
    if _first_pred:
        pred_dataset = args.dataset_pred_type or detect_dataset(load_labels(_first_pred))
    else:
        pred_dataset = args.dataset_pred_type or "mcd"
        print("WARNING: could not find pred label for autodetect; using 'mcd'.", file=sys.stderr)
    pred_lut = build_to_common_lut(pred_dataset)
    print(f"Prediction labels: dataset={pred_dataset}")

    _first_gt = find_label_file(args.gt_dir, scan_files[0].stem)
    if _first_gt:
        gt_dataset = args.dataset_gt_type or detect_dataset(load_labels(_first_gt))
    else:
        gt_dataset = args.dataset_gt_type or pred_dataset
        print("WARNING: could not find GT label for autodetect; using pred dataset.", file=sys.stderr)
    gt_lut = build_to_common_lut(gt_dataset)
    print(f"GT labels:         dataset={gt_dataset}")

    # -------------------------------------------------------------------------
    # Poses / calibration
    # -------------------------------------------------------------------------
    init_rel_pos = None
    if args.init_rel_pos is not None:
        init_rel_pos = np.array(args.init_rel_pos, dtype=np.float64)
    else:
        init_rel_pos = ConfigReader(args.config).init_rel_pos
        if init_rel_pos is not None:
            init_rel_pos = init_rel_pos.astype(np.float64)

    if init_rel_pos is not None:
        print(f"init_rel_pos = {init_rel_pos.tolist()}")
    else:
        print("No init_rel_pos; poses used as-is.")

    poses = None
    body_to_lidar = None
    pose_format = args.pose_format
    if args.pose:
        if pose_format == "mat4":
            poses = load_poses_mat4(args.pose)
            print(f"Loaded {len(poses)} mat4 poses from {args.pose}")
            if args.calib:
                print("WARNING: --calib is ignored with --pose-format mat4 "
                      "(pose already encodes LiDAR-to-world).", file=sys.stderr)
        else:
            poses = load_poses_csv(args.pose)
            if args.calib:
                body_to_lidar = load_body_to_lidar(args.calib)
                print(f"Loaded {len(poses)} poses from {args.pose} with calib {args.calib}")
            else:
                body_to_lidar = np.eye(4, dtype=np.float64)
                print(f"Loaded {len(poses)} poses from {args.pose} (no calib — using identity body→LiDAR)")
    else:
        print("No --pose; using scan coordinates as-is.")

    # -------------------------------------------------------------------------
    # Build shared OSM context (loads OSM file + builds prior raster once for
    # all 5 methods, instead of repeating the expensive raster build each time)
    # -------------------------------------------------------------------------
    print("\n--- Building OSM context (shared across all methods) ---")
    osm_ctx = osm_bki_cpp.PyOSMContext(
        osm_path=args.osm,
        config_path=args.config,
        prior_delta=args.prior_delta,
        resolution=args.resolution,
    )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    def train_bki(use_semantic_kernel, use_spatial_kernel, seed_osm_prior, osm_fallback):
        bki = osm_bki_cpp.PyContinuousBKI(
            ctx=osm_ctx,
            resolution=args.resolution,
            l_scale=args.l_scale,
            sigma_0=args.sigma_0,
            prior_delta=args.prior_delta,
            height_sigma=args.height_sigma,
            use_semantic_kernel=use_semantic_kernel,
            use_spatial_kernel=use_spatial_kernel,
            alpha0=args.alpha0,
            seed_osm_prior=seed_osm_prior,
            osm_prior_strength=args.osm_prior_strength if seed_osm_prior else 0.0,
            osm_fallback_in_infer=osm_fallback and not args.disable_osm_fallback,
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
                if pose_format == "mat4":
                    points_xyz = transform_points_to_world_mat4(
                        points_xyz, poses[frame], init_rel_pos)
                else:
                    points_xyz = transform_points_to_world(
                        points_xyz, poses[frame], body_to_lidar, init_rel_pos)
            bki.update(apply_common_lut(labels, pred_lut).astype(np.uint32), points_xyz)
        return bki

    print("\n--- Training ---")
    bki_maps = {}
    for m in METHODS:
        print(f"  Training '{m['name']}'...")
        bki_maps[m["name"]] = train_bki(
            use_semantic_kernel=m["use_semantic_kernel"],
            use_spatial_kernel=m["use_spatial_kernel"],
            seed_osm_prior=m["seed_osm_prior"],
            osm_fallback=m["osm_fallback"],
        )
        size = bki_maps[m["name"]].get_size()
        print(f"    Map size: {size} voxels")
        save_path = out_dir / f"{m['name']}.bki"
        bki_maps[m["name"]].save(str(save_path))
        print(f"    Saved -> {save_path}")

    # -------------------------------------------------------------------------
    # Testing
    # -------------------------------------------------------------------------
    print("\n--- Testing ---")

    # Per-method accumulation of all pred + gt for aggregated CM
    all_preds = {m["name"]: [] for m in METHODS}
    all_preds["baseline"] = []
    all_gts = []

    labels_out_dir = out_dir / "labels"
    for m in METHODS:
        (labels_out_dir / m["name"]).mkdir(parents=True, exist_ok=True)

    n_evaluated = 0
    for scan_path in test_files:
        stem = scan_path.stem
        frame = get_frame_number(stem)
        gt_path = find_label_file(args.gt_dir, stem)
        if not gt_path:
            continue

        points_xyz, _ = load_scan(str(scan_path))
        if poses is not None and frame is not None and frame in poses:
            if pose_format == "mat4":
                points_xyz = transform_points_to_world_mat4(
                    points_xyz, poses[frame], init_rel_pos)
            else:
                points_xyz = transform_points_to_world(
                    points_xyz, poses[frame], body_to_lidar, init_rel_pos)

        gt = apply_common_lut(load_labels(gt_path), gt_lut).astype(np.uint32)

        preds_this = {}
        for m in METHODS:
            preds_this[m["name"]] = np.asarray(bki_maps[m["name"]].infer(points_xyz), dtype=np.uint32)

        # Baseline: raw input labels converted to common
        input_path = find_label_file(label_dir, stem)
        baseline = (
            apply_common_lut(load_labels(input_path), pred_lut).astype(np.uint32)
            if input_path else None
        )

        n = len(gt)
        for m in METHODS:
            n = min(n, len(preds_this[m["name"]]))
        if baseline is not None:
            n = min(n, len(baseline))
        if n == 0:
            continue

        all_gts.append(gt[:n])
        for m in METHODS:
            pred_arr = preds_this[m["name"]][:n]
            all_preds[m["name"]].append(pred_arr)
            pred_arr.tofile(str(labels_out_dir / m["name"] / f"{stem}.label"))

        if baseline is not None:
            all_preds["baseline"].append(baseline[:n])
        n_evaluated += 1

    print(f"  Evaluated {n_evaluated} / {len(test_files)} test scans with GT")

    if not all_gts:
        print("  No test scans had matching GT labels.")
        print("\nDone.")
        return 0

    # -------------------------------------------------------------------------
    # Aggregate metrics and save outputs
    # -------------------------------------------------------------------------
    gt_concat = np.concatenate(all_gts)
    ignore = IGNORE_LABELS[0] if IGNORE_LABELS else 0

    # Class names for CSV / CM axes (exclude unlabeled for display)
    all_class_ids = list(range(N_COMMON))
    class_names_all = [COMMON_LABELS[c] for c in all_class_ids]
    # For CM plot: skip unlabeled (class 0)
    cm_class_ids = [c for c in all_class_ids if c != ignore]
    cm_class_names = [COMMON_LABELS[c] for c in cm_class_ids]

    method_names_ordered = [m["name"] for m in METHODS] + ["baseline"]
    acc_rows  = [[] for _ in all_class_ids]   # acc_rows[class_idx][method_idx]
    iou_rows  = [[] for _ in all_class_ids]

    summary_rows = []

    print("\n--- Results ---")
    for method_name in method_names_ordered:
        preds_list = all_preds[method_name]
        if not preds_list:
            # baseline may be absent if no input labels found
            for c_idx in range(N_COMMON):
                acc_rows[c_idx].append(float("nan"))
                iou_rows[c_idx].append(float("nan"))
            summary_rows.append((method_name, float("nan"), float("nan")))
            continue

        pred_concat = np.concatenate(preds_list)
        cm = compute_cm(pred_concat, gt_concat, N_COMMON, ignore_label=ignore)
        mets = metrics_from_cm(cm)

        for c_idx in range(N_COMMON):
            acc_rows[c_idx].append(mets["per_class_accuracy"][c_idx])
            iou_rows[c_idx].append(mets["per_class_iou"][c_idx])

        summary_rows.append((method_name, mets["accuracy"], mets["miou"]))

        # Confusion matrix (skip unlabeled row/col for readability)
        cm_reduced = cm[np.ix_(cm_class_ids, cm_class_ids)]
        cm_path = out_dir / f"confusion_matrix_{method_name}.png"
        plot_and_save_confusion_matrix(
            cm_reduced, cm_class_names, str(cm_path),
            title=f"Confusion Matrix — {method_name}",
            normalize=True,
        )
        print(f"  {method_name:<14}  Acc={mets['accuracy']:.4f}  mIoU={mets['miou']:.4f}"
              f"  -> {cm_path.name}")

    # Save CSVs
    acc_csv = out_dir / "accuracy_per_class.csv"
    iou_csv = out_dir / "iou_per_class.csv"
    save_per_class_csv(acc_csv, class_names_all, method_names_ordered, acc_rows)
    save_per_class_csv(iou_csv, class_names_all, method_names_ordered, iou_rows)
    print(f"\n  Saved {acc_csv.name} and {iou_csv.name}")

    # Summary table
    print("\n  Summary:")
    print(f"  {'Method':<14}  {'Accuracy':>10}  {'mIoU':>10}")
    print(f"  {'-'*38}")
    for name, acc, miou in summary_rows:
        acc_s  = f"{acc:.4f}"  if not np.isnan(acc)  else "n/a"
        miou_s = f"{miou:.4f}" if not np.isnan(miou) else "n/a"
        print(f"  {name:<14}  {acc_s:>10}  {miou_s:>10}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
