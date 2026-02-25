#!/usr/bin/env bash
# Run OSM-BKI continuous map training on example data.
# Execute from repo root: ./python/run_osmbki.sh
# Requires: osm_bki_cpp to be built

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Output directory — override by setting RUN_RESULTS_DIR in your environment
# ---------------------------------------------------------------------------
export RUN_RESULTS_DIR="${RUN_RESULTS_DIR:-./run_results}"

# ---------------------------------------------------------------------------
# MCD example data paths (relative to repo root)
# ---------------------------------------------------------------------------
MCD="example_data/mcd"
DATA="$MCD/kth_day_06"
SCAN_DIR="$DATA/lidar_bin/data/"
LABEL_DIR="$DATA/labels_predicted/"
GT_DIR="$DATA/gt_labels/"
OSM="$MCD/kth.osm"
POSE="$DATA/pose_inW.csv"
CALIB="$MCD/hhs_calib.yaml"
CONFIG="configs/mcd_config.yaml"

# --- To switch to KITTI-360, comment the block above and uncomment below: ---
# MCD="example_data/kitti360"
# DATA="$MCD/2013_05_28_drive_0009_sync"
# SCAN_DIR="$DATA/velodyne_points/data/"
# LABEL_DIR="$DATA/inferred_labels/cenet_semkitti/"
# GT_DIR="$DATA/gt_labels/"
# OSM="$MCD/map_0009.osm"
# POSE="$DATA/velodyne_poses.txt"
# CALIB=""
# CONFIG="configs/kitti360_config.yaml"

echo "Results will be written to: $RUN_RESULTS_DIR"

# Pass --calib only if the file actually exists
CALIB_FLAG=""
if [ -n "$CALIB" ] && [ -f "$CALIB" ]; then
    CALIB_FLAG="--calib $CALIB"
fi

# For KITTI-360, add: --pose-format mat4 --dataset-pred-type semkitti --dataset-gt-type kitti360
python python/scripts/continuous_map_train_test.py \
    --config "$CONFIG" \
    --osm "$OSM" \
    $CALIB_FLAG \
    --scan-dir "$SCAN_DIR" \
    --label-dir "$LABEL_DIR" \
    --gt-dir "$GT_DIR" \
    --pose "$POSE" \
    --dataset-pred-type mcd \
    --dataset-gt-type mcd \
    --output-dir "$RUN_RESULTS_DIR" \
    --offset 1 \
    --max-scans 1000 \
    --test-fraction 1 \
    --osm-prior-strength 0.01 \
    --prior-delta 1 \
    --resolution 0.5 \
    --l-scale 1.0
