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
# Example data paths (relative to repo root)
# ---------------------------------------------------------------------------
MCD="example_data/mcd"
DATA="$MCD/kth_day_06"
SCAN_DIR="$DATA/lidar_bin/data"
LABEL_DIR="$DATA/labels_predicted"
GT_DIR="$DATA/gt_labels"
OSM="$MCD/kth.osm"
POSE="$DATA/pose_inW.csv"
CONFIG="configs/mcd_config.yaml"

# Body→LiDAR calibration (optional).
# Set CALIB to the path if your dataset has one; leave empty or unset to use
# an identity transform (i.e. poses are already expressed in the LiDAR frame).
CALIB="$MCD/hhs_calib.yaml"

echo "Results will be written to: $RUN_RESULTS_DIR"

# Pass --calib only if the file actually exists
CALIB_FLAG=""
if [ -n "$CALIB" ] && [ -f "$CALIB" ]; then
    CALIB_FLAG="--calib $CALIB"
fi

python python/scripts/continuous_map_train_test.py \
    --config "$CONFIG" \
    --osm "$OSM" \
    $CALIB_FLAG \
    --init_rel_pos 64.393 66.483 38.514 \
    --osm_origin_lat 59.348268650 \
    --osm_origin_lon 18.073204280 \
    --scan-dir "$SCAN_DIR" \
    --label-dir "$LABEL_DIR" \
    --gt-dir "$GT_DIR" \
    --pose "$POSE" \
    --output-dir "$RUN_RESULTS_DIR" \
    --offset 1 \
    --max-scans 1000 \
    --test-fraction 1 \
    --osm-prior-strength 0.01 \
    --prior-delta 1 \
    --resolution 0.5 \
    --l-scale 1.0
