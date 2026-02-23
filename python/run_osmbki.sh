#!/usr/bin/env bash
# Run OSM-BKI continuous map training on example data.
# Execute from repo root: ./python/run_osmbki.sh
# Requires: conda activate osm-bki (or equivalent env with osm_bki_cpp)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Example data paths (relative to repo root)
# Layout: example_data/mcd/{kth_day_06/, kth.osm, hhs_calib.yaml}
MCD="example_data/mcd"
DATA="$MCD/kth_day_06"
SCAN_DIR="$DATA/lidar_bin/data"
LABEL_DIR="$DATA/labels_predicted"
GT_DIR="$DATA/gt_labels"
OSM="$MCD/kth.osm"
CALIB="$MCD/hhs_calib.yaml"
POSE="$DATA/pose_inW.csv"
CONFIG="configs/mcd_config.yaml"

python python/scripts/continuous_map_train_test.py \
    --config "$CONFIG" \
    --osm "$OSM" \
    --calib "$CALIB" \
    --init_rel_pos 64.393 66.483 38.514 \
    --osm_origin_lat 59.348268650 \
    --osm_origin_lon 18.073204280 \
    --scan-dir "$SCAN_DIR" \
    --label-dir "$LABEL_DIR" \
    --gt-dir "$GT_DIR" \
    --pose "$POSE" \
    --offset 1 \
    --max-scans 1000 \
    --test-fraction 1 \
    --map-state output.bki \
    --prior-delta 0.1 \
    --osm-prior-strength 0.01 \
    --seed-osm-prior true \
    --resolution 0.5 \
    --l-scale 1.0
