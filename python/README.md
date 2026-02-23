# OSM-BKI Python Package

Python bindings and scripts for OSM-S-BKI (Composite BKI): semantic Bayesian Kernel Inference for LiDAR with OpenStreetMap priors.

## Directory Structure

```
python/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── setup.py               # Builds composite_bki_cpp extension
├── run_osmbki.sh          # Run continuous map training on example data
├── run_benchmarks.sh      # Run all benchmarks
│
├── osm_bki/               # Package source
│   ├── __init__.py
│   └── pybind/
│       └── osm_bki_bindings.cpp   # C++ Python bindings
│
├── scripts/               # Standalone tools
│   ├── continuous_map_train_test.py   # Train BKI map, evaluate on test scans
│   ├── confusion_matrix_eval.py      # Confusion matrix vs ground truth
│   ├── visualize_osm_xml.py          # Visualize LiDAR map + OSM overlay
│   ├── bki_tools.py                  # Convert/visualize .bki map files
│   ├── label_utils.py                # Label format conversion (SemanticKITTI ↔ MCD)
│   ├── copy_kth_data.py              # Data copying utilities
│   └── script_utils.py              # Shared config, OSM loading, transforms
│
└── benchmarks/            # Performance and quality benchmarks
    ├── benchmark_utils.py            # Shared helpers (load_scan, poses, metrics)
    ├── throughput_benchmark.py       # Thread/point/resolution scaling
    ├── calibration_benchmark.py     # ECE, MCE, reliability diagram
    ├── forgetting_benchmark.py      # Map quality over time
    ├── multi_scan_benchmark.py      # Convergence vs scan count
    ├── osm_modes_benchmark.py       # OSM prior modes (seed, strength, fallback)
    ├── parameter_sensitivity_benchmark.py
    ├── pred_vs_gt_benchmark.py      # Predicted vs GT input comparison
    ├── temporal_consistency_benchmark.py
    ├── per_class_iou_benchmark.py   # Per-class IoU breakdown
    ├── kernel_ablation_benchmark.py # Spatial vs semantic kernel
    ├── noise_benchmark.py           # Robustness to label noise
    ├── osm_randomization_benchmark.py # OSM map perturbation
    └── plot_results.ipynb           # Visualize benchmark results
```

Configs and example data live at the **repo root**:
- `configs/mcd_config.yaml` – MCD dataset config
- `example_data/kth_day_06/` – Example LiDAR, labels, poses
- `example_data/kth.osm` – OSM map

---

## Setup

**Recommended:** Use the `composite-bki` conda environment (or equivalent with `composite_bki_cpp`, scipy, pandas).

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the extension

From the **python** directory:

```bash
pip install -e .
```

This builds the `composite_bki_cpp` C++ extension and installs the `osm_bki` package in editable mode.

### 3. Verify

```bash
python -c "import composite_bki_cpp; print(composite_bki_cpp.PyContinuousBKI)"
```

---

## Usage

### Run OSM-BKI on example data

From the **repo root**:

```bash
./python/run_osmbki.sh
```

Trains a continuous BKI map on `example_data/kth_day_06` and evaluates on held-out scans. Output: `output.bki`.

### Run all benchmarks

From the **repo root** or **python** directory:

```bash
./python/run_benchmarks.sh
```

Runs all 12 benchmarks with example data. Results go to `benchmarks/*_results/` and `benchmarks/*.csv`.

### Individual scripts

Run from the **repo root** so paths resolve correctly:

```bash
# Continuous map training (same as run_osmbki.sh, but with custom args)
python python/scripts/continuous_map_train_test.py \
  --config configs/mcd_config.yaml \
  --osm example_data/kth.osm \
  --calib example_data/hhs_calib.yaml \
  --scan-dir example_data/kth_day_06/lidar_bin/data \
  --label-dir example_data/kth_day_06/labels_predicted \
  --gt-dir example_data/kth_day_06/gt_labels \
  --pose example_data/kth_day_06/pose_inW.csv \
  --init_rel_pos 64.393 66.483 38.514 \
  --osm_origin_lat 59.348268650 --osm_origin_lon 18.073204280

# Visualize map + OSM overlay (Open3D)
python python/scripts/visualize_osm_xml.py --config configs/mcd_config.yaml

# BKI tools: convert .bki to point cloud, or visualize
python python/scripts/bki_tools.py convert --bki output.bki --config configs/mcd_config.yaml
python python/scripts/bki_tools.py visualize --bki output.bki --config configs/mcd_config.yaml
```

### Individual benchmarks

From the **python** directory:

```bash
cd python
python benchmarks/throughput_benchmark.py --scan-dir ../../example_data/kth_day_06/lidar_bin/data --label-dir ../../example_data/kth_day_06/labels_predicted/labels_predicted --osm ../../example_data/kth.osm --config ../../configs/mcd_config.yaml
```

Or use `run_benchmarks.sh` to run all with correct paths.

---

## Troubleshooting

- **ModuleNotFoundError: composite_bki_cpp**  
  Run `pip install -e .` from the `python` directory with your environment activated.

- **Compiler errors**  
  Use a C++17-capable compiler. The extension uses OpenMP if available.

- **OpenMP on macOS**  
  With conda’s clang, you may need `conda install libomp` or the conda compiler stack.

- **Missing configs or example_data**  
  Paths are relative to the repo root. Run scripts from the repo root, or ensure `configs/` and `example_data/` exist there.
