# OSM-BKI: Composite BKI for LiDAR Semantic Segmentation

Bayesian Kernel Inference (BKI) for semantic label refinement of LiDAR point clouds using OpenStreetMap (OSM) priors.

This project features a **Continuous Mapping Engine** (`continuous_bki`) as its core backend, supporting both single-scan refinement and stateful semantic mapping over time.

---

## Project Structure

```
OSM-BKI/
├── README.md                 # This file
├── configs/                  # Dataset configurations
│   ├── mcd_config.yaml       # MCD (KTH campus) – direct paths or dataset_root_path+sequence
│   └── kitti*.yaml           # KITTI / KITTI360 configs
│
├── example_data/             # Sample data
│   ├── mcd/                  # MCD format (KTH)
│   │   ├── kth_day_06/       # LiDAR, labels, poses
│   │   ├── kth.osm           # OSM map
│   │   └── hhs_calib.yaml    # Calibration
│   └── kitti360/             # KITTI360 format
│
├── python/                   # Python package, scripts, benchmarks
│   ├── README.md             # Python setup & usage
│   ├── run_osmbki.sh         # Run continuous map on example data
│   ├── run_benchmarks.sh     # Run all benchmarks
│   ├── scripts/              # continuous_map_train_test, visualize_osm_xml, bki_tools, etc.
│   └── benchmarks/           # Throughput, calibration, OSM modes, etc.
│
└── cpp/osm_bki/              # C++ library
    ├── README.md             # C++ build & examples
    └── src/, include/        # Core engine
```

---

## Quick Start

### 1. Build the Python extension

```bash
cd python
pip install -r requirements.txt
pip install -e .
```

### 2. Run on example data

From the **repo root**:

```bash
./python/run_osmbki.sh
```

Trains a continuous BKI map on `example_data/mcd/kth_day_06` and evaluates on held-out scans. Output: `output.bki`.

### 3. Run benchmarks

```bash
./python/run_benchmarks.sh
```

---

## Configuration

Paths are specified via **direct keys** in the config (relative to the config file) or via **legacy** `dataset_root_path` + `sequence`:

- **Direct paths**: `lidar_dir`, `label_dir`, `pose_path`, `calibration_path`, `osm_file`
- **Legacy**: `dataset_root_path` + `sequence` – paths derived under `dataset_root_path/sequence/`

See `configs/mcd_config.yaml` and [python/README.md](python/README.md) for details.

---

## Python API

```python
import composite_bki_cpp
import numpy as np

# Initialize BKI engine
bki = composite_bki_cpp.PyContinuousBKI(
    osm_path="example_data/mcd/kth.osm",
    config_path="configs/mcd_config.yaml",
    resolution=0.5,
    l_scale=1.0
)

# Load data (numpy arrays)
points = np.fromfile("scan.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
labels = np.fromfile("labels.bin", dtype=np.uint32)

# Update map and infer refined labels
bki.update(labels, points)
refined = bki.infer(points)

# Save map state
bki.save("map.bki")
```

---

## Features

- **Unified backend**: Single C++ engine (`ContinuousBKI`) for all operations
- **Voxel-based**: Sparse voxel hashing for O(1) lookups
- **Stateful**: Accumulates semantic evidence across thousands of scans
- **OSM priors**: XML or binary OSM geometries for roads, buildings, vegetation, etc.
- **Multi-threaded**: OpenMP support

---

## Requirements

- Python 3.7+
- NumPy, pybind11, PyYAML, Open3D (see `python/requirements.txt`)
- C++17 compiler with OpenMP

---

## Documentation

- [python/README.md](python/README.md) – Python setup, scripts, benchmarks
- [cpp/osm_bki/README.md](cpp/osm_bki/README.md) – C++ build and examples

---

## Citation

If you use this work, please cite:

```bibtex
@software{osm_bki,
  title={OSM-BKI: Bayesian Kernel Inference for LiDAR Semantic Segmentation with OpenStreetMap Priors},
  author={OSM-BKI Team},
  year={2026},
  url={https://github.com/yourrepo/OSM-BKI}
}
```

---

## License

MIT License
