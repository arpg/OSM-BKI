# Composite BKI for LiDAR Semantic Segmentation

Bayesian Kernel Inference (BKI) for semantic label refinement of LiDAR point clouds using OpenStreetMap (OSM) priors.

This project now features a **Continuous Mapping Engine** (`continuous_bki`) as its core backend, supporting both single-scan refinement and stateful semantic mapping over time.

## Project Structure

```
composite_bki/
├── src/                        # C++ accelerated package (MAIN)
│   ├── composite_bki_cpp/      # Core C++ engine (Voxel-based BKI)
│   ├── cli.py                  # Command-line interface
│   └── setup.py                # Build & installation
│
├── configs/                    # Dataset configurations
│   ├── mcd_config.yaml         # MCD configuration
│   └── kitti_config.yaml       # SemanticKITTI configuration
│
├── examples/                   # Usage examples
│   └── basic_usage.py          # Demo script
│
├── benchmarks/                 # Performance benchmarks
│
├── scripts/                    # Utility scripts
│   ├── visualize_osm.py        # Visualize OSM geometries
│   └── copy_kth_data.py        # Data management
│
└── example_data/               # Sample data
```

## Quick Start

### Installation

Build and install the Python package from source:

```bash
cd src/
pip install .
```

For development (editable install):
```bash
cd src/
pip install -e .
```

### Usage

The `composite-bki` tool (installed via `pip`) is the main entry point.

#### 1. Single Scan Refinement (Classic Mode)
Process one scan at a time, treating each independently.

```bash
composite-bki --scan scan.bin --label labels.label --osm map.bin \
              --config configs/mcd_config.yaml \
              --output refined.label
```

#### 2. Continuous Mapping (New!)
Accumulate semantic evidence over multiple scans into a persistent voxel map.

```bash
# Enable continuous mode with --continuous
composite-bki --continuous \
              --scan scan_01.bin --label labels_01.label \
              --osm map.bin \
              --config configs/mcd_config.yaml \
              --output refined_01.label \
              --save-map my_map.bki
```

You can then load the map for subsequent scans:

```bash
composite-bki --continuous \
              --load-map my_map.bki \
              --scan scan_02.bin --label labels_02.label \
              --osm map.bin \
              --config configs/mcd_config.yaml \
              --output refined_02.label \
              --save-map my_map_updated.bki
```

### Python API

```python
import composite_bki_cpp
import numpy as np

# Initialize BKI Engine
bki = composite_bki_cpp.PyContinuousBKI(
    osm_path="map.bin",
    config_path="configs/mcd_config.yaml",
    resolution=0.1,  # Voxel size
    l_scale=3.0      # Spatial kernel scale
)

# Load data (Numpy arrays)
points = np.fromfile("scan.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
labels = np.fromfile("labels.label", dtype=np.uint32)

# Update the map
bki.update(labels, points)

# Infer refined labels
refined_labels = bki.infer(points)

# Save map state
bki.save("global_map.bki")
```

## Features

- **Unified Backend**: Single C++ engine (`ContinuousBKI`) handles all operations.
- **Voxel-based**: Uses sparse voxel hashing for O(1) lookups and memory efficiency.
- **Stateful**: Can accumulate semantic evidence across thousands of scans.
- **Fast**: Multi-threaded with OpenMP.
- **Flexible**: YAML-based configuration for any label format.

## Requirements

- Python 3.7+
- NumPy >= 1.19.0
- Cython >= 0.29.0
- g++ with OpenMP support

## Citation

If you use this work, please cite:

```bibtex
@software{composite_bki,
  title={Composite BKI: Bayesian Kernel Inference for LiDAR Semantic Segmentation},
  author={Composite BKI Team},
  year={2026},
  url={https://github.com/yourrepo/composite-bki}
}
```

## License

MIT License
