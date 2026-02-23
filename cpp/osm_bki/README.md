# osm_bki (C++)

C++ library and tools for OSM-BKI (Bayesian Kernel Inference for LiDAR with OpenStreetMap priors). Here you will find the static library `osm_bki_core` and an optional PCL-based visualization example.

## Requirements

- **CMake** ≥ 3.16  
- **C++17** compiler (e.g. GCC 8+, Clang 6+)  
- **Eigen3** (for core library)  
- **OpenMP** (optional; used when available)

To build the example program `visualize_map_osm` you also need:

- **MPI** (C component)

- **PCL** (Point Cloud Library) with components: `common`, `io`, `visualization`

- **libosmium** (header-only; for reading OSM files) and its runtime deps (expat, bz2, zlib for OSM input)

### Installing dependencies (Ubuntu / Debian)

```bash
sudo apt update
sudo apt install -y build-essential cmake libeigen3-dev libpcl-dev libosmium-dev libopenmpi-dev libexpat1-dev libbz2-dev zlib1g-dev
```

If libosmium is not in your package manager, set `OSMIUM_INCLUDE_DIR` when configuring (see Build below).

### Installing dependencies (MacOS)
```bash
brew install cmake eigen pcl libosmium open-mpi libomp expat bzip2
```

- **Eigen** and **expat** are optional if you only build the core library; **libomp** provides OpenMP on Apple Clang.
- For the `visualize_map_osm` example you need **PCL**, **libosmium**, **open-mpi**, and the rest as above.

If libosmium is not found by CMake, set `OSMIUM_INCLUDE_DIR` when configuring (e.g. `$(brew --prefix libosmium)/include`). See Build below.

## Project structure

```
cpp/osm_bki/
├── CMakeLists.txt
├── README.md
├── include/           # Public headers
│   ├── continuous_bki.hpp
│   ├── dataset_utils.hpp
│   ├── file_io.hpp
│   ├── osm_parser.hpp
│   ├── osm_xml_parser.hpp
│   └── yaml_parser.hpp
├── src/               # Core library and OSM parser
│   ├── continuous_bki.cpp
│   ├── dataset_utils.cpp
│   ├── file_io.cpp
│   ├── osm_loader.cpp
│   └── osm_parser.cpp
├── examples/
│   ├── visualize_map_osm.cpp   # PCL viewer: map + OSM polylines
│   ├── test_cbki.cpp          # Continuous mapping example
│   └── osm_config.yaml        # OSM category filters (for visualize_map_osm)
└── build/             # CMake build output
```

Configs (`mcd_config.yaml`, etc.) and example data live at the **repo root**: `configs/`, `example_data/`. Run from the repo root so paths resolve correctly.

## Build

From the `osm_bki` directory (where this README and `CMakeLists.txt` live):

**Library only (default):**

```bash
cmake -S . -B build
cmake --build build
```

**Library + example (`visualize_map_osm`):**

```bash
cmake -S . -B build -Dosm_bki_BUILD_PCL_EXAMPLES=ON
cmake --build build
```

If libosmium is installed in a non-standard location:

```bash
cmake -S . -B build -Dosm_bki_BUILD_PCL_EXAMPLES=ON -DOSMIUM_INCLUDE_DIR=/path/to/osmium/include
cmake --build build
```

The example executable is produced at `build/visualize_map_osm`.

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `osm_bki_BUILD_PCL_EXAMPLES` | `OFF` | Build the `visualize_map_osm` example (requires PCL and libosmium). |
| `OSMIUM_INCLUDE_DIR` | (empty) | Path to libosmium include directory if not found automatically. |
| `osm_bki_ENABLE_WARNINGS` | `ON` | Enable compiler warnings. |

## Running the example (visualize_map_osm)

`visualize_map_osm` loads an MCD-style dataset (lidar scans, labels, poses), builds a colored point cloud in world frame, parses an OSM file into 2D polylines, and shows both in a PCL viewer.

**Usage:**

```bash
./build/visualize_map_osm [mcd_config.yaml] [osm_config.yaml] [skip_frames]
```

Defaults assume running from **repo root**: `mcd_config.yaml` → `configs/mcd_config.yaml`. `osm_config.yaml` defaults to `examples/osm_config.yaml` (relative to the executable, i.e. `cpp/osm_bki/examples/`).

- **mcd_config.yaml** – Dataset and OSM settings (default: `configs/mcd_config.yaml`; run from repo root).
- **osm_config.yaml** – OSM category toggles (default: `examples/osm_config.yaml` next to the executable).
- **skip_frames** – Optional; overrides `skip_frames` in the MCD config.

**Example (from repo root):**

```bash
./cpp/osm_bki/build/visualize_map_osm configs/mcd_config.yaml
```

Or from `cpp/osm_bki/`:

```bash
./build/visualize_map_osm ../configs/mcd_config.yaml
```

### MCD config (mcd_config.yaml)

Key fields used by the example:

**Path resolution (choose one):**
- **Direct paths** (relative to config file): `lidar_dir`, `label_dir`, `pose_path`, `calibration_path`, `osm_file`.
- **Legacy**: `dataset_root_path` + `sequence` – derive paths under `dataset_root_path/sequence/`.

**Other fields:**
- **init_latlon_day_06** – `[lat, lon]` used as OSM local coordinate origin.
- **init_rel_pos_day_06** – `[x, y, z]` world origin for aligning poses (subtracted from pose positions).
- **skip_frames** – Number of frames to skip between scans used for the map.

### OSM config (examples/osm_config.yaml)

Boolean flags to include or exclude OSM categories: buildings, roads, sidewalks, parking, fences, stairs, grasslands, trees. Lives next to `visualize_map_osm.cpp`; the core library does not depend on this file.

## License

See the top-level OSM-BKI repository license.
