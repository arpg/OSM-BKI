# Dataset Visualization Examples

Scripts for visualizing semantic LiDAR maps from the **KITTI-360** and **MCD**
datasets using Open3D, with optional OSM overlay and uncertainty analysis.

## Scripts

| Script | Dataset | Description |
|---|---|---|
| `visualize_sem_map_KITTI360.py` | KITTI-360 | Visualize sequences from KITTI-360 |
| `visualize_sem_map_MCD.py` | MCD | Visualize sequences from MCD |

Both scripts share a common set of arguments for consistency:

| Argument | Default | Description |
|---|---|---|
| `--dataset-path` | `example_data/kitti360` or `example_data/mcd` | Root dataset directory |
| `--downsample-factor` | `1` | Process every Nth scan |
| `--voxel-size` | `1.0` (KITTI360) / `0.5` (MCD) | Voxel downsampling size in meters |
| `--max-distance` | none / `200.0` | Max distance from sensor per scan (m) |
| `--max-scans` | none / `10000` | Max number of scans to process |
| `--inferred-labels` | `semkitti` | Inference label set: `semkitti`, `kitti360`, `mcd` |
| `--use-multiclass` | off | Enable multiclass mode with accuracy analysis |
| `--filter-by-confusion` | off | Filter points using confusion matrix precision (see below) |
| `--variance` | off | Color by uncertainty (Viridis colormap) |
| `--max-uncertainty` | off | Keep points with uncertainty <= threshold in [0,1] (multiclass only) |
| `--view` | `all` | Filter: `all`, `correct`, or `incorrect` |
| `--with-osm` | off | Overlay OSM building/road outlines |
| `--osm-thickness` | `2.0` | OSM line thickness (m) |
| `--osm-all` | off | Show all OSM features (not just buildings) |
| `--with-path` | off | Show the drive path |
| `--path-thickness` | `1.5` | Path line thickness (m) |

Additional MCD-only arguments:

| Argument | Default | Description |
|---|---|---|
| `--seq` | `kth_day_09` | Sequence name(s) (accepts multiple) |
| `--calib` | `<dataset-path>/hhs_calib.yaml` | Body-to-LiDAR calibration YAML |

KITTI-360 uses `--sequence` (integer index, default `9`).

## Running from the OSM-BKI root directory

### KITTI-360

One-hot GT labels (default):

```bash
python python/scripts/dataset_examples/visualize_sem_map_KITTI360.py
```

Multiclass with accuracy analysis:

```bash
python python/scripts/dataset_examples/visualize_sem_map_KITTI360.py \
    --use-multiclass --inferred-labels semkitti
```

With OSM overlay and drive path:

```bash
python python/scripts/dataset_examples/visualize_sem_map_KITTI360.py \
    --with-osm --with-path --downsample-factor 2
```

With variance visualization:

```bash
python python/scripts/dataset_examples/visualize_sem_map_KITTI360.py \
    --use-multiclass --variance --view incorrect
```

With uncertainty thresholding:

```bash
python python/scripts/dataset_examples/visualize_sem_map_KITTI360.py \
    --use-multiclass --max-uncertainty 0.3
```

With confusion-matrix filtering:

```bash
python python/scripts/dataset_examples/visualize_sem_map_KITTI360.py \
    --use-multiclass --filter-by-confusion
```

### MCD

One-hot GT labels (default):

```bash
python python/scripts/dataset_examples/visualize_sem_map_MCD.py
```

Multiclass with accuracy analysis:

```bash
python python/scripts/dataset_examples/visualize_sem_map_MCD.py \
    --use-multiclass --inferred-labels semkitti
```

With OSM overlay and drive path:

```bash
python python/scripts/dataset_examples/visualize_sem_map_MCD.py \
    --with-osm --with-path --downsample-factor 5
```

Custom dataset path and sequence:

```bash
python python/scripts/dataset_examples/visualize_sem_map_MCD.py \
    --dataset-path /path/to/mcd --seq kth_day_09 kth_day_10
```

With confusion-matrix filtering:

```bash
python python/scripts/dataset_examples/visualize_sem_map_MCD.py \
    --use-multiclass --filter-by-confusion
```

With uncertainty thresholding:

```bash
python python/scripts/dataset_examples/visualize_sem_map_MCD.py \
    --use-multiclass --max-uncertainty 0.3
```

## Confusion Matrix Filtering

When `--use-multiclass` is enabled, the scripts compute accuracy analysis
including per-uncertainty-bin accuracy, Spearman correlation, and AUROC.

The `--filter-by-confusion` flag adds a confusion-matrix-driven filtering step
that adapts to each predicted class individually. It works as follows:

1. A confusion matrix (predicted x true) is built over all valid points in the
   common taxonomy.
2. For each predicted class, the **precision** is computed: the fraction of
   predictions for that class that are correct.
3. The precision is used as a percentile threshold on uncertainty. For a class
   with 80% precision, the 80% most confident predictions are kept and the 20%
   most uncertain are dropped.

This means high-precision classes (e.g. road at 98%) lose almost no points,
while low-precision classes (e.g. other-object at 45%) are aggressively pruned.
The approach requires no tuning parameters -- the confusion matrix itself
determines how much to trust each class.

## Label Configurations

Semantic label definitions for each dataset are stored in YAML files:

- `labels_kitti360.yaml` -- KITTI-360 label IDs, colors, and `learning_map` / `learning_map_inv`
- `labels_semkitti.yaml` -- SemanticKITTI label definitions
- `labels_mcd.yaml` -- MCD label definitions

The `label_mappings.py` module defines a **common 13-class taxonomy** and
provides per-dataset mappings (`KITTI360_TO_COMMON`, `SEMKITTI_TO_COMMON`,
`MCD_TO_COMMON`) so that GT and inferred labels from any dataset can be
compared in a shared label space. Labels listed in `IGNORE_LABELS` (e.g.
`unlabeled`) are excluded from accuracy calculations.

## GPS Origin Coordinates

OSM data is aligned to the LiDAR maps using dataset-specific GPS origin
coordinates (defined in `utils.py`):

| Dataset | Latitude | Longitude |
|---|---|---|
| KITTI-360 | 48.9843445 | 8.4295857 |
| MCD | 59.347671416 | 18.072069652 |

These origins define the world-frame `(0, 0, 0)` for Web Mercator projection,
allowing OSM features to be rendered in the same coordinate system as the LiDAR
point clouds.

## Shared Utilities

`utils.py` contains shared code used by both scripts:

- Binary file I/O (`read_bin_file`)
- Label config loading (`load_label_config`)
- Class index to label mapping (`map_class_indices_to_labels`)
- Web Mercator projection functions
- OSM geometry loading and rendering (`OSMLoader`)
- Drive path visualization (`create_path_geometry`)
- Viridis colormap for uncertainty visualization
- Accuracy / uncertainty analysis (`run_accuracy_analysis`)
- Confusion-matrix-based point filtering (`filter_by_confusion_matrix`)
