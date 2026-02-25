#!/usr/bin/env python3
"""
Label file utilities: convert between KITTI/MCD formats and inspect label files.

Subcommands:
  kitti2mcd  - Convert SemanticKITTI labels to MCD format
  mcd2kitti  - Convert MCD labels to SemanticKITTI format
  prettyprint - Inspect label file (distribution, format detection)
"""

import argparse
import os
from pathlib import Path
import numpy as np


# ---------------------------------------------------------------------------
# Common taxonomy for cross-dataset label comparison.
#
# Maps labels from KITTI-360, SemanticKITTI, and MCD to a shared 13-class
# taxonomy so that GT labels and inferred labels (from any network) can be
# compared in the same space.
# ---------------------------------------------------------------------------

N_COMMON = 13

COMMON_LABELS = {
    0: "unlabeled",
    1: "road",
    2: "sidewalk",
    3: "parking",
    4: "other-ground",
    5: "building",
    6: "fence",
    7: "pole",
    8: "traffic-sign",
    9: "vegetation",
    10: "two-wheeler",
    11: "vehicle",
    12: "other-object",
}

COMMON_COLORS_RGB = {
    0:  [0,   0,   0],    # unlabeled
    1:  [255, 0,   255],  # road
    2:  [244, 35,  232],  # sidewalk
    3:  [160, 170, 250],  # parking
    4:  [232, 35,  244],  # other-ground
    5:  [0,   100, 0],    # building
    6:  [153, 153, 190],  # fence
    7:  [153, 153, 153],  # pole
    8:  [0,   255, 255],  # traffic-sign
    9:  [35,  142, 107],  # vegetation
    10: [119, 11,  32],   # two-wheeler
    11: [51,  0,   51],   # vehicle
    12: [255, 255, 50],   # other-object
}

# Labels to ignore when calculating accuracy metrics
IGNORE_LABELS = [0]

# ---------------------------------------------------------------------------
# Per-dataset raw-label-ID -> common-taxonomy-ID
# ---------------------------------------------------------------------------

KITTI360_TO_COMMON = {
    0: 0,       # unlabeled
    1: 0,       # ego vehicle        -> unlabeled
    2: 0,       # rectification bdr  -> unlabeled
    3: 0,       # out of roi         -> unlabeled
    4: 12,      # static             -> other-object
    5: 12,      # dynamic            -> other-object
    6: 4,       # ground             -> other-ground
    7: 1,       # road
    8: 2,       # sidewalk
    9: 3,       # parking
    10: 4,      # rail track         -> other-ground
    11: 5,      # building
    12: 6,      # wall               -> fence
    13: 6,      # fence
    14: 6,      # guard rail         -> fence
    15: 5,      # bridge             -> building
    16: 5,      # tunnel             -> building
    17: 7,      # pole
    18: 7,      # polegroup          -> pole
    19: 8,      # traffic light      -> traffic-sign
    20: 8,      # traffic sign
    21: 9,      # vegetation
    22: 4,      # terrain            -> other-ground
    23: 0,      # sky                -> unlabeled
    24: 12,     # person             -> other-object
    25: 12,     # rider              -> other-object
    26: 11,     # car                -> vehicle
    27: 11,     # truck              -> vehicle
    28: 11,     # bus                -> vehicle
    29: 11,     # caravan            -> vehicle
    30: 11,     # trailer            -> vehicle
    31: 11,     # train              -> vehicle
    32: 10,     # motorcycle         -> two-wheeler
    33: 10,     # bicycle            -> two-wheeler
    34: 5,      # garage             -> building
    35: 6,      # gate               -> fence
    36: 8,      # stop               -> traffic-sign
    37: 7,      # smallpole          -> pole
    38: 7,      # lamp               -> pole
    39: 12,     # trash bin          -> other-object
    40: 12,     # vending machine    -> other-object
    41: 12,     # box                -> other-object
    42: 12,     # unknown constr.    -> other-object
    43: 11,     # unknown vehicle    -> vehicle
    44: 12,     # unknown object     -> other-object
    65535: 0,   # invalid            -> unlabeled
}

SEMKITTI_TO_COMMON = {
    0: 0,       # unlabeled
    1: 0,       # outlier            -> unlabeled
    10: 11,     # car                -> vehicle
    11: 10,     # bicycle            -> two-wheeler
    13: 11,     # bus                -> vehicle
    15: 10,     # motorcycle         -> two-wheeler
    16: 11,     # on-rails           -> vehicle
    18: 11,     # truck              -> vehicle
    20: 11,     # other-vehicle      -> vehicle
    30: 12,     # person             -> other-object
    31: 12,     # bicyclist          -> other-object
    32: 12,     # motorcyclist       -> other-object
    40: 1,      # road
    44: 3,      # parking
    48: 2,      # sidewalk
    49: 4,      # other-ground
    50: 5,      # building
    51: 6,      # fence
    52: 12,     # other-structure    -> other-object
    60: 1,      # lane-marking       -> road
    70: 9,      # vegetation
    71: 9,      # trunk              -> vegetation
    72: 4,      # terrain            -> other-ground
    80: 7,      # pole
    81: 8,      # traffic-sign
    99: 12,     # other-object
    252: 11,    # moving-car         -> vehicle
    253: 12,    # moving-bicyclist   -> other-object
    254: 12,    # moving-person      -> other-object
    255: 12,    # moving-motorcyclist -> other-object
    256: 11,    # moving-on-rails    -> vehicle
    257: 11,    # moving-bus         -> vehicle
    258: 11,    # moving-truck       -> vehicle
    259: 11,    # moving-other-veh.  -> vehicle
}

MCD_TO_COMMON = {
    0: 6,       # barrier            -> fence
    1: 10,      # bike               -> two-wheeler
    2: 5,       # building
    3: 12,      # chair              -> other-object
    4: 4,       # cliff              -> other-ground
    5: 12,      # container          -> other-object
    6: 4,       # curb               -> other-ground
    7: 6,       # fence
    8: 12,      # hydrant            -> other-object
    9: 8,       # infosign           -> traffic-sign
    10: 1,      # lanemarking        -> road
    11: 0,      # noise              -> unlabeled
    12: 12,     # other              -> other-object
    13: 3,      # parkinglot         -> parking
    14: 12,     # pedestrian         -> other-object
    15: 7,      # pole
    16: 1,      # road
    17: 5,      # shelter            -> building
    18: 2,      # sidewalk
    19: 4,      # stairs             -> other-ground
    20: 12,     # structure-other    -> other-object
    21: 8,      # traffic-cone       -> traffic-sign
    22: 8,      # traffic-sign
    23: 12,     # trashbin           -> other-object
    24: 9,      # treetrunk          -> vegetation
    25: 9,      # vegetation
    26: 11,     # vehicle-dynamic    -> vehicle
    27: 11,     # vehicle-other      -> vehicle
    28: 11,     # vehicle-static     -> vehicle
}

DATASET_TO_COMMON = {
    "kitti360": KITTI360_TO_COMMON,
    "semkitti": SEMKITTI_TO_COMMON,
    "mcd": MCD_TO_COMMON,
}


def build_to_common_lut(dataset_name: str) -> np.ndarray:
    """
    Build a numpy LUT that maps dataset raw label IDs to common taxonomy IDs.
    Unmapped IDs default to 0 (unlabeled).
    """
    mapping = DATASET_TO_COMMON[dataset_name]
    max_id = max(mapping.keys())
    lut = np.zeros(max_id + 1, dtype=np.int32)
    for k, v in mapping.items():
        lut[k] = v
    return lut


def apply_common_lut(label_ids, lut: np.ndarray) -> np.ndarray:
    """Apply a common-taxonomy LUT to an array of label IDs."""
    ids = np.asarray(label_ids, dtype=np.int64)
    safe = np.clip(ids, 0, len(lut) - 1)
    return lut[safe]


_COMMON_COLOR_LUT = np.zeros((N_COMMON, 3), dtype=np.float64)
for _cid, _rgb in COMMON_COLORS_RGB.items():
    _COMMON_COLOR_LUT[_cid] = np.array(_rgb, dtype=np.float64) / 255.0


def common_labels_to_colors(common_ids) -> np.ndarray:
    """Map (N,) common taxonomy IDs to (N, 3) RGB in [0, 1]."""
    ids = np.clip(np.asarray(common_ids, dtype=np.int32), 0, N_COMMON - 1)
    return _COMMON_COLOR_LUT[ids].copy()


# IDs that are exclusive to one dataset and therefore act as fingerprints.
# SemanticKITTI: sidewalk/building/vegetation/pole start at 48; moving objects at 252+.
_SEMKITTI_SIGNATURE_IDS = frozenset({48, 49, 50, 51, 52, 60, 70, 71, 72, 80, 81, 99,
                                      252, 253, 254, 255, 256, 257, 258, 259})
# KITTI-360: invalid sentinel, plus IDs 34-38 (garage/gate/stop/smallpole/lamp)
# which fall in the 0-44 range but are never used by SemanticKITTI.
_KITTI360_SIGNATURE_IDS = frozenset({65535, 34, 35, 36, 37, 38})


def detect_dataset(label_ids) -> str:
    """
    Infer which dataset a set of label IDs belongs to.
    """
    unique_ids = frozenset(int(x) for x in np.unique(np.asarray(label_ids)))

    if unique_ids & _SEMKITTI_SIGNATURE_IDS or any(i >= 48 for i in unique_ids):
        return "semkitti"

    if unique_ids & _KITTI360_SIGNATURE_IDS:
        return "kitti360"

    max_id = max(unique_ids) if unique_ids else 0
    if max_id <= 28:
        return "mcd"

    # IDs in 29-44 with no distinctive signature: SemanticKITTI only uses 40
    # and 44 in this band (road and parking), which overlap with KITTI-360.
    # Treat as KITTI-360 since it is the only schema with a contiguous 0-44 range.
    if max_id <= 44:
        return "kitti360"

    raise ValueError(
        f"Could not determine dataset: max label ID {max_id} with unique IDs {sorted(unique_ids)}"
    )


# SemanticKITTI -> MCD mapping (from convert_data.py / test_idea.py)
KITTI_TO_MCD = {
    40: 16,  # Road -> Road
    44: 13,  # Parking -> Parkinglot
    48: 18,  # Sidewalk -> Sidewalk
    49: 18,  # Other-ground -> Sidewalk (approx)
    70: 25,  # Vegetation -> Vegetation
    71: 24,  # Trunk -> Treetrunk
    72: 25,  # Terrain -> Vegetation (approx)
    50: 2,   # Building -> Building
    51: 7,   # Fence -> Fence
    52: 20,  # Other-structure -> Structure-other
    10: 26, 11: 1, 13: 26, 15: 26, 16: 26, 18: 26, 20: 27,  # Vehicles -> Dyn/Bike/Other
    30: 14, 31: 1, 32: 26,  # Person -> Ped, Bicyclist -> Bike, Motorcyclist -> Dyn
    80: 15,  # Pole -> Pole
    81: 22,  # Traffic-sign -> Traffic-sign
    99: 12,  # Other-object -> Other
    0: 0,    # Unlabeled -> Barrier
    1: 12,   # Outlier -> Other
}

# MCD -> SemanticKITTI (canonical inverse; many MCD labels have no KITTI equivalent -> 0)
MCD_TO_KITTI = {
    0: 0,   # barrier / unlabeled
    1: 11,  # bike
    2: 50,  # building
    3: 99,  # chair -> other-object
    4: 99,  # cliff
    5: 99,  # container
    6: 48,  # curb -> sidewalk
    7: 51,  # fence
    8: 80,  # hydrant -> pole
    9: 81,  # infosign -> traffic-sign
    10: 48, # lanemarking -> sidewalk
    11: 0,  # noise -> unlabeled
    12: 99, # other
    13: 44, # parkinglot
    14: 30, # pedestrian
    15: 80, # pole
    16: 40, # road
    17: 50, # shelter -> building
    18: 48, # sidewalk
    19: 48, # stairs -> sidewalk
    20: 52, # structure-other
    21: 99, # traffic-cone
    22: 81, # traffic-sign
    23: 99, # trashbin
    24: 71, # treetrunk
    25: 70, # vegetation
    26: 10, # vehicle-dynamic -> car
    27: 20, # vehicle-other
    28: 10, # vehicle-static -> car
}


def convert_kitti_to_mcd(input_path: str, output_path: str):
    """Convert SemanticKITTI label file to MCD format."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Converting {input_path} -> {output_path}")
    raw_data = np.fromfile(input_path, dtype=np.uint32)
    sem_labels = raw_data & 0xFFFF
    instance_ids = raw_data & 0xFFFF0000

    new_sem_labels = np.full_like(sem_labels, 12)
    for k_id, m_id in KITTI_TO_MCD.items():
        mask = sem_labels == k_id
        new_sem_labels[mask] = m_id

    explicit_other = {k for k, v in KITTI_TO_MCD.items() if v == 12}
    unmapped = np.unique(sem_labels[new_sem_labels == 12])
    really_unmapped = [u for u in unmapped if u not in explicit_other]
    if really_unmapped:
        print(f"Warning: SemanticKITTI classes mapped to Other (12) by default: {really_unmapped}")

    final_data = instance_ids | new_sem_labels.astype(np.uint32)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_data.tofile(output_path)


def convert_mcd_to_kitti(input_path: str, output_path: str):
    """Convert MCD label file to SemanticKITTI format."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Converting {input_path} -> {output_path}")
    raw_data = np.fromfile(input_path, dtype=np.uint32)
    sem_labels = raw_data & 0xFFFF
    instance_ids = raw_data & 0xFFFF0000

    new_sem_labels = np.full_like(sem_labels, 0)  # unlabeled default
    for m_id, k_id in MCD_TO_KITTI.items():
        mask = sem_labels == m_id
        new_sem_labels[mask] = k_id

    final_data = instance_ids | new_sem_labels.astype(np.uint32)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_data.tofile(output_path)


def prettyprint_labels(label_file: str):
    """Inspect label file (same logic as debug_labels.py)."""
    print(f"\n=== Inspecting: {label_file} ===")

    labels_raw = np.fromfile(label_file, dtype=np.uint32)
    labels = labels_raw & 0xFFFF

    print(f"Total points: {len(labels)}")
    print(f"Raw label range: [{labels_raw.min()}, {labels_raw.max()}]")
    print(f"Semantic label range: [{labels.min()}, {labels.max()}]")

    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\nUnique labels found: {len(unique_labels)}")
    print("Label distribution:")

    sorted_indices = np.argsort(counts)[::-1]
    for idx in sorted_indices[:20]:
        label = unique_labels[idx]
        count = counts[idx]
        pct = (count / len(labels)) * 100
        print(f"  Label {label:3d}: {count:8d} points ({pct:5.2f}%)")

    if len(unique_labels) > 20:
        print(f"  ... and {len(unique_labels) - 20} more labels")

    max_label = labels.max()
    has_kitti = any(l in [40, 44, 48, 50, 70, 80, 81] for l in unique_labels)

    print("\n--- Format Detection ---")
    print(f"Max label: {max_label}")
    print(f"Has KITTI-specific labels: {has_kitti}")
    if max_label > 30 or has_kitti:
        print("→ Detected as: SemanticKITTI format")
    else:
        print("→ Detected as: MCD format")

    print("\n--- Potential Issues ---")
    if len(unique_labels) == 1:
        print("⚠️  WARNING: Only ONE unique label found!")
    elif np.sum(labels == 0) / len(labels) > 0.9:
        print("⚠️  WARNING: >90% of points have label 0!")
    else:
        print("✓ Label distribution looks reasonable")


def _glob_label_files(directory: str):
    """Return sorted .label then .bin files in directory."""
    files = sorted(Path(directory).glob("*.label"))
    if not files:
        files = sorted(Path(directory).glob("*.bin"))
    return [str(f) for f in files]


def cmd_kitti2mcd(args):
    converter = convert_kitti_to_mcd
    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        files = _glob_label_files(args.input)
        print(f"Found {len(files)} files in {args.input}")
        for fp in files:
            out = os.path.join(args.output, os.path.basename(fp))
            converter(fp, out)
        print("Batch conversion done.")
    else:
        converter(args.input, args.output)
        print("Done.")
    return 0


def cmd_mcd2kitti(args):
    converter = convert_mcd_to_kitti
    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        files = _glob_label_files(args.input)
        print(f"Found {len(files)} files in {args.input}")
        for fp in files:
            out = os.path.join(args.output, os.path.basename(fp))
            converter(fp, out)
        print("Batch conversion done.")
    else:
        converter(args.input, args.output)
        print("Done.")
    return 0


def cmd_prettyprint(args):
    prettyprint_labels(args.label_file)
    return 0


def convert_to_common(input_path: str, output_path: str, dataset_name: str):
    """Convert a label file from a named dataset to the common 13-class taxonomy."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Converting {input_path} -> {output_path}  [{dataset_name} -> common]")
    raw_data = np.fromfile(input_path, dtype=np.uint32)
    sem_labels = raw_data & 0xFFFF
    instance_ids = raw_data & 0xFFFF0000

    lut = build_to_common_lut(dataset_name)
    new_sem_labels = apply_common_lut(sem_labels, lut).astype(np.uint32)

    final_data = instance_ids | new_sem_labels
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_data.tofile(output_path)


def _cmd_to_common(args, dataset_name: str):
    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        files = _glob_label_files(args.input)
        print(f"Found {len(files)} files in {args.input}")
        for fp in files:
            out = os.path.join(args.output, os.path.basename(fp))
            convert_to_common(fp, out, dataset_name)
        print("Batch conversion done.")
    else:
        convert_to_common(args.input, args.output, dataset_name)
        print("Done.")
    return 0


def cmd_kitti2common(args):
    return _cmd_to_common(args, "semkitti")


def cmd_mcd2common(args):
    return _cmd_to_common(args, "mcd")


def cmd_kitti3602common(args):
    return _cmd_to_common(args, "kitti360")


def main():
    parser = argparse.ArgumentParser(
        description="Label file utilities: convert KITTI<->MCD and inspect.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # kitti2mcd
    p = subparsers.add_parser("kitti2mcd", help="Convert SemanticKITTI labels to MCD")
    p.add_argument("--input", "-i", required=True, help="Input .label/.bin file or directory")
    p.add_argument("--output", "-o", required=True, help="Output file or directory")
    p.set_defaults(func=cmd_kitti2mcd)

    # mcd2kitti
    p = subparsers.add_parser("mcd2kitti", help="Convert MCD labels to SemanticKITTI")
    p.add_argument("--input", "-i", required=True, help="Input .label/.bin file or directory")
    p.add_argument("--output", "-o", required=True, help="Output file or directory")
    p.set_defaults(func=cmd_mcd2kitti)

    # kitti-to-common
    p = subparsers.add_parser("kitti-to-common", help="Convert SemanticKITTI labels to common 13-class taxonomy")
    p.add_argument("--input", "-i", required=True, help="Input .label/.bin file or directory")
    p.add_argument("--output", "-o", required=True, help="Output file or directory")
    p.set_defaults(func=cmd_kitti2common)

    # mcd-to-common
    p = subparsers.add_parser("mcd-to-common", help="Convert MCD labels to common 13-class taxonomy")
    p.add_argument("--input", "-i", required=True, help="Input .label/.bin file or directory")
    p.add_argument("--output", "-o", required=True, help="Output file or directory")
    p.set_defaults(func=cmd_mcd2common)

    # kitti360-to-common
    p = subparsers.add_parser("kitti360-to-common", help="Convert KITTI-360 labels to common 13-class taxonomy")
    p.add_argument("--input", "-i", required=True, help="Input .label/.bin file or directory")
    p.add_argument("--output", "-o", required=True, help="Output file or directory")
    p.set_defaults(func=cmd_kitti3602common)

    # prettyprint
    p = subparsers.add_parser("prettyprint", help="Inspect label file (distribution, format)")
    p.add_argument("label_file", help="Path to .label or .bin file")
    p.set_defaults(func=cmd_prettyprint)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
