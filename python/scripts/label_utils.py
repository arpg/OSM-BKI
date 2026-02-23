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

    # prettyprint
    p = subparsers.add_parser("prettyprint", help="Inspect label file (distribution, format)")
    p.add_argument("label_file", help="Path to .label or .bin file")
    p.set_defaults(func=cmd_prettyprint)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
