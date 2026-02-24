"""
Common taxonomy for cross-dataset label comparison.

Maps labels from KITTI-360, SemanticKITTI, and MCD to a shared 17-class
taxonomy so that GT labels and inferred labels (from any network) can be
compared in the same space.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Common taxonomy: 17 classes (0-16)
# ---------------------------------------------------------------------------

N_COMMON = 17

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
    10: "terrain",
    11: "person",
    12: "car",
    13: "truck",
    14: "two-wheeler",
    15: "other-vehicle",
    16: "other-object",
}

COMMON_COLORS_RGB = {
    0: [0, 0, 0],
    1: [128, 64, 128],
    2: [244, 35, 232],
    3: [250, 170, 160],
    4: [81, 0, 81],
    5: [70, 70, 70],
    6: [190, 153, 153],
    7: [153, 153, 153],
    8: [220, 220, 0],
    9: [107, 142, 35],
    10: [152, 251, 152],
    11: [220, 20, 60],
    12: [0, 0, 142],
    13: [0, 0, 70],
    14: [119, 11, 32],
    15: [0, 60, 100],
    16: [255, 255, 50],
}


# ---------------------------------------------------------------------------
# Per-dataset raw-label-ID -> common-taxonomy-ID
# ---------------------------------------------------------------------------

KITTI360_TO_COMMON = {
    0: 0,       # unlabeled
    1: 0,       # ego vehicle        -> unlabeled
    2: 0,       # rectification bdr  -> unlabeled
    3: 0,       # out of roi         -> unlabeled
    4: 16,      # static             -> other-object
    5: 16,      # dynamic            -> other-object
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
    22: 10,     # terrain
    23: 0,      # sky                -> unlabeled
    24: 11,     # person
    25: 11,     # rider              -> person
    26: 12,     # car
    27: 13,     # truck
    28: 15,     # bus                -> other-vehicle
    29: 15,     # caravan            -> other-vehicle
    30: 15,     # trailer            -> other-vehicle
    31: 15,     # train              -> other-vehicle
    32: 14,     # motorcycle         -> two-wheeler
    33: 14,     # bicycle            -> two-wheeler
    34: 5,      # garage             -> building
    35: 6,      # gate               -> fence
    36: 8,      # stop               -> traffic-sign
    37: 7,      # smallpole          -> pole
    38: 7,      # lamp               -> pole
    39: 16,     # trash bin          -> other-object
    40: 16,     # vending machine    -> other-object
    41: 16,     # box                -> other-object
    42: 16,     # unknown constr.    -> other-object
    43: 15,     # unknown vehicle    -> other-vehicle
    44: 16,     # unknown object     -> other-object
    65535: 0,   # invalid            -> unlabeled
}

SEMKITTI_TO_COMMON = {
    0: 0,       # unlabeled
    1: 0,       # outlier            -> unlabeled
    10: 12,     # car
    11: 14,     # bicycle            -> two-wheeler
    13: 15,     # bus                -> other-vehicle
    15: 14,     # motorcycle         -> two-wheeler
    16: 15,     # on-rails           -> other-vehicle
    18: 13,     # truck
    20: 15,     # other-vehicle
    30: 11,     # person
    31: 11,     # bicyclist           -> person
    32: 11,     # motorcyclist        -> person
    40: 1,      # road
    44: 3,      # parking
    48: 2,      # sidewalk
    49: 4,      # other-ground
    50: 5,      # building
    51: 6,      # fence
    52: 16,     # other-structure     -> other-object
    60: 1,      # lane-marking        -> road
    70: 9,      # vegetation
    71: 9,      # trunk               -> vegetation
    72: 10,     # terrain
    80: 7,      # pole
    81: 8,      # traffic-sign
    99: 16,     # other-object
    252: 12,    # moving-car          -> car
    253: 11,    # moving-bicyclist    -> person
    254: 11,    # moving-person       -> person
    255: 11,    # moving-motorcyclist -> person
    256: 15,    # moving-on-rails     -> other-vehicle
    257: 15,    # moving-bus          -> other-vehicle
    258: 13,    # moving-truck        -> truck
    259: 15,    # moving-other-veh.   -> other-vehicle
}

MCD_TO_COMMON = {
    0: 6,       # barrier            -> fence
    1: 14,      # bike               -> two-wheeler
    2: 5,       # building
    3: 16,      # chair              -> other-object
    4: 10,      # cliff              -> terrain
    5: 16,      # container          -> other-object
    6: 4,       # curb               -> other-ground
    7: 6,       # fence
    8: 16,      # hydrant            -> other-object
    9: 8,       # infosign           -> traffic-sign
    10: 1,      # lanemarking        -> road
    11: 0,      # noise              -> unlabeled
    12: 16,     # other              -> other-object
    13: 3,      # parkinglot         -> parking
    14: 11,     # pedestrian         -> person
    15: 7,      # pole
    16: 1,      # road
    17: 5,      # shelter            -> building
    18: 2,      # sidewalk
    19: 4,      # stairs             -> other-ground
    20: 16,     # structure-other    -> other-object
    21: 8,      # traffic-cone       -> traffic-sign
    22: 8,      # traffic-sign
    23: 16,     # trashbin           -> other-object
    24: 9,      # treetrunk          -> vegetation
    25: 9,      # vegetation
    26: 12,     # vehicle-dynamic    -> car
    27: 15,     # vehicle-other      -> other-vehicle
    28: 12,     # vehicle-static     -> car
}

DATASET_TO_COMMON = {
    "kitti360": KITTI360_TO_COMMON,
    "semkitti": SEMKITTI_TO_COMMON,
    "mcd": MCD_TO_COMMON,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def build_to_common_lut(dataset_name):
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


def apply_common_lut(label_ids, lut):
    """Apply a common-taxonomy LUT to an array of label IDs."""
    ids = np.asarray(label_ids, dtype=np.int64)
    safe = np.clip(ids, 0, len(lut) - 1)
    return lut[safe]


_COMMON_COLOR_LUT = np.zeros((N_COMMON, 3), dtype=np.float64)
for _cid, _rgb in COMMON_COLORS_RGB.items():
    _COMMON_COLOR_LUT[_cid] = np.array(_rgb, dtype=np.float64) / 255.0


def common_labels_to_colors(common_ids):
    """Map (N,) common taxonomy IDs to (N, 3) RGB in [0, 1]."""
    ids = np.clip(np.asarray(common_ids, dtype=np.int32), 0, N_COMMON - 1)
    return _COMMON_COLOR_LUT[ids].copy()
