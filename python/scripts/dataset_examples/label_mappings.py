"""
Common taxonomy for cross-dataset label comparison.

Maps labels from KITTI-360, SemanticKITTI, and MCD to a shared 13-class
taxonomy so that GT labels and inferred labels (from any network) can be
compared in the same space.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Common taxonomy: 13 classes (0-12)
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
    0: [0, 0, 0],        # unlabeled
    1: [255, 0, 255],    # road
    2: [244, 35, 232],   # sidewalk
    3: [160, 170, 250],  # parking
    4: [232, 35, 244],   # other-ground
    5: [0, 100, 0],      # building
    6: [153, 153, 190],  # fence
    7: [153, 153, 153],  # pole
    8: [0, 255, 255],    # traffic-sign
    9: [35, 142, 107],   # vegetation
    10: [119, 11, 32],   # two-wheeler
    11: [51, 0, 51],     # vehicle
    12: [255, 255, 50],  # other-object
}

# Labels to ignore in calculating accuracy
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
    31: 12,     # bicyclist           -> other-object
    32: 12,     # motorcyclist        -> other-object
    40: 1,      # road
    44: 3,      # parking
    48: 2,      # sidewalk
    49: 4,      # other-ground
    50: 5,      # building
    51: 6,      # fence
    52: 12,     # other-structure     -> other-object
    60: 1,      # lane-marking        -> road
    70: 9,      # vegetation
    71: 9,      # trunk               -> vegetation
    72: 4,      # terrain             -> other-ground
    80: 7,      # pole
    81: 8,      # traffic-sign
    99: 12,     # other-object
    252: 11,    # moving-car          -> vehicle
    253: 12,    # moving-bicyclist    -> other-object
    254: 12,    # moving-person       -> other-object
    255: 12,    # moving-motorcyclist -> other-object
    256: 11,    # moving-on-rails     -> vehicle
    257: 11,    # moving-bus          -> vehicle
    258: 11,    # moving-truck        -> vehicle
    259: 11,    # moving-other-veh.   -> vehicle
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
