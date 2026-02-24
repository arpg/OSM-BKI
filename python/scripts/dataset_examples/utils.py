"""
Shared utilities for KITTI-360 and MCD semantic map visualization scripts.
"""

import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import open3d as o3d
import yaml

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OSMBKI_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------
INFERRED_LABEL_CONFIGS = {
    "mcd": os.path.join(_SCRIPT_DIR, "labels_mcd.yaml"),
    "semkitti": os.path.join(_SCRIPT_DIR, "labels_semkitti.yaml"),
    "kitti360": os.path.join(_SCRIPT_DIR, "labels_kitti360.yaml"),
}
INFERRED_SUBDIRS = {
    "mcd": "cenet_mcd",
    "semkitti": "cenet_semkitti",
    "kitti360": "cenet_kitti360",
}

KITTI360_ORIGIN_LATLON = (48.9843445, 8.4295857)
MCD_ORIGIN_LATLON = (59.347671416, 18.072069652)

EARTH_RADIUS_M = 6378137.0

OSM_COLORS = {
    "building": [0.8, 0.2, 0.2],
    "highway": [0.35, 0.35, 0.35],
    "landuse": [0.5, 0.85, 0.5],
    "natural": [0.1, 0.55, 0.1],
    "barrier": [0.55, 0.3, 0.1],
    "amenity": [0.4, 0.4, 1.0],
    "default": [0.5, 0.5, 0.5],
}

# ---------------------------------------------------------------------------
# Binary file I/O
# ---------------------------------------------------------------------------

def read_bin_file(file_path, dtype, shape=None):
    """Read a .bin file and optionally reshape."""
    data = np.fromfile(file_path, dtype=dtype)
    if shape is not None:
        return data.reshape(shape)
    return data

# ---------------------------------------------------------------------------
# Label config helpers
# ---------------------------------------------------------------------------

def load_label_config(config_path):
    """Load learning_map and learning_map_inv from a label YAML config."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    learning_map = {int(k): int(v) for k, v in cfg.get("learning_map", {}).items()}
    learning_map_inv = {int(k): int(v) for k, v in cfg.get("learning_map_inv", {}).items()}
    return {"learning_map": learning_map, "learning_map_inv": learning_map_inv}


def map_class_indices_to_labels(class_indices, learning_map_inv):
    """Map class indices (0..n_classes-1) to semantic label IDs via learning_map_inv."""
    maxkey = max(learning_map_inv.keys(), default=0)
    lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, value in learning_map_inv.items():
        try:
            lut[int(key)] = int(value)
        except IndexError:
            pass
    return lut[class_indices]

# ---------------------------------------------------------------------------
# Viridis colormap
# ---------------------------------------------------------------------------

_VIRIDIS_KEY = np.array([
    [0.267004, 0.004874, 0.329415],
    [0.282327, 0.140926, 0.457517],
    [0.127568, 0.566949, 0.550556],
    [0.369214, 0.788888, 0.383287],
    [0.993248, 0.906157, 0.143936],
], dtype=np.float32)
_VIRIDIS_LUT = np.zeros((256, 3), dtype=np.float32)
for _i in range(256):
    _t = _i / 255.0
    _idx = _t * 4
    _j = int(np.clip(np.floor(_idx), 0, 3))
    _u = _idx - _j
    _VIRIDIS_LUT[_i] = (1 - _u) * _VIRIDIS_KEY[_j] + _u * _VIRIDIS_KEY[_j + 1]


def scalar_to_viridis_rgb(values, normalize_range=True):
    """Map scalar values to RGB via Viridis colormap. Returns (N, 3) in [0, 1]."""
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    if normalize_range:
        vmin, vmax = np.min(v), np.max(v)
        rng = vmax - vmin if (vmax - vmin) > 0 else 1.0
        v = (v - vmin) / rng
    v = np.clip(v, 0.0, 1.0)
    idx = np.clip((v * 255).astype(np.int32), 0, 255)
    return _VIRIDIS_LUT[idx].copy()

# ---------------------------------------------------------------------------
# Mercator projection
# ---------------------------------------------------------------------------

def lat_to_scale(lat_deg):
    """Mercator scale factor at given latitude."""
    return math.cos(math.radians(lat_deg))


def latlon_to_mercator_absolute(lat_deg, lon_deg, scale):
    """Convert lat/lon to Web Mercator x, y in meters."""
    lon_rad = math.radians(lon_deg)
    lat_rad = math.radians(lat_deg)
    x = scale * lon_rad * EARTH_RADIUS_M
    y = scale * EARTH_RADIUS_M * math.log(math.tan(math.pi / 4 + lat_rad / 2))
    return x, y


def latlon_to_mercator_relative(lat_deg, lon_deg, origin_lat, origin_lon):
    """Convert lat/lon to meters relative to an origin point via Web Mercator."""
    scale = lat_to_scale(origin_lat)
    ox, oy = latlon_to_mercator_absolute(origin_lat, origin_lon, scale)
    mx, my = latlon_to_mercator_absolute(lat_deg, lon_deg, scale)
    return mx - ox, my - oy


def mercator_relative_to_latlon(rel_x, rel_y, origin_lat, origin_lon):
    """Inverse of latlon_to_mercator_relative."""
    scale = lat_to_scale(origin_lat)
    origin_lon_rad = math.radians(origin_lon)
    origin_lat_rad = math.radians(origin_lat)
    lon_rad = rel_x / (scale * EARTH_RADIUS_M) + origin_lon_rad
    log_origin = math.log(math.tan(math.pi / 4 + origin_lat_rad / 2))
    lat_rad = 2.0 * math.atan(math.exp(rel_y / (scale * EARTH_RADIUS_M) + log_origin)) - math.pi / 2
    return math.degrees(lat_rad), math.degrees(lon_rad)

# ---------------------------------------------------------------------------
# OSM geometry rendering
# ---------------------------------------------------------------------------

def create_thick_lines(points, lines, color, radius=2.0):
    """Build Open3D triangle mesh for line segments (cylinders)."""
    meshes = []
    points = np.array(points, dtype=np.float64)
    for line in lines:
        start, end = points[line[0]], points[line[1]]
        vec = end - start
        length = np.linalg.norm(vec)
        if length < 1e-6:
            continue
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=8)
        cyl.paint_uniform_color(color)
        z_axis = np.array([0.0, 0.0, 1.0])
        direction = vec / length
        rot_axis = np.cross(z_axis, direction)
        rot_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
        if np.linalg.norm(rot_axis) > 1e-6:
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
            R_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * rot_angle)
            cyl.rotate(R_mat, center=[0, 0, 0])
        elif np.dot(z_axis, direction) < 0:
            R_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]) * np.pi)
            cyl.rotate(R_mat, center=[0, 0, 0])
        cyl.translate((start + end) / 2)
        meshes.append(cyl)
    if not meshes:
        return None
    out = meshes[0]
    for m in meshes[1:]:
        out += m
    return out


class OSMLoader:
    """Load .osm XML and convert ways to relative world coordinates via Web Mercator."""

    def __init__(self, xml_path, origin_latlon=None):
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        self.nodes = {}
        self.ways = []
        self.origin_lat = None
        self.origin_lon = None
        self._origin_latlon = origin_latlon
        self._parse()

    def _parse(self):
        bounds = self.root.find("bounds")
        if bounds is not None:
            if self._origin_latlon is not None:
                self.origin_lat, self.origin_lon = self._origin_latlon
            else:
                self.origin_lat = (float(bounds.get("minlat")) + float(bounds.get("maxlat"))) / 2.0
                self.origin_lon = (float(bounds.get("minlon")) + float(bounds.get("maxlon"))) / 2.0
        else:
            first_node = self.root.find("node")
            if first_node is not None:
                self.origin_lat = float(first_node.get("lat"))
                self.origin_lon = float(first_node.get("lon"))
            elif self._origin_latlon is not None:
                self.origin_lat, self.origin_lon = self._origin_latlon
            else:
                raise ValueError("No bounds or nodes in OSM and no origin given")
        for node in self.root.findall("node"):
            nid = node.get("id")
            lat, lon = float(node.get("lat")), float(node.get("lon"))
            x, y = latlon_to_mercator_relative(lat, lon, self.origin_lat, self.origin_lon)
            self.nodes[nid] = (x, y)
        for way in self.root.findall("way"):
            node_ids = [nd.get("ref") for nd in way.findall("nd")]
            tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
            if node_ids:
                self.ways.append({"nodes": node_ids, "tags": tags})

    def get_geometries(self, z_offset=0.0, thickness=2.0, buildings_only=True):
        """Return list of Open3D TriangleMesh for OSM ways."""
        batches = defaultdict(lambda: {"points": [], "lines": [], "color": OSM_COLORS["default"]})
        for way in self.ways:
            tags, node_ids = way["tags"], way["nodes"]
            if buildings_only and "building" not in tags:
                continue
            cat, color = "default", OSM_COLORS["default"]
            if "building" in tags:
                cat, color = "building", OSM_COLORS["building"]
            elif not buildings_only and "highway" in tags:
                cat, color = "highway", OSM_COLORS["highway"]
            elif not buildings_only and "landuse" in tags and tags.get("landuse") in (
                "grass", "meadow", "forest", "commercial", "residential",
            ):
                cat, color = "landuse", OSM_COLORS["landuse"]
            elif not buildings_only and "natural" in tags:
                cat, color = "natural", OSM_COLORS["natural"]
            elif not buildings_only and "barrier" in tags:
                cat, color = "barrier", OSM_COLORS["barrier"]
            elif not buildings_only and "amenity" in tags:
                cat, color = "amenity", OSM_COLORS["amenity"]
            points = []
            for nid in node_ids:
                if nid in self.nodes:
                    x, y = self.nodes[nid]
                    points.append([x, y, z_offset])
            if len(points) < 2:
                continue
            batch = batches[cat]
            batch["color"] = color
            start_idx = len(batch["points"])
            batch["points"].extend(points)
            n_pts = len(points)
            batch["lines"].extend([[start_idx + i, start_idx + i + 1] for i in range(n_pts - 1)])
            if cat in ("building", "landuse", "amenity", "natural") and n_pts > 2:
                batch["lines"].append([start_idx + n_pts - 1, start_idx])
        geometries = []
        for data in batches.values():
            if not data["points"]:
                continue
            mesh = create_thick_lines(data["points"], data["lines"], data["color"], radius=thickness / 2.0)
            if mesh is not None:
                geometries.append(mesh)
        return geometries


def create_path_geometry(points_xyz, color=(0.0, 0.8, 0.0), thickness=1.5):
    """Create Open3D geometry for a path (thick line strip). points_xyz: (N, 3)."""
    if points_xyz is None or len(points_xyz) < 2:
        return None
    lines = [[i, i + 1] for i in range(len(points_xyz) - 1)]
    return create_thick_lines(points_xyz.tolist(), lines, list(color), radius=thickness / 2.0)

# ---------------------------------------------------------------------------
# Accuracy / uncertainty analysis
# ---------------------------------------------------------------------------

def run_accuracy_analysis(result):
    """
    Run accuracy analysis on a multiclass result dict.

    Expects dict with keys: gt_labels, inf_labels, inf_variance, n_classes.
    Returns (correct_all, uncertainty_all, bin_edges, bin_acc_array).
      - correct_all: (N,) bool
      - uncertainty_all: (N,) float in [0, 1]
      - bin_edges: (n_bins+1,) uncertainty bin boundaries
      - bin_acc_array: (n_bins,) accuracy in [0, 1] per bin
    """
    from scipy.stats import rankdata, spearmanr
    import matplotlib.pyplot as plt
    from label_mappings import IGNORE_LABELS

    gt_labels = result["gt_labels"]
    inf_labels = result["inf_labels"]
    inf_variance = result["inf_variance"]
    n_classes = result["n_classes"]

    ignore_set = set(IGNORE_LABELS)
    valid = np.array([g not in ignore_set for g in gt_labels], dtype=bool)
    print(f"Ignoring {int(np.sum(~valid))} points with GT label in {IGNORE_LABELS}")

    correct_all = (inf_labels == gt_labels)
    correct = correct_all[valid]
    n_correct = int(np.sum(correct))
    n_total = len(correct)

    max_var = (n_classes - 1) / (n_classes ** 2) if n_classes > 1 else 1.0
    uncertainty_all = 1.0 - np.clip(inf_variance / max_var, 0.0, 1.0)
    uncertainty = uncertainty_all[valid]
    median_unc = float(np.median(uncertainty))

    # -- Baseline accuracy (all valid points) ------------------------------
    print(f"Accuracy (all): {n_correct}/{n_total} ({100.0 * n_correct / n_total:.2f}%)")
    print(f"Median uncertainty: {median_unc:.6f}")

    # -- Accuracy after removing above-median uncertainty (diagnostic) -----
    low_mask = uncertainty <= median_unc
    n_kept = int(np.sum(low_mask))
    n_correct_kept = int(np.sum(correct[low_mask]))
    accuracy_kept = 100.0 * n_correct_kept / n_kept if n_kept > 0 else 0.0
    print(f"Accuracy (median threshold): {n_correct_kept}/{n_kept} ({accuracy_kept:.2f}%)")

    # -- Uncertainty calibration -------------------------------------------
    incorrect = np.asarray(~correct, dtype=np.float64)
    rho, p_val = spearmanr(uncertainty, incorrect)
    print(f"\nSpearman(uncertainty, incorrect): rho={rho:.4f} (p={p_val:.2e})")

    n_incorrect = int(np.sum(incorrect))
    n_correct_count = n_total - n_incorrect
    if n_incorrect > 0 and n_correct_count > 0:
        ranks = rankdata(uncertainty)
        S = np.sum(ranks[~correct])
        auroc = (S - n_incorrect * (n_incorrect + 1) / 2) / (n_incorrect * n_correct_count)
        print(f"AUROC: {auroc:.4f}")

    n_bins = 10
    bin_edges = np.percentile(uncertainty, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-9
    bin_acc_array = np.ones(n_bins, dtype=np.float64)
    bin_centers = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    print(f"\nAccuracy by uncertainty bin:")
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        bin_centers[i] = (lo + hi) / 2
        b = (uncertainty >= lo) & (uncertainty < hi)
        n_b = int(np.sum(b))
        bin_counts[i] = n_b
        if n_b > 0:
            bin_acc_array[i] = float(np.sum(correct[b])) / n_b
            print(f"  bin {i+1:2d} [{lo:.3f}-{hi:.3f}]: {100.0 * bin_acc_array[i]:.1f}% (n={n_b})")
    non_empty = bin_counts > 0
    if np.any(non_empty):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(bin_centers[non_empty], bin_acc_array[non_empty] * 100.0,
                "o-", color="steelblue", linewidth=2, markersize=8)
        ax.set_xlabel("Uncertainty")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy by uncertainty bin")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show(block=True)
        plt.close(fig)

    var_valid = inf_variance[valid]
    avg_var = float(np.mean(var_valid))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(var_valid, bins=80, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(avg_var, color="coral", linewidth=2, label=f"Mean={avg_var:.4f}")
    ax.set_xlabel("Variance of class probabilities")
    ax.set_ylabel("Count")
    ax.set_title(f"Variance distribution (n={n_total})")
    ax.legend()
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)

    return correct_all, uncertainty_all, bin_edges, bin_acc_array


def filter_by_confusion_matrix(result, uncertainty_all):
    """
    Filter points using the confusion matrix.

    Builds a confusion matrix (predicted × true) over valid points.  For each
    predicted class, precision = P(correct | predicted as that class).  The
    precision is then used as a percentile threshold on uncertainty: for a
    class with 80% precision, keep the 80% most confident predictions (drop
    the 20% most uncertain).  High-precision classes keep nearly everything;
    low-precision classes are aggressively pruned.

    Points with ignored GT labels are always kept.
    Returns a boolean keep mask (True = keep).
    """
    from label_mappings import IGNORE_LABELS, COMMON_LABELS, N_COMMON

    gt_labels = result["gt_labels"]
    inf_labels = result["inf_labels"]

    ignore_set = set(IGNORE_LABELS)
    valid = np.array([g not in ignore_set for g in gt_labels], dtype=bool)
    correct = (inf_labels == gt_labels)

    # -- Build confusion matrix (rows = predicted, cols = true) ------------
    conf = np.zeros((N_COMMON, N_COMMON), dtype=np.int64)
    for pred, gt in zip(inf_labels[valid], gt_labels[valid]):
        if 0 <= pred < N_COMMON and 0 <= gt < N_COMMON:
            conf[pred, gt] += 1

    # -- Print confusion matrix --------------------------------------------
    present = sorted(set(np.unique(inf_labels[valid])) | set(np.unique(gt_labels[valid])))
    present = [c for c in present if c in COMMON_LABELS and c not in ignore_set]
    abbrev = {c: COMMON_LABELS[c][:7] for c in present}

    header = "pred\\true  " + "  ".join(f"{abbrev[c]:>7s}" for c in present)
    print(f"\nConfusion matrix (rows=predicted, cols=true):\n{header}")
    for p in present:
        row_vals = "  ".join(f"{conf[p, t]:7d}" for t in present)
        total_p = int(np.sum(conf[p, :]))
        prec = conf[p, p] / total_p * 100 if total_p > 0 else 0.0
        print(f"  {abbrev[p]:>7s}   {row_vals}   prec={prec:.1f}%")

    # -- Filter using precision as keep-percentile -------------------------
    keep = np.ones(len(uncertainty_all), dtype=bool)

    print(f"\nPrecision-based filtering:")
    for cls in sorted(np.unique(inf_labels)):
        cls_valid = (inf_labels == cls) & valid
        n_cls = int(np.sum(cls_valid))
        if n_cls == 0:
            continue

        cls_name = COMMON_LABELS.get(int(cls), f"class_{cls}")
        total_predicted = int(np.sum(conf[cls, :]))
        precision = conf[cls, cls] / total_predicted if total_predicted > 0 else 0.0

        if precision >= 1.0 or n_cls == 0:
            acc_cls = 100.0 * np.sum(correct[cls_valid]) / n_cls
            print(f"  {cls_name:15s}: prec=100.0%  kept {n_cls}/{n_cls}  acc {acc_cls:.1f}%")
            continue

        cls_unc = uncertainty_all[cls_valid]
        threshold = float(np.percentile(cls_unc, precision * 100))
        above = cls_valid & (uncertainty_all > threshold)
        keep[above] = False

        n_kept = int(np.sum(keep[cls_valid]))
        acc_before = 100.0 * np.sum(correct[cls_valid]) / n_cls
        acc_after = 100.0 * np.sum(correct[cls_valid & keep]) / n_kept if n_kept > 0 else 0.0
        print(f"  {cls_name:15s}: prec={100*precision:.1f}%  "
              f"kept {n_kept}/{n_cls}  acc {acc_before:.1f}% -> {acc_after:.1f}%")

    keep[~valid] = True

    n_total = len(keep)
    n_kept = int(np.sum(keep))
    print(f"Total: kept {n_kept}/{n_total} (dropped {n_total - n_kept})")

    kept_valid = keep & valid
    n_kv = int(np.sum(kept_valid))
    if n_kv > 0:
        n_correct = int(np.sum(correct[kept_valid]))
        print(f"Accuracy after filtering: {n_correct}/{n_kv} ({100.0 * n_correct / n_kv:.2f}%)")

    return keep
