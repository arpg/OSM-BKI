"""
Shared utilities for OSM-BKI Python scripts.

Contains:
- ConfigReader: unified config loading from YAML
- OSM visualization helpers (load_osm_geometries, OSMLoader, create_thick_lines)
- Data-loading helpers used across training, evaluation, and visualization scripts
  (load_body_to_lidar, load_poses_csv, transform_points_to_world, load_scan,
   load_labels, find_label_file, get_frame_number)
"""

import math
import os
from pathlib import Path

import yaml
import numpy as np
import open3d as o3d

try:
    import osm_bki_cpp
except ImportError:
    osm_bki_cpp = None


# ---------------------------------------------------------------------------
# ConfigReader
# ---------------------------------------------------------------------------

class ConfigReader:
    """
    Unified reader for OSM-BKI YAML config files.
    Loads the config once and exposes parsed values as properties.
    """

    def __init__(self, config_path: str):
        self.config_path = str(config_path)
        with open(config_path) as f:
            self._raw = yaml.safe_load(f) or {}
        self._config_dir = Path(config_path).resolve().parent

    @property
    def raw(self):
        """Raw YAML dict."""
        return self._raw

    # --- Label mapping (for BKI) ---

    def get_label_mapping(self):
        """Return (dense_to_raw, K) for BKI voxel label lookup."""
        labels = self._raw.get("labels") or {}
        raw_ids = sorted(int(k) for k in labels.keys())
        return raw_ids, len(raw_ids)

    @property
    def label_names(self):
        """{label_id: name} from labels block."""
        labels = self._raw.get("labels") or {}
        return {int(k): str(v) for k, v in labels.items()}

    # --- Colors ---

    @property
    def colors_by_label(self):
        """{label_id: [r,g,b]} from colors block, normalised to [0, 1]."""
        colors = {}
        if "colors" in self._raw:
            for lbl_str, rgb in self._raw["colors"].items():
                colors[int(lbl_str)] = [c / 255.0 for c in rgb]
        return colors

    # --- OSM ---

    @property
    def osm_origin_lat(self):
        if self._raw.get("osm_origin_lat") is not None and self._raw.get("osm_origin_lon") is not None:
            return float(self._raw["osm_origin_lat"])
        if "init_latlon_day_06" in self._raw:
            return float(self._raw["init_latlon_day_06"][0])
        return None

    @property
    def osm_origin_lon(self):
        if self._raw.get("osm_origin_lat") is not None and self._raw.get("osm_origin_lon") is not None:
            return float(self._raw["osm_origin_lon"])
        if "init_latlon_day_06" in self._raw:
            return float(self._raw["init_latlon_day_06"][1])
        return None

    @property
    def osm_world_offset_x(self):
        return float(self._raw.get("osm_world_offset_x", 0.0))

    @property
    def osm_world_offset_y(self):
        return float(self._raw.get("osm_world_offset_y", 0.0))

    @property
    def osm_path(self):
        """Resolved path to OSM file, or None if not configured/missing.
        Resolves osm_file relative to the config file's directory."""
        if "osm_file" not in self._raw:
            return None
        path = (self._config_dir / self._raw["osm_file"]).resolve()
        return str(path) if path.exists() else None

    @property
    def skip_frames(self):
        return int(self._raw.get("skip_frames", 0))

    # --- Pose / world frame ---

    @property
    def init_rel_pos(self):
        """[x,y,z] from init_rel_pos_day_06, or None."""
        if "init_rel_pos_day_06" not in self._raw:
            return None
        rp = self._raw["init_rel_pos_day_06"]
        return np.array([float(rp[0]), float(rp[1]), float(rp[2])])

    def get_visualize_config(self):
        """Dict for bki_tools visualize: osm_path, osm_origin_*, osm_world_offset_*."""
        return {
            "osm_path": self.osm_path,
            "osm_origin_lat": self.osm_origin_lat,
            "osm_origin_lon": self.osm_origin_lon,
            "osm_world_offset_x": self.osm_world_offset_x,
            "osm_world_offset_y": self.osm_world_offset_y,
        }

    def get_non_path_config(self, overrides=None):
        """
        Dict of config values that are not file paths (for scripts that require paths via CLI).
        Returns: skip_frames, init_rel_pos, osm_origin_*, use_* flags.
        overrides: optional dict to override (e.g. from CLI).
        """
        cfg = {
            "skip_frames": self.skip_frames,
            "use_osm_origin": self.osm_origin_lat is not None,
            "osm_origin_lat": self.osm_origin_lat or 0.0,
            "osm_origin_lon": self.osm_origin_lon or 0.0,
            "use_init_rel_pos": self.init_rel_pos is not None,
            "init_rel_pos": self.init_rel_pos if self.init_rel_pos is not None else np.zeros(3),
        }
        if overrides:
            cfg.update({k: v for k, v in overrides.items() if v is not None})
        return cfg


# ---------------------------------------------------------------------------
# OSM visualisation helpers
# ---------------------------------------------------------------------------

def create_thick_lines(points, lines, color, radius=5.0):
    """Cylinder-mesh thick lines (slow on large maps; only used with --thick)."""
    meshes = []
    pts = np.array(points)
    for line in lines:
        start = pts[line[0]]
        end = pts[line[1]]
        vec = end - start
        length = np.linalg.norm(vec)
        if length < 0.01:
            continue
        cyl = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=length, resolution=8)
        cyl.paint_uniform_color(color)
        z_axis = np.array([0.0, 0.0, 1.0])
        direction = vec / length
        rot_axis = np.cross(z_axis, direction)
        rot_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
        if np.linalg.norm(rot_axis) > 0.001:
            cyl.rotate(
                o3d.geometry.get_rotation_matrix_from_axis_angle(
                    rot_axis / np.linalg.norm(rot_axis) * rot_angle),
                center=[0, 0, 0])
        elif np.dot(z_axis, direction) < 0:
            cyl.rotate(
                o3d.geometry.get_rotation_matrix_from_axis_angle(
                    np.array([1.0, 0.0, 0.0]) * math.pi),
                center=[0, 0, 0])
        cyl.translate((start + end) / 2)
        meshes.append(cyl)
    if not meshes:
        return None
    combined = meshes[0]
    for m in meshes[1:]:
        combined += m
    return combined


def get_osm_geometries(osm_path, config_path, z_offset=0.05,
                       use_thick=False, thickness=10.0):
    """
    Load OSM via C++ and return Open3D geometries.
    Uses osm_bki_cpp.load_osm_geometries (config supplies origin, etc.).
    """
    if osm_bki_cpp is None:
        raise ImportError(
            "osm_bki_cpp not available. Build the C++ extension to use OSM loader.")
    if not Path(osm_path).exists():
        raise FileNotFoundError(f"OSM file not found: {osm_path}")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    data = osm_bki_cpp.load_osm_geometries(osm_path, config_path, z_offset)
    geoms = []
    for d in data:
        pts = np.array(d["points"], dtype=np.float64)
        lines = np.array(d["lines"], dtype=np.int32)
        color = d["color"]
        if use_thick:
            mesh = create_thick_lines(pts, lines, color, radius=thickness / 2.0)
            if mesh is not None:
                geoms.append(mesh)
        else:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(pts)
            ls.lines = o3d.utility.Vector2iVector(lines)
            ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
            geoms.append(ls)
    return geoms


class OSMLoader:
    """Thin wrapper for OSM loading via C++. Config supplies origin and projection."""

    def __init__(self, osm_path: str, config_path: str,
                 origin_lat_override=None,
                 origin_lon_override=None,
                 world_offset_x: float = 0.0,
                 world_offset_y: float = 0.0):
        self.osm_path = osm_path
        self.config_path = config_path

    def get_geometries(self, z_offset: float = 0.05,
                       thickness: float = 10.0,
                       use_thick: bool = False):
        """Return Open3D geometry objects for all OSM ways."""
        return get_osm_geometries(
            self.osm_path, self.config_path,
            z_offset=z_offset, use_thick=use_thick, thickness=thickness)


# ---------------------------------------------------------------------------
# Data-loading helpers (shared by train, eval, and visualisation scripts)
# ---------------------------------------------------------------------------

def load_body_to_lidar(calib_yaml_path: str) -> np.ndarray:
    """Read body/os_sensor/T from hhs_calib.yaml -> 4x4 float64 matrix."""
    with open(calib_yaml_path) as f:
        calib = yaml.safe_load(f)
    rows = calib['body']['os_sensor']['T']
    mat = np.array(rows, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(
            f"Expected 4x4 matrix in {calib_yaml_path}, got {mat.shape}")
    return mat


def load_poses_csv(pose_csv_path: str) -> dict:
    """
    Read pose CSV and return {scan_id: np.array([x,y,z,qx,qy,qz,qw])}.

    Expected columns (by position): num, t, x, y, z, qx, qy, qz[, qw].
    Handles both comma-separated and whitespace-separated files.
    Header rows and non-numeric tokens are silently skipped.
    qw defaults to 1.0 if the column is absent.
    """
    poses: dict = {}
    with open(pose_csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            sep = ',' if ',' in line else None
            vals = []
            for tok in (line.split(sep) if sep else line.split()):
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
            if len(vals) < 8:
                continue
            scan_id = int(vals[0])
            qw = vals[8] if len(vals) > 8 else 1.0
            poses[scan_id] = np.array(
                [vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], qw],
                dtype=np.float64)
    return poses


def transform_points_to_world(points, pose, body_to_lidar, init_rel_pos=None):
    """Delegate to C++ transform_scan_to_world (pose_utils.hpp)."""
    if osm_bki_cpp is None:
        raise ImportError(
            "osm_bki_cpp not available. Build the C++ extension.")
    return osm_bki_cpp.transform_scan_to_world(
        np.ascontiguousarray(points, dtype=np.float32),
        np.asarray(pose, dtype=np.float64),
        np.ascontiguousarray(body_to_lidar, dtype=np.float64),
        np.asarray(init_rel_pos, dtype=np.float64) if init_rel_pos is not None else None,
    )


def load_poses_mat4(pose_path: str) -> dict:
    """
    Load KITTI-360 style velodyne_poses.txt.

    Each non-comment line contains a frame index followed by 12 values (row-major
    3×4 LiDAR-to-world matrix) or 16 values (4×4).  Example::

        0 r00 r01 r02 tx  r10 r11 r12 ty  r20 r21 r22 tz

    Returns {frame_id: np.ndarray(4, 4)} where the last row is padded to
    [0, 0, 0, 1] for 3×4 inputs.
    """
    poses: dict = {}
    with open(pose_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            sep = ',' if ',' in line else None
            vals = []
            for tok in (line.split(sep) if sep else line.split()):
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
            if len(vals) == 13:
                # frame_index + 12 values -> 3x4 matrix
                frame_id = int(vals[0])
                mat = np.array(vals[1:13], dtype=np.float64).reshape(3, 4)
                T = np.eye(4, dtype=np.float64)
                T[:3, :] = mat
                poses[frame_id] = T
            elif len(vals) == 17:
                # frame_index + 16 values -> 4x4 matrix
                frame_id = int(vals[0])
                poses[frame_id] = np.array(vals[1:17], dtype=np.float64).reshape(4, 4)
            else:
                continue
    return poses


def transform_points_to_world_mat4(points, pose_mat4, init_rel_pos=None):
    """Delegate to C++ transform_scan_to_world_mat4 (pose_utils.hpp)."""
    if osm_bki_cpp is None:
        raise ImportError(
            "osm_bki_cpp not available. Build the C++ extension.")
    return osm_bki_cpp.transform_scan_to_world_mat4(
        np.ascontiguousarray(points, dtype=np.float32),
        np.ascontiguousarray(pose_mat4, dtype=np.float64),
        np.asarray(init_rel_pos, dtype=np.float64) if init_rel_pos is not None else None,
    )


def load_scan(bin_path):
    """Load point cloud (N,4) float32 and return (N,3) xyz, (N,) intensity."""
    scan = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return scan[:, :3], scan[:, 3]


def load_labels(label_path):
    """Load semantic labels; lower 16 bits (matches build_voxel_map convention)."""
    raw = np.fromfile(label_path, dtype=np.uint32)
    return (raw & 0xFFFF).astype(np.uint32)


def find_label_file(label_dir, scan_stem):
    """Find label file for a scan; try .label, .bin, _prediction.label, _prediction.bin."""
    exts = [".label", ".bin", "_prediction.label", "_prediction.bin"]
    for ext in exts:
        p = Path(label_dir) / f"{scan_stem}{ext}"
        if p.exists():
            return str(p)
    return None


def get_frame_number(stem):
    """Parse an integer frame number from a filename stem, or None."""
    try:
        return int(stem)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Label colour helpers
# ---------------------------------------------------------------------------

def load_label_colors(config_path):
    """
    Load {label_id: [r,g,b]} from config. Convenience wrapper around ConfigReader.
    Prefer ConfigReader(config_path).colors_by_label when you have a ConfigReader.
    """
    return ConfigReader(config_path).colors_by_label


def _hash_color(label: int) -> list:
    """Deterministic fallback colour for labels missing from the config."""
    h = (label * 2654435761) & 0xFFFFFFFF
    return [((h >> 16) & 0xFF) / 255.0,
            ((h >>  8) & 0xFF) / 255.0,
            ( h        & 0xFF) / 255.0]


def map_labels_to_colors(labels, colors_by_label):
    """
    Map (N,) uint32 labels -> (N, 3) RGB float64.
    Uses *colors_by_label* dict for known labels and a deterministic hash
    fallback for anything not in the map.
    """
    colors = np.zeros((len(labels), 3))
    for lbl in np.unique(labels):
        mask = labels == lbl
        lbl_int = int(lbl)
        if lbl_int in colors_by_label:
            colors[mask] = colors_by_label[lbl_int]
        else:
            colors[mask] = _hash_color(lbl_int)
    return colors
