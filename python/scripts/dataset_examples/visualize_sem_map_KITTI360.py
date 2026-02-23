#!/usr/bin/env python3
"""
Visualize a semantic map from a KITTI-360 sequence using Open3D.

Loads LiDAR scans and semantic labels, transforms to world frame using poses,
and displays the accumulated point cloud colored by semantic class.
Optionally overlays OSM building/road outlines and the drive path.

    Usage:
        python visualize_sem_map_KITTI360.py
        python visualize_sem_map_KITTI360.py --sequence 2 --scan-skip 5 --downsample 0.2
        python visualize_sem_map_KITTI360.py --with-osm --with-path
        python visualize_sem_map_KITTI360.py --with-osm --osm-all --osm-thickness 3
"""

import argparse
import glob
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import open3d as o3d

# Default paths
KITTI360_ROOT = "./example_data/kitti360"
DEFAULT_SEQUENCE = 9  # first sequence (2013_05_28_drive_0000_sync)

# KITTI-360 Karlsruhe origin for lat/lon -> meters (same as Lidar2OSM)
ORIGIN_LATLON = (48.9843445, 8.4295857)

# Label set options: pick via --labels (e.g. --labels kitti360)
Label = __import__("collections").namedtuple("Label", ["name", "id", "color"])

# KITTI-360 labels 0-46 from kitti360_labels.yaml (colors in RGB)
_KITTI360_NAMES = {
    0: "unlabeled", 1: "ego vehicle", 2: "rectification border", 3: "out of roi",
    4: "static", 5: "dynamic", 6: "ground", 7: "road", 8: "sidewalk", 9: "parking",
    10: "rail track", 11: "building", 12: "wall", 13: "fence", 14: "guard rail",
    15: "bridge", 16: "tunnel", 17: "pole", 18: "polegroup", 19: "traffic light",
    20: "traffic sign", 21: "vegetation", 22: "terrain", 23: "sky", 24: "person",
    25: "rider", 26: "car", 27: "truck", 28: "bus", 29: "caravan", 30: "trailer",
    31: "train", 32: "motorcycle", 33: "bicycle", 34: "garage", 35: "gate",
    36: "stop", 37: "smallpole", 38: "lamp", 39: "trash bin", 40: "vending machine",
    41: "box", 42: "unknown construction", 43: "unknown vehicle", 44: "unknown object",
    45: "OSM Building", 46: "OSM Road",
}
# RGB [R, G, B] per label (converted from yaml BGR)
_KITTI360_COLORS_RGB = [
    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 74, 111], [81, 0, 81],
    [128, 64, 128], [232, 35, 244], [160, 170, 250], [140, 150, 230], [0, 100, 0],
    [156, 102, 102], [153, 153, 190], [180, 165, 180], [100, 100, 150], [90, 120, 150],
    [153, 153, 153], [153, 153, 153], [30, 170, 250], [0, 220, 220], [35, 142, 107],
    [152, 251, 152], [180, 130, 70], [60, 20, 220], [0, 0, 255], [142, 0, 0], [70, 0, 0],
    [100, 60, 0], [90, 0, 0], [110, 0, 0], [100, 80, 0], [230, 0, 0], [32, 11, 119],
    [128, 128, 64], [153, 153, 190], [90, 120, 150], [153, 153, 153], [64, 64, 0],
    [192, 128, 0], [0, 64, 128], [128, 64, 64], [0, 0, 102], [51, 0, 51], [32, 32, 32],
    [255, 255, 255], [255, 0, 0],
]
KITTI360_LABELS = [
    Label(_KITTI360_NAMES[i], i, tuple(_KITTI360_COLORS_RGB[i]))
    for i in range(47)
]

# Semantic KITTI-style sparse IDs (subset used in some pipelines)
SEMKITTI_LABELS = [
    Label("unlabeled", 0, (0, 0, 0)),
    Label("outlier", 1, (0, 0, 0)),
    Label("car", 10, (0, 0, 142)),
    Label("bicycle", 11, (119, 11, 32)),
    Label("bus", 13, (250, 80, 100)),
    Label("motorcycle", 15, (0, 0, 230)),
    Label("on-rails", 16, (255, 0, 0)),
    Label("truck", 18, (0, 0, 70)),
    Label("other-vehicle", 20, (51, 0, 51)),
    Label("person", 30, (220, 20, 60)),
    Label("bicyclist", 31, (200, 40, 255)),
    Label("motorcyclist", 32, (90, 30, 150)),
    Label("road", 40, (128, 64, 128)),
    Label("parking", 44, (250, 170, 160)),
    Label("sidewalk", 48, (244, 35, 232)),
    Label("other-ground", 49, (81, 0, 81)),
    Label("building", 50, (0, 100, 0)),
    Label("fence", 51, (190, 153, 153)),
    Label("other-structure", 52, (0, 150, 255)),
    Label("lane-marking", 60, (170, 255, 150)),
    Label("vegetation", 70, (107, 142, 35)),
    Label("trunk", 71, (0, 60, 135)),
    Label("terrain", 72, (152, 251, 152)),
    Label("pole", 80, (153, 153, 153)),
    Label("traffic-sign", 81, (0, 0, 255)),
    Label("other-object", 99, (255, 255, 50)),
]

# Dict of label-set options for --labels
LABELS_OPTIONS = {
    "kitti360": KITTI360_LABELS,
    "semkitti": SEMKITTI_LABELS,
}


def get_label_id_to_color(labels_list):
    """Build id -> RGB [0,1] map from a list of Label(name, id, color)."""
    return {lb.id: np.array(lb.color, dtype=np.float64) / 255.0 for lb in labels_list}


# Default: use KITTI360_LABELS (overridden in main() when --labels is set)
LABEL_ID_TO_COLOR = get_label_id_to_color(KITTI360_LABELS)
# Default color for unknown label IDs (gray)
UNKNOWN_COLOR = np.array([0.5, 0.5, 0.5], dtype=np.float64)

# ---------------------------------------------------------------------------
# OSM overlay constants
# ---------------------------------------------------------------------------

def read_poses(poses_path):
    """
    Read KITTI-360 pose file: frame_index followed by 16 floats (4x4) or 12 floats (3x4).
    Skips empty lines and lines that don't parse (e.g. header). Returns dict[frame_index -> 4x4 numpy array].
    """
    poses = {}
    with open(poses_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 13:
                continue
            try:
                frame_index = int(parts[0])
                values = np.array(parts[1:], dtype=float)
            except (ValueError, TypeError):
                continue
            if len(values) == 16:
                poses[frame_index] = values.reshape(4, 4)
            elif len(values) == 12:
                m = values.reshape(3, 4)
                pose_4x4 = np.eye(4)
                pose_4x4[:3, :] = m
                poses[frame_index] = pose_4x4
            # else: skip line (wrong length)
    return poses


def get_velodyne_poses(kitti360_root, sequence_dir):
    """
    Load Velodyne poses; if velodyne_poses.txt missing, derive from poses.txt (IMU) + calibration.
    Tries new layout first (root/<seq>/velodyne_poses.txt or root/<seq>/poses/), then
    old layout (root/data_poses/<seq>/).
    """
    # New layout: poses inside sequence dir
    seq_dir_full = os.path.join(kitti360_root, sequence_dir)
    for poses_dir in (seq_dir_full, os.path.join(seq_dir_full, "poses")):
        velodyne_file = os.path.join(poses_dir, "velodyne_poses.txt")
        imu_file = os.path.join(poses_dir, "poses.txt")
        if os.path.exists(velodyne_file):
            poses = read_poses(velodyne_file)
            if not poses:
                raise ValueError(
                    f"velodyne_poses.txt at {velodyne_file} is empty or has no valid lines "
                    "(expected: frame_id then 16 or 12 floats per line). "
                    "If you trimmed scans with keep_lidar_limit.py, ensure some poses were kept."
                )
            return poses
        if os.path.exists(imu_file):
            translation = np.array([0.81, 0.32, -0.83])
            rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            imu_to_lidar = np.eye(4)
            imu_to_lidar[:3, :3] = rotation
            imu_to_lidar[:3, 3] = translation
            imu_poses = read_poses(imu_file)
            if imu_poses:
                return {idx: imu_T @ imu_to_lidar for idx, imu_T in imu_poses.items()}
    # Old layout: data_poses/<seq>/
    data_poses_dir = os.path.join(kitti360_root, "data_poses", sequence_dir)
    velodyne_file = os.path.join(data_poses_dir, "velodyne_poses.txt")
    imu_file = os.path.join(data_poses_dir, "poses.txt")
    if os.path.exists(velodyne_file):
        poses = read_poses(velodyne_file)
        if not poses:
            raise ValueError(
                f"velodyne_poses.txt at {velodyne_file} is empty or has no valid lines."
            )
        return poses
    if not os.path.exists(imu_file):
        raise FileNotFoundError(
            f"No poses found. Tried: {seq_dir_full}, {os.path.join(seq_dir_full, 'poses')}, {data_poses_dir}"
        )
    translation = np.array([0.81, 0.32, -0.83])
    rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    imu_to_lidar = np.eye(4)
    imu_to_lidar[:3, :3] = rotation
    imu_to_lidar[:3, 3] = translation
    imu_poses = read_poses(imu_file)
    return {idx: imu_T @ imu_to_lidar for idx, imu_T in imu_poses.items()}


def transform_points_to_world(points_xyz, pose_4x4):
    """Transform (N, 3) points from LiDAR frame to world using 4x4 pose (lidar-to-world)."""
    ones = np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)
    xyz_h = np.hstack([points_xyz, ones])  # (N, 4)
    return (xyz_h @ pose_4x4.T)[:, :3]


def labels_to_colors(label_ids):
    """Map (N,) label IDs to (N, 3) RGB in [0, 1]."""
    colors = np.zeros((len(label_ids), 3), dtype=np.float64)
    for i, lid in enumerate(label_ids):
        lid = int(lid)
        if lid in LABEL_ID_TO_COLOR:
            colors[i] = LABEL_ID_TO_COLOR[lid]
        else:
            colors[i] = UNKNOWN_COLOR
    return colors


def get_sequence_paths(kitti360_root, sequence_index):
    """
    Return paths for a given sequence index (e.g. 0 -> 2013_05_28_drive_0000_sync).
    Supports:
    - New layout: root/<seq>/velodyne_points/data, root/<seq>/gt_labels or root/<seq>/inferred_labels/cenet_semkitti
    - Old layout: root/data_3d_raw/<seq>/velodyne_points/data, root/data_3d_semantics/<seq>/gt_labels (or orig_labels/labels)
    """
    seq_dir = f"2013_05_28_drive_{sequence_index:04d}_sync"
    # New layout: sequences directly under root
    raw_pc_new = os.path.join(kitti360_root, seq_dir, "velodyne_points", "data")
    gt_labels_new = os.path.join(kitti360_root, seq_dir, "gt_labels")
    inferred_new = os.path.join(kitti360_root, seq_dir, "inferred_labels", "cenet_kitti360")
    if os.path.isdir(raw_pc_new):
        if os.path.isdir(gt_labels_new):
            label_dir = gt_labels_new
        elif os.path.isdir(inferred_new):
            label_dir = inferred_new
        else:
            label_dir = gt_labels_new
        return {
            "sequence_dir": seq_dir,
            "raw_pc_dir": raw_pc_new,
            "label_dir": label_dir,
        }
    # Old layout: data_3d_raw, data_3d_semantics
    raw_pc_dir = os.path.join(kitti360_root, "data_3d_raw", seq_dir, "velodyne_points", "data")
    semantics_dir = os.path.join(kitti360_root, "data_3d_semantics", seq_dir)
    for sub in ("gt_labels", "labels", "orig_labels"):
        candidate = os.path.join(semantics_dir, sub)
        if os.path.isdir(candidate):
            return {
                "sequence_dir": seq_dir,
                "raw_pc_dir": raw_pc_dir,
                "label_dir": candidate,
            }
    label_dir = os.path.join(semantics_dir, "gt_labels")
    return {
        "sequence_dir": seq_dir,
        "raw_pc_dir": raw_pc_dir,
        "label_dir": label_dir,
    }


def find_frame_range(label_dir):
    """Return (min_frame, max_frame) from existing .bin files in label_dir."""
    pattern = os.path.join(label_dir, "*.bin")
    files = glob.glob(pattern)
    if not files:
        return None, None
    numbers = []
    for p in files:
        base = os.path.basename(p)
        try:
            numbers.append(int(os.path.splitext(base)[0]))
        except ValueError:
            continue
    if not numbers:
        return None, None
    return min(numbers), max(numbers)


def build_semantic_map(
    kitti360_root,
    sequence_index=0,
    scan_skip=1,
    voxel_size=0.15,
    max_range=None,
    max_points=None,
):
    """
    Load scans and labels for one sequence, transform to world, and return (points, colors).
    scan_skip: use every Nth scan (1 = all scans). voxel_size: voxel size in m for downsampling (0 = disable).
    If max_points is set, stop once total points reach that limit (truncating the last scan if needed).
    """
    paths = get_sequence_paths(kitti360_root, sequence_index)
    seq_dir = paths["sequence_dir"]
    raw_pc_dir = paths["raw_pc_dir"]
    label_dir = paths["label_dir"]

    print(f"Label directory: {label_dir}")
    if not os.path.isdir(raw_pc_dir):
        raise FileNotFoundError(f"Raw LiDAR directory not found: {raw_pc_dir}")
    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"Semantic labels directory not found: {label_dir}")

    velodyne_poses = get_velodyne_poses(kitti360_root, seq_dir)
    if not velodyne_poses:
        raise ValueError("No poses loaded.")

    min_f, max_f = find_frame_range(label_dir)
    if min_f is None:
        raise ValueError(f"No label .bin files in {label_dir}")

    frame_indices = sorted(velodyne_poses.keys())
    frame_indices = [f for f in frame_indices if min_f <= f <= max_f]
    if scan_skip > 1:
        frame_indices = frame_indices[::scan_skip]

    all_xyz = []
    all_colors = []

    for frame_id in frame_indices:
        pose = velodyne_poses.get(frame_id)
        if pose is None:
            continue

        pc_path = os.path.join(raw_pc_dir, f"{frame_id:010d}.bin")
        label_path = os.path.join(label_dir, f"{frame_id:010d}.bin")
        if not os.path.isfile(pc_path) or not os.path.isfile(label_path):
            continue

        try:
            pc = read_bin_file(pc_path, dtype=np.float32, shape=(-1, 4))
            xyz = pc[:, :3]
            # KITTI-360 semantic labels are stored as int16
            labels = read_bin_file(label_path, dtype=np.uint32, shape=(-1)).astype(np.int32)
        except Exception as e:
            print(f"  Skip frame {frame_id}: {e}", file=sys.stderr)
            continue

        if len(labels) != len(xyz):
            # Align: truncate labels or pad with 0 (unlabeled) so we can still visualize
            if len(labels) > len(xyz):
                labels = labels[: len(xyz)]
            else:
                labels = np.resize(labels, len(xyz))
        world_xyz = transform_points_to_world(xyz, pose)

        if max_range is not None and max_range > 0:
            dist = np.linalg.norm(world_xyz - pose[:3, 3], axis=1)
            mask = dist <= max_range
            world_xyz = world_xyz[mask]
            labels = labels[mask]

        colors = labels_to_colors(labels)

        if voxel_size is not None and voxel_size > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(world_xyz.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            world_xyz = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)

        current_total = sum(len(x) for x in all_xyz)
        if max_points is not None and current_total >= max_points:
            break
        if max_points is not None and current_total + len(world_xyz) > max_points:
            need = max_points - current_total
            world_xyz = world_xyz[:need]
            colors = colors[:need]
        all_xyz.append(world_xyz)
        all_colors.append(colors)

    if not all_xyz:
        return None, None

    points = np.vstack(all_xyz)
    colors = np.vstack(all_colors)
    return points, colors


# ---------------------------------------------------------------------------
# OSM overlay helpers
# ---------------------------------------------------------------------------

def lat_to_scale(lat_deg):
    """Mercator scale factor at given latitude (cos(lat))."""
    return math.cos(math.radians(lat_deg))


def latlon_to_mercator_absolute(lat_deg, lon_deg, scale):
    """Convert lat/lon to Web Mercator (EPSG:3857) x,y in meters."""
    lon_rad = math.radians(lon_deg)
    lat_rad = math.radians(lat_deg)
    x = scale * lon_rad * EARTH_RADIUS_M
    y = scale * EARTH_RADIUS_M * math.log(math.tan(math.pi / 4 + lat_rad / 2))
    return x, y


def latlon_to_mercator_relative(lat_deg, lon_deg, origin_lat, origin_lon):
    """
    Convert lat/lon to relative world coordinates (meters) using Web Mercator.
    Origin is at (origin_lat, origin_lon); returns (x, y) relative to that point.
    """
    scale = lat_to_scale(origin_lat)
    ox, oy = latlon_to_mercator_absolute(origin_lat, origin_lon, scale)
    mx, my = latlon_to_mercator_absolute(lat_deg, lon_deg, scale)
    return mx - ox, my - oy


def create_thick_lines(points, lines, color, radius=2.0):
    """Build Open3D triangle mesh for line segments (cylinders)."""
    meshes = []
    points = np.array(points, dtype=np.float64)
    for line in lines:
        start = points[line[0]]
        end = points[line[1]]
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
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * rot_angle)
            cyl.rotate(R, center=[0, 0, 0])
        elif np.dot(z_axis, direction) < 0:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]) * np.pi)
            cyl.rotate(R, center=[0, 0, 0])
        midpoint = (start + end) / 2
        cyl.translate(midpoint)
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
            minlat = float(bounds.get("minlat"))
            minlon = float(bounds.get("minlon"))
            maxlat = float(bounds.get("maxlat"))
            maxlon = float(bounds.get("maxlon"))
            if self._origin_latlon is not None:
                self.origin_lat, self.origin_lon = self._origin_latlon
            else:
                self.origin_lat = (minlat + maxlat) / 2.0
                self.origin_lon = (minlon + maxlon) / 2.0
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
            lat = float(node.get("lat"))
            lon = float(node.get("lon"))
            x, y = latlon_to_mercator_relative(lat, lon, self.origin_lat, self.origin_lon)
            self.nodes[nid] = (x, y)
        for way in self.root.findall("way"):
            node_ids = [nd.get("ref") for nd in way.findall("nd")]
            tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
            if node_ids:
                self.ways.append({"nodes": node_ids, "tags": tags})

    def get_geometries(self, z_offset=0.0, thickness=2.0, buildings_only=True):
        """Return list of Open3D TriangleMesh. If buildings_only, only building ways are included."""
        batches = defaultdict(lambda: {"points": [], "lines": [], "color": OSM_COLORS["default"]})
        for way in self.ways:
            tags = way["tags"]
            node_ids = way["nodes"]
            if buildings_only and "building" not in tags:
                continue
            cat = "default"
            color = OSM_COLORS["default"]
            if "building" in tags:
                cat, color = "building", OSM_COLORS["building"]
            elif not buildings_only and "highway" in tags:
                cat, color = "highway", OSM_COLORS["highway"]
            elif not buildings_only and "landuse" in tags and tags.get("landuse") in ("grass", "meadow", "forest", "commercial", "residential"):
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


def get_osm_path(kitti360_root, sequence_index):
    """
    Path to OSM file. Tries new layout first (root/kitti360.osm), then old layout
    (data_osm/kitti360.osm or data_osm/map_<seq>.osm).
    """
    root_kitti360 = os.path.join(kitti360_root, "kitti360.osm")
    if os.path.isfile(root_kitti360):
        return root_kitti360
    seq_dir = f"2013_05_28_drive_{sequence_index:04d}_sync"
    seq_osm = os.path.join(kitti360_root, seq_dir, f"map_{sequence_index:04d}.osm")
    if os.path.isfile(seq_osm):
        return seq_osm
    return root_kitti360


def get_path_from_velodyne_poses(kitti360_root, sequence_index):
    """
    Build path from velodyne_poses.txt in world frame, shifted so the first pose is at (0, 0, 0).
    Returns (path_xyz (N,3), first_pose_xyz (3,)) or (None, None) if no poses.
    """
    seq_dir = f"2013_05_28_drive_{sequence_index:04d}_sync"
    try:
        poses = get_velodyne_poses(kitti360_root, seq_dir)
    except (FileNotFoundError, OSError):
        return None, None
    if not poses:
        return None, None
    frame_ids = sorted(poses.keys())
    positions = np.array([poses[fid][:3, 3] for fid in frame_ids], dtype=np.float64)
    first_pose = positions[0].copy()
    path_xyz = positions - first_pose
    return path_xyz, first_pose


def create_path_geometry(points_xyz, color=(0.0, 0.8, 0.0), thickness=1.5):
    """Create Open3D geometry for the path (thick line strip). points_xyz: (N, 3)."""
    if points_xyz is None or len(points_xyz) < 2:
        return None
    lines = [[i, i + 1] for i in range(len(points_xyz) - 1)]
    return create_thick_lines(points_xyz.tolist(), lines, list(color), radius=thickness / 2.0)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize KITTI-360 semantic map with Open3D (first sequence by default)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=KITTI360_ROOT,
        help=f"KITTI360 dataset root (default: {KITTI360_ROOT})",
    )
    parser.add_argument(
        "--sequence",
        type=int,
        default=DEFAULT_SEQUENCE,
        help=f"Sequence index, e.g. 0 -> 2013_05_28_drive_0000_sync (default: {DEFAULT_SEQUENCE})",
    )
    parser.add_argument(
        "--scan-skip",
        type=int,
        default=1,
        help="Use every Nth scan (1=all, 2=every other, etc.) (default: 1)",
    )
    parser.add_argument(
        "--downsample",
        type=float,
        default=1.0,
        help="Voxel size in meters for voxel downsampling (default: 0.15, use 0 to disable)",
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=None,
        help="Max distance from sensor per scan in meters (default: none)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Stop accumulating once total points reach this limit (default: none)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save point cloud as .ply",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="kitti360",
        choices=list(LABELS_OPTIONS.keys()),
        help="Label set to use for coloring (default: kitti360). Options: %(choices)s",
    )
    # OSM overlay arguments
    parser.add_argument("--with-osm", action="store_true", help="Overlay OSM map on the semantic map")
    parser.add_argument("--osm-thickness", type=float, default=2.0, help="OSM line thickness in meters (default: 2.0)")
    parser.add_argument("--osm-all", action="store_true", help="Show all OSM features; default is buildings only")
    parser.add_argument("--with-path", action="store_true", help="Show drive path from velodyne_poses.txt")
    parser.add_argument("--path-thickness", type=float, default=1.5, help="Path line thickness in meters (default: 1.5)")
    args = parser.parse_args()

    voxel_size = args.downsample if args.downsample > 0 else None

    # Use selected label set for id -> color mapping
    global LABEL_ID_TO_COLOR
    LABEL_ID_TO_COLOR = get_label_id_to_color(LABELS_OPTIONS[args.labels])

    print(f"KITTI360 root: {args.root}")
    print(f"Sequence index: {args.sequence}")

    geoms = []
    first_pose_for_osm = None

    # -- Velodyne path (also needed to align OSM) -------------------------
    path_xyz, first_pose_for_osm = get_path_from_velodyne_poses(args.root, args.sequence)
    path_z_default = 0.0
    if path_xyz is not None and len(path_xyz) >= 2:
        path_z_default = float(np.median(path_xyz[:, 2]))
        if args.with_path:
            path_geom = create_path_geometry(path_xyz, color=(0.0, 0.8, 0.0), thickness=args.path_thickness)
            if path_geom is not None:
                geoms.append(path_geom)
                print(f"Path: {len(path_xyz)} poses from velodyne_poses.txt")
    else:
        if args.with_path or args.with_osm:
            print("Warning: no velodyne poses or too few frames for path/OSM alignment.", file=sys.stderr)

    # -- OSM overlay ------------------------------------------------------
    if args.with_osm:
        osm_file = get_osm_path(args.root, args.sequence)
        if os.path.isfile(osm_file):
            print(f"Loading OSM: {osm_file}")
            loader = OSMLoader(osm_file, origin_latlon=ORIGIN_LATLON)
            osm_geoms = loader.get_geometries(
                z_offset=path_z_default, thickness=args.osm_thickness, buildings_only=not args.osm_all,
            )
            if osm_geoms:
                if first_pose_for_osm is not None:
                    trans = np.array([-first_pose_for_osm[0], -first_pose_for_osm[1], 0.0])
                    for mesh in osm_geoms:
                        mesh.translate(trans)
                    print("OSM: shifted by -first_pose to align with semantic map")
                geoms.extend(osm_geoms)
                kind = "building" if not args.osm_all else "geometry"
                print(f"OSM: {len(osm_geoms)} {kind} groups")
        else:
            print(f"OSM file not found: {osm_file}", file=sys.stderr)

    # -- Semantic map -----------------------------------------------------
    print("Building semantic map...")
    points, colors = build_semantic_map(
        args.root,
        sequence_index=args.sequence,
        scan_skip=args.scan_skip,
        voxel_size=voxel_size,
        max_range=args.max_range,
        max_points=args.max_points,
    )

    if points is not None:
        if first_pose_for_osm is not None and (args.with_osm or args.with_path):
            points = points - first_pose_for_osm.astype(points.dtype)

        print(f"Total points: {len(points)}")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geoms.insert(0, pcd)

        if args.save:
            out_dir = os.path.dirname(os.path.abspath(args.save))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            o3d.io.write_point_cloud(args.save, pcd)
            print(f"Saved to {args.save}")
    else:
        print("No semantic points loaded. Check paths and sequence.", file=sys.stderr)

    if not geoms:
        print("Nothing to display.", file=sys.stderr)
        sys.exit(1)

    print("Opening Open3D viewer (mouse: rotate, shift+mouse: pan, wheel: zoom, Q/ESC: quit).")
    o3d.visualization.draw_geometries(geoms)


if __name__ == "__main__":
    main()
