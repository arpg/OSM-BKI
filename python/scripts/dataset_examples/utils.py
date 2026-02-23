"""
Shared utilities for KITTI-360 and MCD semantic map visualization scripts.
"""

import math
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import open3d as o3d

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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
# Label mapping helpers
# ---------------------------------------------------------------------------


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
    """
    Map scalar values to RGB using the Viridis colormap (dark = low, yellow = high).
    Returns (N, 3) RGB in [0, 1].
    """
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    if normalize_range:
        vmin, vmax = np.min(v), np.max(v)
        rng = vmax - vmin
        if rng <= 0:
            rng = 1.0
        v = (v - vmin) / rng
    v = np.clip(v, 0.0, 1.0)
    idx = np.clip((v * 255).astype(np.int32), 0, 255)
    return _VIRIDIS_LUT[idx].copy()


# ---------------------------------------------------------------------------
# Mercator projection
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


# ---------------------------------------------------------------------------
# OSM geometry rendering
# ---------------------------------------------------------------------------


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
        cyl = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=length, resolution=8
        )
        cyl.paint_uniform_color(color)
        z_axis = np.array([0.0, 0.0, 1.0])
        direction = vec / length
        rot_axis = np.cross(z_axis, direction)
        rot_angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
        if np.linalg.norm(rot_axis) > 1e-6:
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
            R_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(
                rot_axis * rot_angle
            )
            cyl.rotate(R_mat, center=[0, 0, 0])
        elif np.dot(z_axis, direction) < 0:
            R_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(
                np.array([1.0, 0.0, 0.0]) * np.pi
            )
            cyl.rotate(R_mat, center=[0, 0, 0])
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
            x, y = latlon_to_mercator_relative(
                lat, lon, self.origin_lat, self.origin_lon
            )
            self.nodes[nid] = (x, y)
        for way in self.root.findall("way"):
            node_ids = [nd.get("ref") for nd in way.findall("nd")]
            tags = {tag.get("k"): tag.get("v") for tag in way.findall("tag")}
            if node_ids:
                self.ways.append({"nodes": node_ids, "tags": tags})

    def get_geometries(self, z_offset=0.0, thickness=2.0, buildings_only=True):
        """Return list of Open3D TriangleMesh."""
        batches = defaultdict(
            lambda: {"points": [], "lines": [], "color": OSM_COLORS["default"]}
        )
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
            batch["lines"].extend(
                [[start_idx + i, start_idx + i + 1] for i in range(n_pts - 1)]
            )
            if cat in ("building", "landuse", "amenity", "natural") and n_pts > 2:
                batch["lines"].append([start_idx + n_pts - 1, start_idx])
        geometries = []
        for data in batches.values():
            if not data["points"]:
                continue
            mesh = create_thick_lines(
                data["points"], data["lines"], data["color"],
                radius=thickness / 2.0,
            )
            if mesh is not None:
                geometries.append(mesh)
        return geometries


def create_path_geometry(points_xyz, color=(0.0, 0.8, 0.0), thickness=1.5):
    """Create Open3D geometry for the path (thick line strip). points_xyz: (N, 3)."""
    if points_xyz is None or len(points_xyz) < 2:
        return None
    lines = [[i, i + 1] for i in range(len(points_xyz) - 1)]
    return create_thick_lines(
        points_xyz.tolist(), lines, list(color), radius=thickness / 2.0,
    )
