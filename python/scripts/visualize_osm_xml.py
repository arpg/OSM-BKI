#!/usr/bin/env python3
"""
Python port of visualize_map_osm.cpp.

Accumulates a labelled LiDAR map from all frames in a MCD sequence and
overlays OSM polylines, using the same transform chain as the C++ tool:

    lidar_to_map = body_to_world_shifted * inv(body_to_lidar)

where  body_to_world_shifted = poseToMatrix(pose) – init_rel_pos.

Usage (config-file driven, mirrors C++ positional args):
    python visualize_osm_xml.py --config configs/mcd_config.yaml

Usage (explicit paths, no config required):
    python visualize_osm_xml.py \\
        --osm       kth.osm \\
        --scan_dir  kth_day_06/lidar_bin/data \\
        --label_dir kth_day_06/gt_labels \\
        --pose      kth_day_06/pose_inW.csv \\
        --calib     hhs_calib.yaml \\
        --osm_origin_lat 59.348268650 \\
        --osm_origin_lon 18.073204280 \\
        --init_rel_pos 64.393 66.483 38.514 \\
        --skip_frames 10 \\
        --max_scans 100
"""

import argparse
import math
from typing import Optional
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from composite_bki_cpp import latlon_to_mercator


# ─── OSM colour scheme ────────────────────────────────────────────────────────
# Float [0,1] values; RGB byte values match colorForOSM() in visualize_map_osm.cpp.
OSM_COLORS = {
    'building': [ 30/255, 180/255,  30/255],
    'highway':  [240/255, 120/255,  20/255],
    'sidewalk': [220/255, 220/255, 220/255],
    'parking':  [245/255, 210/255,  80/255],
    'barrier':  [170/255, 120/255,  70/255],
    'stairs':   [150/255, 100/255,  60/255],
    'landuse':  [ 60/255, 170/255,  80/255],
    'natural':  [ 20/255, 130/255,  20/255],
    'default':  [140/255, 140/255, 140/255],
}

# Semantic label colours (from mcd_config.yaml `colors:` block, normalised to [0,1])
MCD_LABEL_COLORS = {
    0:  [128/255, 128/255, 128/255],  # barrier
    1:  [119/255,  11/255,  32/255],  # bike
    2:  [  0/255, 100/255,   0/255],  # building
    3:  [139/255,  69/255,  19/255],  # chair
    4:  [101/255,  67/255,  33/255],  # cliff
    5:  [160/255, 160/255, 160/255],  # container
    6:  [244/255,  35/255, 232/255],  # curb
    7:  [190/255, 153/255, 153/255],  # fence
    8:  [255/255, 165/255,   0/255],  # hydrant
    9:  [255/255, 255/255,   0/255],  # infosign
    10: [170/255, 255/255, 150/255],  # lanemarking
    11: [  0/255,   0/255,   0/255],  # noise
    12: [255/255, 255/255,  50/255],  # other
    13: [250/255, 170/255, 160/255],  # parkinglot
    14: [220/255,  20/255,  60/255],  # pedestrian
    15: [153/255, 153/255, 153/255],  # pole
    16: [128/255,  64/255, 128/255],  # road
    17: [  0/255, 100/255,   0/255],  # shelter
    18: [244/255,  35/255, 232/255],  # sidewalk
    19: [128/255,   0/255, 128/255],  # stairs
    20: [  0/255, 150/255, 255/255],  # structure-other
    21: [255/255,  69/255,   0/255],  # traffic-cone
    22: [  0/255,   0/255, 255/255],  # traffic-sign
    23: [139/255,   0/255, 139/255],  # trashbin
    24: [  0/255,  60/255, 135/255],  # treetrunk
    25: [107/255, 142/255,  35/255],  # vegetation
    26: [245/255, 150/255, 100/255],  # vehicle-dynamic
    27: [ 51/255,   0/255,  51/255],  # vehicle-other
    28: [  0/255,   0/255, 142/255],  # vehicle-static
}

# Vectorised look-up table: index = label id, value = [r, g, b] float32
_MAX_LABEL = max(MCD_LABEL_COLORS) + 1
_COLOR_LUT  = np.zeros((_MAX_LABEL, 3), dtype=np.float32)
for _lbl, _rgb in MCD_LABEL_COLORS.items():
    _COLOR_LUT[_lbl] = _rgb


def _color_from_label(label: int) -> list:
    """Return [r, g, b] in [0, 1]; MCD lookup first, then hash fallback
    matching colorFromLabel() in visualize_map_osm.cpp."""
    if 0 <= label < _MAX_LABEL:
        return _COLOR_LUT[label].tolist()
    h = (label * 2654435761) & 0xFFFFFFFF
    return [((h >> 16) & 0xFF) / 255.0,
            ((h >>  8) & 0xFF) / 255.0,
            ( h        & 0xFF) / 255.0]


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def create_thick_lines(points, lines, color, radius: float = 5.0):
    """Cylinder-mesh thick lines (slow on large maps; only used with --thick)."""
    meshes = []
    pts = np.array(points)
    for line in lines:
        start  = pts[line[0]]
        end    = pts[line[1]]
        vec    = end - start
        length = np.linalg.norm(vec)
        if length < 0.01:
            continue
        cyl = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=length, resolution=8)
        cyl.paint_uniform_color(color)
        z_axis    = np.array([0.0, 0.0, 1.0])
        direction = vec / length
        rot_axis  = np.cross(z_axis, direction)
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


# ─── OSM loader ───────────────────────────────────────────────────────────────

class OSMLoader:
    """
    Parses an OSM XML file and projects nodes to local ENU metres using the
    scaled Mercator projection matching the C++ MercatorProjection.

    Also imported by visualize_osm.py.
    """

    def __init__(self, xml_file: str,
                 origin_lat_override=None,
                 origin_lon_override=None,
                 world_offset_x: float = 0.0,
                 world_offset_y: float = 0.0):
        """
        Args:
            xml_file:            Path to .osm file.
            origin_lat_override: Overrides the auto-derived origin latitude.
                                 Use init_latlon_day_06[0] from mcd_config.yaml
                                 (KTH: 59.348268650) for LiDAR-frame alignment.
            origin_lon_override: Corresponding longitude (KTH: 18.073204280).
            world_offset_x:      World-frame X of the GPS reference point
                                 (osm_world_offset_x in mcd_config.yaml; default 0.0).
            world_offset_y:      World-frame Y of the GPS reference point (default 0.0).
        """
        print(f"Loading {xml_file}...")
        self.tree = ET.parse(xml_file)
        self.root = self.tree.getroot()
        self.nodes: dict  = {}   # id → (x, y) metres
        self.ways:  list  = []   # [{'nodes': [...], 'tags': {...}}]
        self.bounds = {'minlat': 0.0, 'minlon': 0.0, 'maxlat': 0.0, 'maxlon': 0.0}
        self.origin_lat = 0.0
        self.origin_lon = 0.0
        self.world_offset_x = world_offset_x
        self.world_offset_y = world_offset_y
        self._origin_lat_override = origin_lat_override
        self._origin_lon_override = origin_lon_override
        self._parse()

    def _parse(self):
        # 1. Determine projection origin
        bounds = self.root.find('bounds')
        if bounds is not None:
            self.bounds['minlat'] = float(bounds.get('minlat'))
            self.bounds['minlon'] = float(bounds.get('minlon'))
            self.bounds['maxlat'] = float(bounds.get('maxlat'))
            self.bounds['maxlon'] = float(bounds.get('maxlon'))
            self.origin_lat = (self.bounds['minlat'] + self.bounds['maxlat']) / 2.0
            self.origin_lon = (self.bounds['minlon'] + self.bounds['maxlon']) / 2.0
        else:
            # No <bounds> element (e.g. Overpass API output): derive from all nodes.
            all_nodes = self.root.findall('node')
            if all_nodes:
                lats = [float(n.get('lat')) for n in all_nodes]
                lons = [float(n.get('lon')) for n in all_nodes]
                self.bounds = {
                    'minlat': min(lats), 'maxlat': max(lats),
                    'minlon': min(lons), 'maxlon': max(lons),
                }
                self.origin_lat = (self.bounds['minlat'] + self.bounds['maxlat']) / 2.0
                self.origin_lon = (self.bounds['minlon'] + self.bounds['maxlon']) / 2.0
                print(f"No <bounds> element; computed origin from {len(all_nodes)} nodes.")

        # Explicit override always wins (required for correct LiDAR-frame alignment).
        if (self._origin_lat_override is not None and
                self._origin_lon_override is not None):
            self.origin_lat = self._origin_lat_override
            self.origin_lon = self._origin_lon_override
            print(f"Using explicit OSM origin: "
                  f"({self.origin_lat:.9f}, {self.origin_lon:.9f})")

        print(f"OSM origin: {self.origin_lat:.6f}, {self.origin_lon:.6f}")

        # 2. Project nodes with scaled Mercator (matches C++ MercatorProjection)
        print("Projecting nodes...")
        for node in self.root.findall('node'):
            nid = node.get('id')
            lat = float(node.get('lat'))
            lon = float(node.get('lon'))
            x, y = latlon_to_mercator(lat, lon,
                                      self.origin_lat, self.origin_lon,
                                      self.world_offset_x, self.world_offset_y)
            self.nodes[nid] = (x, y)
        print(f"Processed {len(self.nodes)} nodes.")

        # 3. Parse ways
        for way in self.root.findall('way'):
            node_ids = [nd.get('ref') for nd in way.findall('nd')]
            tags     = {tag.get('k'): tag.get('v') for tag in way.findall('tag')}
            if node_ids:
                self.ways.append({'nodes': node_ids, 'tags': tags})
        print(f"Processed {len(self.ways)} ways.")

    def _batch_ways(self, z_offset: float):
        """Categorise ways and bucket them by category for batch rendering."""
        batches = defaultdict(
            lambda: {'points': [], 'lines': [], 'color': OSM_COLORS['default']})

        for way in self.ways:
            tags     = way['tags']
            node_ids = way['nodes']

            # Map OSM tags → C++ Category equivalents
            if 'building' in tags:
                cat, color = 'building', OSM_COLORS['building']
            elif 'highway' in tags:
                val = tags['highway']
                if val in ('footway', 'path', 'steps', 'pedestrian', 'cycleway'):
                    cat, color = 'sidewalk', OSM_COLORS['sidewalk']
                else:
                    cat, color = 'highway', OSM_COLORS['highway']
            elif 'amenity' in tags and tags['amenity'] == 'parking':
                cat, color = 'parking', OSM_COLORS['parking']
            elif 'barrier' in tags:
                val = tags['barrier']
                cat  = 'stairs' if val == 'stairs' else 'barrier'
                color = OSM_COLORS[cat]
            elif 'landuse' in tags:
                cat, color = 'landuse', OSM_COLORS['landuse']
            elif 'natural' in tags:
                cat, color = 'natural', OSM_COLORS['natural']
            else:
                cat, color = 'default', OSM_COLORS['default']

            pts = [[self.nodes[nid][0], self.nodes[nid][1], z_offset]
                   for nid in node_ids if nid in self.nodes]
            if len(pts) < 2:
                continue

            batch = batches[cat]
            batch['color'] = color
            start = len(batch['points'])
            batch['points'].extend(pts)
            n = len(pts)
            batch['lines'].extend([[start + i, start + i + 1] for i in range(n - 1)])
            # Close polygon for area features
            if cat in ('building', 'landuse', 'parking', 'natural') and n > 2:
                batch['lines'].append([start + n - 1, start])

        return batches

    def get_geometries(self, z_offset: float = 0.05,
                       thickness: float = 10.0,
                       use_thick: bool = False):
        """
        Return Open3D geometry objects for all ways.

        z_offset defaults to 0.05 – matching the C++ which draws OSM segments
        at z=0.05f so they sit just above the ground point cloud.

        use_thick=True renders thick cylinder meshes (slow on large maps).
        """
        batches    = self._batch_ways(z_offset)
        total_segs = sum(len(d['lines']) for d in batches.values())
        mode       = "thick-line meshes" if use_thick else "LineSets"
        print(f"Constructing {mode} for {total_segs} OSM segments...")

        geoms = []
        for _cat, data in batches.items():
            if not data['points']:
                continue
            if use_thick:
                mesh = create_thick_lines(data['points'], data['lines'],
                                          data['color'], radius=thickness / 2.0)
                if mesh:
                    geoms.append(mesh)
            else:
                pts = np.array(data['points'])
                ls  = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(pts)
                ls.lines  = o3d.utility.Vector2iVector(data['lines'])
                ls.colors = o3d.utility.Vector3dVector(
                    [data['color']] * len(data['lines']))
                geoms.append(ls)

        print(f"Created {len(geoms)} {mode}.")
        return geoms


# ─── Dataset / calibration loading ───────────────────────────────────────────

def load_dataset_config(config_path: str) -> dict:
    """
    Read mcd_config.yaml and return a dict mirroring DatasetConfig in
    dataset_utils.hpp, with all paths fully resolved.

    Mirrors loadDatasetConfig() + path construction in dataset_utils.cpp.
    """
    import yaml
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    root = raw['dataset_root_path']
    seq  = raw['sequence']
    base = os.path.join(root, seq)

    cfg: dict = {
        'skip_frames':       int(raw.get('skip_frames', 0)),
        'lidar_dir':         os.path.join(base, 'lidar_bin', 'data'),
        'label_dir':         os.path.join(base, 'gt_labels'),
        'pose_path':         os.path.join(base, 'pose_inW.csv'),
        'calibration_path':  os.path.join(root, 'hhs_calib.yaml'),
        'osm_file':          '',
        'use_osm_origin':    False,
        'osm_origin_lat':    0.0,
        'osm_origin_lon':    0.0,
        'use_init_rel_pos':  False,
        'init_rel_pos':      np.zeros(3),
        'colors_by_label':   {},
    }

    if 'osm_file' in raw:
        cfg['osm_file'] = os.path.join(root, raw['osm_file'])

    # init_latlon_day_06 → OSM projection origin (matches use_osm_origin_from_mcd)
    if 'init_latlon_day_06' in raw:
        ll = raw['init_latlon_day_06']
        cfg['osm_origin_lat'] = float(ll[0])
        cfg['osm_origin_lon'] = float(ll[1])
        cfg['use_osm_origin'] = True

    # init_rel_pos_day_06 → pose re-centring (matches use_init_rel_pos)
    if 'init_rel_pos_day_06' in raw:
        rp = raw['init_rel_pos_day_06']
        cfg['init_rel_pos']     = np.array([float(rp[0]), float(rp[1]), float(rp[2])])
        cfg['use_init_rel_pos'] = True

    # Semantic label colours from config `colors:` block
    if 'colors' in raw:
        for lbl_str, rgb in raw['colors'].items():
            cfg['colors_by_label'][int(lbl_str)] = [c / 255.0 for c in rgb]

    return cfg


def load_body_to_lidar(calib_yaml_path: str) -> np.ndarray:
    """
    Read body/os_sensor/T from hhs_calib.yaml → 4×4 float64 matrix.
    Mirrors readBodyToLidarCalibration() in file_io.cpp.
    """
    import yaml
    with open(calib_yaml_path) as f:
        calib = yaml.safe_load(f)
    rows = calib['body']['os_sensor']['T']
    mat  = np.array(rows, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(
            f"Expected 4×4 matrix in {calib_yaml_path}, got {mat.shape}")
    print(f"Loaded body→LiDAR calibration from {calib_yaml_path}")
    return mat


def load_poses_csv(pose_csv_path: str) -> dict:
    """
    Read pose CSV (num, t, x, y, z, qx, qy, qz[, qw]) and return
    {scan_id: np.array([x, y, z, qx, qy, qz, qw])}.

    Mirrors readPosesCSV() / parsePoseLine() in file_io.cpp:
      col 0 → scan_id   col 2-4 → x, y, z
      col 5-7 → qx, qy, qz   col 8 → qw (default 1.0 if absent)
    Non-numeric tokens (CSV header) are silently skipped.
    """
    poses: dict = {}
    with open(pose_csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            sep  = ',' if ',' in line else None
            vals = []
            for tok in (line.split(sep) if sep else line.split()):
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
            if len(vals) < 8:
                continue
            scan_id       = int(vals[0])
            qw            = vals[8] if len(vals) > 8 else 1.0
            poses[scan_id] = np.array(
                [vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], qw],
                dtype=np.float64)
    print(f"Loaded {len(poses)} poses from {pose_csv_path}")
    return poses


def collect_scan_label_pairs(lidar_dir: str, label_dir: str) -> list:
    """
    Return a sorted list of (scan_id, scan_path, label_path) for all .bin
    files in lidar_dir that have a matching .bin in label_dir.
    Mirrors collectScanLabelPairs() in dataset_utils.cpp.
    """
    pairs = []
    for lf in sorted(Path(lidar_dir).glob('*.bin')):
        lp = Path(label_dir) / (lf.stem + '.bin')
        if lp.exists():
            try:
                sid = int(lf.stem)
            except ValueError:
                sid = -1
            pairs.append((sid, str(lf), str(lp)))
    print(f"Found {len(pairs)} scan/label pairs in {lidar_dir}")
    return pairs


def build_lidar_to_map(pose: np.ndarray,
                       body_to_lidar: np.ndarray,
                       init_rel_pos: np.ndarray) -> np.ndarray:
    """
    Mirrors the per-pose transform in visualize_map_osm.cpp:

        body_to_world_rel = poseToMatrix(pose)
        body_to_world_rel[:3, 3] -= init_rel_pos      # re-centre world origin
        lidar_to_map = body_to_world_rel @ inv(body_to_lidar)

    pose = [x, y, z, qx, qy, qz, qw]  (C++ poseToMatrix uses Eigen::Quaterniond(qw,qx,qy,qz);
                                         scipy from_quat takes [qx,qy,qz,qw])
    """
    x, y, z, qx, qy, qz, qw = pose
    rot = R.from_quat([qx, qy, qz, qw]).as_matrix()

    body_to_world = np.eye(4, dtype=np.float64)
    body_to_world[:3, :3] = rot
    body_to_world[0, 3]   = x - init_rel_pos[0]
    body_to_world[1, 3]   = y - init_rel_pos[1]
    body_to_world[2, 3]   = z - init_rel_pos[2]

    return body_to_world @ np.linalg.inv(body_to_lidar)


def build_map_cloud(pairs: list,
                    lidar_to_map_by_id: dict,
                    colors_by_label: dict,
                    step: int = 1,
                    max_scans: Optional[int] = None) -> 'o3d.geometry.PointCloud | None':
    """
    Accumulate scans (stride=step) into a single Open3D PointCloud coloured
    by semantic label.  Mirrors the scan loop in visualize_map_osm.cpp.

    Binary format: lidar .bin = float32[N×4] (x,y,z,intensity),
                   label .bin = uint32[N].
    """
    all_pts    = []
    all_colors = []
    loaded = skipped = 0

    for i in range(0, len(pairs), step):
        if max_scans is not None and loaded >= max_scans:
            break
        scan_id, scan_path, label_path = pairs[i]
        if scan_id not in lidar_to_map_by_id:
            skipped += 1
            continue

        try:
            raw = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
            pts = raw[:, :3].astype(np.float64)
        except Exception as e:
            print(f"  Warning: could not read {scan_path}: {e}")
            skipped += 1
            continue

        try:
            labels = np.fromfile(label_path, dtype=np.uint32)
        except Exception as e:
            print(f"  Warning: could not read {label_path}: {e}")
            skipped += 1
            continue

        if len(labels) != len(pts):
            skipped += 1
            continue

        # Transform to map frame  (vectorised)
        T      = lidar_to_map_by_id[scan_id]
        pts_h  = np.hstack([pts, np.ones((len(pts), 1))])
        world  = (T @ pts_h.T).T[:, :3]

        # Colour by label (vectorised per unique label)
        colours = np.zeros((len(labels), 3), dtype=np.float32)
        for lbl in np.unique(labels):
            mask = labels == lbl
            if int(lbl) in colors_by_label:
                colours[mask] = colors_by_label[int(lbl)]
            else:
                colours[mask] = _color_from_label(int(lbl))

        all_pts.append(world)
        all_colors.append(colours)
        loaded += 1

    print(f"Scans loaded: {loaded}, skipped: {skipped}")
    if not all_pts:
        return None

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
    cloud.colors = o3d.utility.Vector3dVector(
        np.vstack(all_colors).astype(np.float64))
    return cloud


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Python port of visualize_map_osm.cpp – accumulates a "
                    "labelled LiDAR map and overlays OSM polylines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    # ── Config file (mirrors C++ first two positional args) ────────────────
    parser.add_argument("--config", type=str, default=None,
                        help="Path to mcd_config.yaml (resolves all paths "
                             "automatically, matching C++ loadDatasetConfig).")
    parser.add_argument("--skip_frames", type=int, default=None,
                        help="Override skip_frames from config "
                             "(C++ third positional arg; default: use config or 0).")
    parser.add_argument("--max_scans", type=int, default=None,
                        help="Maximum number of scans to add to the map (default: no limit).")

    # ── Explicit path overrides ────────────────────────────────────────────
    parser.add_argument("--osm", type=str, default=None,
                        help="Path to .osm file.")
    parser.add_argument("--scan_dir", type=str, default=None,
                        help="Directory containing lidar .bin files "
                             "(dataset_root/sequence/lidar_bin/data).")
    parser.add_argument("--label_dir", type=str, default=None,
                        help="Directory containing label .bin files "
                             "(dataset_root/sequence/gt_labels).")
    parser.add_argument("--pose", type=str, default=None,
                        help="Path to pose_inW.csv.")
    parser.add_argument("--calib", type=str, default=None,
                        help="Path to hhs_calib.yaml (body→LiDAR calibration).")
    parser.add_argument("--init_rel_pos", type=float, nargs=3,
                        metavar=('X', 'Y', 'Z'), default=None,
                        help="World-frame initial position to subtract from "
                             "poses (init_rel_pos_day_06 in mcd_config.yaml).")
    parser.add_argument("--osm_origin_lat", type=float, default=None,
                        help="Override OSM projection origin latitude "
                             "(init_latlon_day_06[0]; KTH: 59.348268650).")
    parser.add_argument("--osm_origin_lon", type=float, default=None,
                        help="Override OSM projection origin longitude "
                             "(init_latlon_day_06[1]; KTH: 18.073204280).")
    parser.add_argument("--osm_world_offset_x", type=float, default=0.0,
                        help="World-frame X offset for the OSM origin (default 0.0).")
    parser.add_argument("--osm_world_offset_y", type=float, default=0.0,
                        help="World-frame Y offset for the OSM origin (default 0.0).")

    # ── Visualisation tweaks ───────────────────────────────────────────────
    parser.add_argument("--z_offset", type=float, default=0.05,
                        help="Z height for OSM lines (default 0.05, matching C++).")
    parser.add_argument("--thick", action="store_true",
                        help="Render OSM as thick cylinder meshes (slow on large maps).")
    parser.add_argument("--thickness", type=float, default=20.0,
                        help="Cylinder radius in metres when --thick is active.")
    args = parser.parse_args()

    # ── Resolve configuration ──────────────────────────────────────────────
    cfg: dict = {
        'skip_frames':      0,
        'lidar_dir':        None,
        'label_dir':        None,
        'pose_path':        None,
        'calibration_path': None,
        'osm_file':         '',
        'use_osm_origin':   False,
        'osm_origin_lat':   0.0,
        'osm_origin_lon':   0.0,
        'use_init_rel_pos': False,
        'init_rel_pos':     np.zeros(3),
        'colors_by_label':  {},
    }

    if args.config:
        cfg.update(load_dataset_config(args.config))
        print(f"Loaded config from {args.config}")

    # CLI args always override config file values
    if args.skip_frames  is not None:  cfg['skip_frames']      = args.skip_frames
    if args.scan_dir:                  cfg['lidar_dir']         = args.scan_dir
    if args.label_dir:                 cfg['label_dir']         = args.label_dir
    if args.pose:                      cfg['pose_path']         = args.pose
    if args.calib:                     cfg['calibration_path']  = args.calib
    if args.osm:                       cfg['osm_file']          = args.osm
    if args.init_rel_pos is not None:
        cfg['init_rel_pos']     = np.array(args.init_rel_pos)
        cfg['use_init_rel_pos'] = True
    if args.osm_origin_lat is not None and args.osm_origin_lon is not None:
        cfg['osm_origin_lat'] = args.osm_origin_lat
        cfg['osm_origin_lon'] = args.osm_origin_lon
        cfg['use_osm_origin'] = True

    step = cfg['skip_frames'] + 1

    # ── Validate required inputs ───────────────────────────────────────────
    missing = [k for k in ('lidar_dir', 'label_dir', 'pose_path',
                           'calibration_path', 'osm_file')
               if not cfg.get(k)]
    if missing:
        parser.error(
            f"Missing required inputs: {missing}. "
            "Provide --config or the individual path flags.")

    # ── Load poses ─────────────────────────────────────────────────────────
    poses_by_id = load_poses_csv(cfg['pose_path'])
    if not poses_by_id:
        sys.exit("No poses loaded.")

    # ── Load calibration ───────────────────────────────────────────────────
    body_to_lidar = load_body_to_lidar(cfg['calibration_path'])
    init_rel_pos  = cfg['init_rel_pos'] if cfg['use_init_rel_pos'] else np.zeros(3)

    print(f"init_rel_pos = [{init_rel_pos[0]:.4f}, "
          f"{init_rel_pos[1]:.4f}, {init_rel_pos[2]:.4f}]")

    # ── Pre-compute lidar_to_map per pose (mirrors C++ loop before scan loop)
    lidar_to_map_by_id = {
        scan_id: build_lidar_to_map(pose, body_to_lidar, init_rel_pos)
        for scan_id, pose in poses_by_id.items()
    }

    # ── Collect scan/label pairs ───────────────────────────────────────────
    pairs = collect_scan_label_pairs(cfg['lidar_dir'], cfg['label_dir'])
    if not pairs:
        sys.exit("No scan/label pairs found.")

    # ── Accumulate map cloud ───────────────────────────────────────────────
    print(f"Accumulating map (step={step}, pairs={len(pairs)}"
          + (f", max_scans={args.max_scans})" if args.max_scans else ")") + "...")
    map_cloud = build_map_cloud(
        pairs, lidar_to_map_by_id, cfg['colors_by_label'],
        step=step, max_scans=args.max_scans)
    if map_cloud is None or len(map_cloud.points) == 0:
        sys.exit("No map points accumulated.")
    print(f"Map cloud: {len(map_cloud.points):,} points")

    # ── Load OSM ───────────────────────────────────────────────────────────
    origin_lat = cfg['osm_origin_lat'] if cfg['use_osm_origin'] else None
    origin_lon = cfg['osm_origin_lon'] if cfg['use_osm_origin'] else None
    loader = OSMLoader(cfg['osm_file'],
                       origin_lat_override=origin_lat,
                       origin_lon_override=origin_lon,
                       world_offset_x=args.osm_world_offset_x,
                       world_offset_y=args.osm_world_offset_y)
    osm_geoms = loader.get_geometries(
        z_offset=args.z_offset, use_thick=args.thick, thickness=args.thickness)

    # ── Summary (mirrors C++ stdout) ───────────────────────────────────────
    print(f"\nMap points={len(map_cloud.points):,}, "
          f"skip_frames={cfg['skip_frames']}, "
          f"initial_position_xyz=[{init_rel_pos[0]:.3f}, "
          f"{init_rel_pos[1]:.3f}, {init_rel_pos[2]:.3f}], "
          f"OSM polylines={len(loader.ways)}")

    # ── Visualise (matches PCLVisualizer setup in visualize_map_osm.cpp) ──
    print("\nLaunching visualisation…")
    print("  Mouse: rotate / zoom / pan   R: reset view   Q / Esc: quit")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Map + OSM Viewer", width=1280, height=720)

    vis.add_geometry(map_cloud)
    for g in osm_geoms:
        vis.add_geometry(g)

    # Coordinate axes (matches viewer->addCoordinateSystem(1.0) in C++)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=5.0, origin=[0, 0, 0])
    vis.add_geometry(axes)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.05])  # matches C++ 0.05,0.05,0.05
    opt.point_size = 2.0                                  # matches PCL_VISUALIZER_POINT_SIZE 2

    vis.poll_events()
    vis.update_renderer()
    vis.reset_view_point(True)

    ctr = vis.get_view_control()
    ctr.set_constant_z_far(1_000_000)
    ctr.set_constant_z_near(0.1)
    ctr.set_front([0, 0, -1])   # top-down view
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.3)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
