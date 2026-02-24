#!/usr/bin/env python3
"""
Python port of visualize_map_osm.cpp.

Accumulates a labelled LiDAR map from all frames in a MCD sequence and
overlays OSM polylines, using the same transform chain as the C++ tool:

    lidar_to_map = body_to_world_shifted * inv(body_to_lidar)

where  body_to_world_shifted = poseToMatrix(pose) – init_rel_pos.

Usage (all paths required; --config optional for init_rel_pos, colors, osm_origin):
    python visualize_osm_xml.py \\
        --config configs/mcd_config.yaml \\
        --osm example_data/mcd/kth.osm \\
        --scan_dir example_data/mcd/kth_day_06/lidar_bin/data \\
        --label_dir example_data/mcd/kth_day_06/gt_labels \\
        --pose example_data/mcd/kth_day_06/pose_inW.csv \\
        --calib example_data/mcd/hhs_calib.yaml
"""

import argparse
from typing import Optional
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from script_utils import (
    ConfigReader, OSMLoader, create_thick_lines,
    load_body_to_lidar, load_poses_csv, transform_points_to_world,
    map_labels_to_colors,
)


# ─── Dataset / calibration loading ───────────────────────────────────────────



def _collect_scan_label_pairs(lidar_dir: str, label_dir: str) -> list:
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




def build_map_cloud(pairs: list,
                    poses_by_id: dict,
                    body_to_lidar: np.ndarray,
                    init_rel_pos: np.ndarray,
                    colors_by_label: dict,
                    step: int = 1,
                    max_scans: Optional[int] = None) -> 'o3d.geometry.PointCloud | None':
    """
    Accumulate scans (stride=step) into a single Open3D PointCloud coloured
    by semantic label.  Mirrors the scan loop in visualize_map_osm.cpp.

    Binary format: lidar .bin = float32[N*4] (x,y,z,intensity),
                   label .bin = uint32[N].
    """
    all_pts    = []
    all_colors = []
    loaded = skipped = 0

    for i in range(0, len(pairs), step):
        if max_scans is not None and loaded >= max_scans:
            break
        scan_id, scan_path, label_path = pairs[i]
        if scan_id not in poses_by_id:
            skipped += 1
            continue

        try:
            raw = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)
            pts = raw[:, :3]
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

        world = transform_points_to_world(
            pts, poses_by_id[scan_id], body_to_lidar, init_rel_pos)

        colours = map_labels_to_colors(labels, colors_by_label)

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

    # ── Config file (for init_rel_pos, colors, osm_origin; paths come from CLI) ─
    parser.add_argument("--config", type=str, default=None,
                        help="Path to mcd_config.yaml (init_rel_pos, colors, osm_origin).")
    parser.add_argument("--skip_frames", type=int, default=None,
                        help="Override skip_frames from config "
                             "(C++ third positional arg; default: use config or 0).")
    parser.add_argument("--max_scans", type=int, default=None,
                        help="Maximum number of scans to add to the map (default: no limit).")

    # ── Required paths ────────────────────────────────────────────────────
    parser.add_argument("--osm", type=str, required=True,
                        help="Path to .osm file.")
    parser.add_argument("--scan_dir", type=str, required=True,
                        help="Directory containing lidar .bin files.")
    parser.add_argument("--label_dir", type=str, required=True,
                        help="Directory containing label .bin files.")
    parser.add_argument("--pose", type=str, required=True,
                        help="Path to pose_inW.csv.")
    parser.add_argument("--calib", type=str, required=True,
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
        'lidar_dir':        args.scan_dir,
        'label_dir':        args.label_dir,
        'pose_path':        args.pose,
        'calibration_path': args.calib,
        'osm_file':         args.osm,
        'skip_frames':      0,
        'use_osm_origin':   False,
        'osm_origin_lat':   0.0,
        'osm_origin_lon':   0.0,
        'use_init_rel_pos': False,
        'init_rel_pos':     np.zeros(3),
    }

    config_reader = None
    if args.config:
        config_reader = ConfigReader(args.config)
        cfg.update(config_reader.get_non_path_config())
        print(f"Loaded config from {args.config}")

    # CLI overrides for non-path config
    if args.skip_frames is not None:
        cfg['skip_frames'] = args.skip_frames
    if args.init_rel_pos is not None:
        cfg['init_rel_pos'] = np.array(args.init_rel_pos)
        cfg['use_init_rel_pos'] = True
    if args.osm_origin_lat is not None and args.osm_origin_lon is not None:
        cfg['osm_origin_lat'] = args.osm_origin_lat
        cfg['osm_origin_lon'] = args.osm_origin_lon
        cfg['use_osm_origin'] = True

    step = cfg['skip_frames'] + 1

    # ── Load poses ─────────────────────────────────────────────────────────
    poses_by_id = load_poses_csv(cfg['pose_path'])
    if not poses_by_id:
        sys.exit("No poses loaded.")

    # ── Load calibration ───────────────────────────────────────────────────
    body_to_lidar = load_body_to_lidar(cfg['calibration_path'])
    init_rel_pos  = cfg['init_rel_pos'] if cfg['use_init_rel_pos'] else np.zeros(3)

    print(f"init_rel_pos = [{init_rel_pos[0]:.4f}, "
          f"{init_rel_pos[1]:.4f}, {init_rel_pos[2]:.4f}]")

    # ── Collect scan/label pairs ───────────────────────────────────────────
    pairs = _collect_scan_label_pairs(cfg['lidar_dir'], cfg['label_dir'])
    if not pairs:
        sys.exit("No scan/label pairs found.")

    # ── Accumulate map cloud (C++ transform_scan_to_world per scan) ───────
    print(f"Accumulating map (step={step}, pairs={len(pairs)}"
          + (f", max_scans={args.max_scans})" if args.max_scans else ")") + "...")
    colors_by_label = config_reader.colors_by_label if config_reader else {}
    map_cloud = build_map_cloud(
        pairs, poses_by_id, body_to_lidar, init_rel_pos,
        colors_by_label,
        step=step, max_scans=args.max_scans)
    if map_cloud is None or len(map_cloud.points) == 0:
        sys.exit("No map points accumulated.")
    print(f"Map cloud: {len(map_cloud.points):,} points")

    # ── Load OSM (uses C++ loader; requires --config) ──────────────────────
    if args.config:
        loader = OSMLoader(cfg['osm_file'], args.config)
        osm_geoms = loader.get_geometries(
            z_offset=args.z_offset, use_thick=args.thick, thickness=args.thickness)
    else:
        print("Warning: OSM overlay requires --config; skipping OSM.")
        osm_geoms = []

    # ── Summary (mirrors C++ stdout) ───────────────────────────────────────
    n_osm = sum(len(g.lines) for g in osm_geoms if hasattr(g, 'lines'))
    print(f"\nMap points={len(map_cloud.points):,}, "
          f"skip_frames={cfg['skip_frames']}, "
          f"initial_position_xyz=[{init_rel_pos[0]:.3f}, "
          f"{init_rel_pos[1]:.3f}, {init_rel_pos[2]:.3f}], "
          f"OSM segments={n_osm}")

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
