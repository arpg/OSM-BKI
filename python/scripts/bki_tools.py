#!/usr/bin/env python3
"""
BKI map tools: convert and visualize .bki map files.

Subcommands:
  convert   - Export .bki to point cloud (.bin) and labels (.label)
  visualize - Display .bki voxels in an Open3D viewer, colored by semantic label

The .bki format (version 2/3) stores a sparse block grid. Both commands read it
via BKIReader.read(), filtering voxels by min alpha sum.
"""

import argparse
import struct
import sys
import numpy as np
import open3d as o3d
from pathlib import Path

from script_utils import ConfigReader, OSMLoader, map_labels_to_colors


class BKIReader:
    """Read .bki map files and export voxels as point clouds with labels."""

    BLOCK_SIZE = 8

    @staticmethod
    def _load_label_mapping(config):
        """Load dense index -> raw label ID mapping. config: path (str) or ConfigReader."""
        if isinstance(config, ConfigReader):
            return config.get_label_mapping()
        return ConfigReader(config).get_label_mapping()

    @staticmethod
    def _read_header(f):
        """Read .bki file header. Returns (version, resolution, num_blocks)."""
        version = struct.unpack("B", f.read(1))[0]
        if version not in (2, 3):
            raise ValueError(f"Unsupported .bki version {version} (expected 2 or 3)")
        resolution = struct.unpack("f", f.read(4))[0]
        struct.unpack("f", f.read(4))  # l_scale
        struct.unpack("f", f.read(4))  # sigma_0
        if version >= 3:
            struct.unpack("i", f.read(4))  # current_time
        num_blocks = struct.unpack("Q", f.read(8))[0]
        return version, resolution, num_blocks

    @staticmethod
    def _passes_filter(alpha, min_alpha_sum, min_max_prob):
        """Return (passes: bool, best_raw_label: int) for a voxel's alpha vector."""
        s = float(np.sum(alpha))
        if s < min_alpha_sum:
            return False, 0
        if np.max(alpha) - np.min(alpha) < 1e-6:
            return False, 0
        max_prob = np.max(alpha) / s if s > 0 else 0.0
        if max_prob < min_max_prob:
            return False, 0
        best = int(np.argmax(alpha))
        return True, best

    @staticmethod
    def _voxel_to_world(bx, by, bz, lx, ly, lz, resolution):
        """Convert block + local voxel indices to world (x, y, z)."""
        vx = bx * BKIReader.BLOCK_SIZE + lx
        vy = by * BKIReader.BLOCK_SIZE + ly
        vz = bz * BKIReader.BLOCK_SIZE + lz
        return (
            (vx + 0.5) * resolution,
            (vy + 0.5) * resolution,
            (vz + 0.5) * resolution,
        )

    @classmethod
    def read(cls, bki_path, config, min_alpha_sum=1e-6, min_max_prob=0.0):
        """
        Read .bki file and return (points_xyz, labels_raw).
        points_xyz: (N, 3) float32, labels_raw: (N,) uint32.
        """
        dense_to_raw, K = cls._load_label_mapping(config)
        block_alpha_size = cls.BLOCK_SIZE ** 3 * K

        points_list = []
        labels_list = []

        with open(bki_path, "rb") as f:
            version, resolution, num_blocks = cls._read_header(f)

            for _ in range(num_blocks):
                bx, by, bz = struct.unpack("iii", f.read(12))
                if version >= 3:
                    struct.unpack("i", f.read(4))  # last_updated

                alpha_bytes = f.read(block_alpha_size * 4)
                if len(alpha_bytes) != block_alpha_size * 4:
                    break

                alpha = np.frombuffer(alpha_bytes, dtype=np.float32)
                alpha = alpha.reshape((cls.BLOCK_SIZE, cls.BLOCK_SIZE, cls.BLOCK_SIZE, K))

                for lz in range(cls.BLOCK_SIZE):
                    for ly in range(cls.BLOCK_SIZE):
                        for lx in range(cls.BLOCK_SIZE):
                            a = alpha[lx, ly, lz, :]
                            passes, best = cls._passes_filter(a, min_alpha_sum, min_max_prob)
                            if not passes:
                                continue
                            raw = dense_to_raw[best] if best < len(dense_to_raw) else 0
                            x, y, z = cls._voxel_to_world(bx, by, bz, lx, ly, lz, resolution)
                            points_list.append([x, y, z])
                            labels_list.append(raw)

        if not points_list:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.uint32)

        return (
            np.array(points_list, dtype=np.float32),
            np.array(labels_list, dtype=np.uint32),
        )





def cmd_convert(args):
    """Export .bki to .bin and .label files."""
    bki_path = Path(args.bki)
    if not bki_path.exists():
        raise SystemExit(f"File not found: {bki_path}")
    if not Path(args.config).exists():
        raise SystemExit(f"Config not found: {args.config}")

    stem = bki_path.stem
    out_bin = args.output_bin or str(bki_path.parent / f"{stem}_map.bin")
    out_label = args.output_label or str(bki_path.parent / f"{stem}_map.label")

    cfg = ConfigReader(args.config)
    print(f"Reading {bki_path} with config {args.config}...")
    points_xyz, labels_raw = BKIReader.read(str(bki_path), cfg, min_alpha_sum=args.min_alpha, min_max_prob=args.min_max_prob)
    n = len(points_xyz)
    print(f"Exporting {n} voxels.")

    cloud = np.hstack([points_xyz, np.zeros((n, 1), dtype=np.float32)])
    cloud.tofile(out_bin)
    labels_raw.tofile(out_label)
    print(f"Wrote {out_bin}")
    print(f"Wrote {out_label}")
    return 0


def cmd_visualize(args):
    """Display .bki voxels in Open3D viewer, with optional OSM overlay underneath."""
    bki_path = Path(args.bki)
    if not bki_path.exists():
        raise SystemExit(f"File not found: {args.bki}")
    if not Path(args.config).exists():
        raise SystemExit(f"Config not found: {args.config}")

    cfg = ConfigReader(args.config)
    viz_cfg = cfg.get_visualize_config()
    colors_by_label = cfg.colors_by_label
    osm_path = None if args.no_osm else (args.osm or viz_cfg['osm_path'])

    print(f"Loading {bki_path}...")
    points_xyz, labels_raw = BKIReader.read(str(bki_path), cfg, min_alpha_sum=args.min_alpha, min_max_prob=args.min_max_prob)
    n = len(points_xyz)
    if n == 0:
        raise SystemExit("No voxels to show (try lowering --min-alpha or --min-max-prob).")

    colors = map_labels_to_colors(labels_raw, colors_by_label)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"BKI Map - {bki_path.name}", width=1280, height=720)

    # Add OSM underneath first (if configured and file exists; uses C++ loader)
    osm_shown = False
    if osm_path and Path(osm_path).exists():
        try:
            loader = OSMLoader(osm_path, args.config)
            osm_geoms = loader.get_geometries(z_offset=args.osm_z_offset, use_thick=args.osm_thick, thickness=args.osm_thickness)
            for g in osm_geoms:
                vis.add_geometry(g)
            osm_shown = True
        except ImportError as e:
            print(f"Warning: {e}; skipping OSM overlay.")
        except Exception as e:
            print(f"Warning: OSM load failed: {e}; skipping OSM overlay.")

    vis.add_geometry(pcd)
    vis.get_render_option().point_size = args.point_size
    vis.get_render_option().background_color = np.array([0.05, 0.05, 0.05])

    print(f"Showing {n} voxels" + (" + OSM" if osm_shown else "") + ". Close the window to exit.")
    vis.run()
    vis.destroy_window()
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="BKI map tools: convert and visualize .bki files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bki_tools convert map.bki --config mcd_config.yaml
  bki_tools visualize map.bki --config mcd_config.yaml --min-alpha 0.01 --min-max-prob 0.5
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand")

    # convert
    p_convert = subparsers.add_parser("convert", help="Export .bki to .bin and .label files")
    p_convert.add_argument("bki", help="Path to .bki map file")
    p_convert.add_argument("--config", required=True, help="Path to YAML config (same as used when building the map)")
    p_convert.add_argument("--output-bin", default=None, help="Output .bin path (default: <bki_stem>_map.bin)")
    p_convert.add_argument("--output-label", default=None, help="Output .label path (default: <bki_stem>_map.label)")
    p_convert.add_argument("--min-alpha", type=float, default=1e-6, help="Min alpha sum per voxel (default: 1e-6)")
    p_convert.add_argument("--min-max-prob", type=float, default=0.0, help="Min predicted class probability max(alpha)/sum(alpha) (default: 0.0)")
    p_convert.set_defaults(func=cmd_convert)

    # visualize
    p_vis = subparsers.add_parser("visualize", help="Display .bki voxels in Open3D viewer")
    p_vis.add_argument("bki", help="Path to .bki map file")
    p_vis.add_argument("--config", required=True, help="Path to YAML config used when building the map")
    p_vis.add_argument("--min-alpha", type=float, default=1e-6, help="Min alpha sum per voxel (default: 1e-6)")
    p_vis.add_argument("--min-max-prob", type=float, default=0.0, help="Min predicted class probability max(alpha)/sum(alpha) (default: 0.0)")
    p_vis.add_argument("--point-size", type=float, default=2.0, help="Point size (default: 2.0)")
    p_vis.add_argument("--osm-z-offset", type=float, default=0.05, help="Z height for OSM lines (default: 0.05)")
    p_vis.add_argument("--osm-thick", action="store_true", help="Render OSM as thick cylinder meshes")
    p_vis.add_argument("--osm-thickness", type=float, default=10.0, help="Cylinder radius when --osm-thick (default: 10.0)")
    p_vis.add_argument("--osm", type=str, default=None, help="Override OSM file path (from config by default)")
    p_vis.add_argument("--no-osm", action="store_true", help="Disable OSM overlay even if config has osm_file")
    p_vis.set_defaults(func=cmd_visualize)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
