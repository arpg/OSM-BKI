#!/usr/bin/env python3
"""
Compute the GPS coordinates of the world-frame origin (0, 0, 0).

Given a known GPS point (ORIGIN_LATLON) and its position in the world frame
(ORIGIN_REL_POS), this script inverts the Mercator projection to find the
lat/lon that corresponds to world-frame (0, 0, 0).

This lets you replace both ORIGIN_LATLON and ORIGIN_REL_POS with a single
GPS origin at world (0,0,0), eliminating the translation step in OSM overlay.

Usage:
    python calc_origin_gps.py
    python calc_origin_gps.py --lat 59.348268650 --lon 18.073204280 --rx 64.39 --ry 66.48
"""

import argparse

from utils import latlon_to_mercator_relative, mercator_relative_to_latlon


def calc_world_origin_gps(origin_lat, origin_lon, rel_x, rel_y):
    """
    Given a GPS reference point and its (x, y) in the world frame, compute
    the GPS coordinates of world-frame (0, 0, 0).

    The OSM overlay maps GPS -> Mercator-relative coords, then shifts by
    +(rel_x, rel_y) to align with the world frame.  So world (0, 0) is at
    Mercator-relative (-rel_x, -rel_y) from the reference GPS point.
    """
    return mercator_relative_to_latlon(-rel_x, -rel_y, origin_lat, origin_lon)


def main():
    parser = argparse.ArgumentParser(
        description="Compute GPS coordinates of world-frame origin (0,0,0)"
    )
    parser.add_argument("--lat", type=float, default=59.348268650,
                        help="Reference GPS latitude (default: MCD KTH day_06)")
    parser.add_argument("--lon", type=float, default=18.073204280,
                        help="Reference GPS longitude (default: MCD KTH day_06)")
    parser.add_argument("--rx", type=float, default=64.3932532565158,
                        help="Reference point x in world frame (ORIGIN_REL_POS[0])")
    parser.add_argument("--ry", type=float, default=66.4832330946657,
                        help="Reference point y in world frame (ORIGIN_REL_POS[1])")
    args = parser.parse_args()

    print(f"Reference GPS:    lat={args.lat:.9f}, lon={args.lon:.9f}")
    print(f"Reference world:  x={args.rx:.6f}, y={args.ry:.6f}")

    origin_lat, origin_lon = calc_world_origin_gps(
        args.lat, args.lon, args.rx, args.ry,
    )

    print(f"\nWorld-frame origin (0,0,0) GPS:")
    print(f"  lat = {origin_lat:.9f}")
    print(f"  lon = {origin_lon:.9f}")

    # Round-trip verification
    check_x, check_y = latlon_to_mercator_relative(
        args.lat, args.lon, origin_lat, origin_lon,
    )
    print(f"\nVerification (forward-project reference GPS relative to new origin):")
    print(f"  expected: ({args.rx:.6f}, {args.ry:.6f})")
    print(f"  got:      ({check_x:.6f}, {check_y:.6f})")
    err = ((check_x - args.rx) ** 2 + (check_y - args.ry) ** 2) ** 0.5
    print(f"  error:    {err:.9f} m")


if __name__ == "__main__":
    main()
