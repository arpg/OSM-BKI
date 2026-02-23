
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


def read_bin_file(file_path, dtype, shape=None):
    """Read a .bin file and optionally reshape."""
    data = np.fromfile(file_path, dtype=dtype)
    if shape is not None:
        return data.reshape(shape)
    return data