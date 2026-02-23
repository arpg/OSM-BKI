"""
Composite BKI - Fast Semantic-Spatial Bayesian Kernel Inference for LiDAR

A high-performance C++ implementation with Python bindings for semantic
segmentation refinement of LiDAR point clouds using OpenStreetMap priors.

Features:
- 8-15x faster than pure Python implementation
- Multi-threaded with OpenMP (automatic parallelization)
- Support for MCD and SemanticKITTI label formats
- Optional GPU acceleration (CUDA)

Example:
    >>> import composite_bki_cpp
    >>> bki = composite_bki_cpp.PyContinuousBKI(
    ...     osm_path="osm_map.bin",
    ...     config_path="configs/mcd_config.yaml"
    ... )
    >>> bki.update(labels, points)
    >>> refined = bki.infer(points)

Command-line usage:
    $ composite-bki --scan scan.bin --label labels.label --osm map.bin --output refined.label
"""

__version__ = "2.0.0"
__author__ = "Composite BKI Team"
__license__ = "MIT"

# Import the Cython extension when the package is imported
try:
    import composite_bki_cpp
    from composite_bki_cpp import (
        PyContinuousBKI,
        latlon_to_mercator
    )
    
    __all__ = [
        'PyContinuousBKI',
        'latlon_to_mercator',
        '__version__',
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(
        f"Could not import compiled extension: {e}\n"
        "Please build the extension first:\n"
        "  python setup.py build_ext --inplace\n"
        "Or install the package:\n"
        "  pip install -e ."
    )
    __all__ = ['__version__']


def get_version():
    """Get the package version."""
    return __version__


def print_info():
    """Print package information."""
    print(f"Composite BKI C++ v{__version__}")
    print("High-performance semantic segmentation for LiDAR point clouds")
    print("\nFeatures:")
    print("  - Multi-threaded CPU processing (OpenMP)")
    print("  - Optional GPU acceleration (CUDA)")
    print("  - MCD and SemanticKITTI label formats")
    print("  - 8-15x faster than Python implementation")
    print("\nUsage:")
    print("  Library: import composite_bki_cpp")
    print("  CLI:     composite-bki --help")
