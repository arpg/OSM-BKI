"""
Build the composite_bki_cpp extension (pybind11) and install osm_bki package.
Scripts expect: import composite_bki_cpp; composite_bki_cpp.PyContinuousBKI(...)
"""

import atexit
import os
import sys
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from distutils.cmd import Command
import shutil

_setup_dir = Path(__file__).resolve().parent


def _remove_build_artifacts():
    """Remove build/ and *.egg-info/ from the project dir. Runs at process exit after install."""
    root = _setup_dir
    to_remove = ["build", "osm_bki.egg-info", "osms_bki.egg-info"]
    for p in root.glob("osm_bki-*.egg-info"):
        to_remove.append(p.name)
    for name in to_remove:
        path = root / name
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            print(f"Removed {path}")


class install_then_clean(install):
    """Run normal install, then register atexit to remove build/ and egg-info when process exits."""
    def run(self):
        install.run(self)
        atexit.register(_remove_build_artifacts)


class clean(Command):
    """Remove build/ and *.egg-info/. Run manually: python setup.py clean"""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        _remove_build_artifacts()


def get_include_dirs():
    root = Path(__file__).resolve().parent
    cpp_include = root.parent / "cpp" / "osm_bki" / "include"
    if not cpp_include.is_dir():
        raise RuntimeError(f"C++ include dir not found: {cpp_include}")
    return [str(cpp_include)]


def get_sources():
    root = Path(__file__).resolve().parent
    cpp_src = root.parent / "cpp" / "osm_bki" / "src"
    pybind_src = root / "osm_bki" / "pybind" / "osm_bki_bindings.cpp"
    sources = [
        str(pybind_src),
        str(cpp_src / "continuous_bki.cpp"),
        str(cpp_src / "osm_loader.cpp"),
    ]
    for s in sources:
        if not Path(s).exists():
            raise RuntimeError(f"Source not found: {s}")
    return sources


class get_pybind_include(object):
    def __str__(self):
        import pybind11
        return pybind11.get_include()


def extra_compile_args():
    args = ["-std=c++17", "-O3"]
    # OpenMP
    import subprocess
    try:
        subprocess.check_output(["g++", "-fopenmp", "-E", "-"], stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        args.append("-fopenmp")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return args


def extra_link_args():
    args = []
    try:
        import subprocess
        subprocess.check_output(["g++", "-fopenmp", "-E", "-"], stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        args.append("-fopenmp")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return args


ext = Extension(
    "composite_bki_cpp",
    sources=get_sources(),
    include_dirs=[
        get_pybind_include(),
        *get_include_dirs(),
    ],
    language="c++",
    extra_compile_args=extra_compile_args(),
    extra_link_args=extra_link_args(),
)

setup(
    name="osm_bki",
    version="2.0.0",
    description="OSM-S-BKI / Composite BKI: semantic BKI for LiDAR with OSM priors",
    author="Composite BKI Team",
    license="MIT",
    python_requires=">=3.7",
    packages=["osm_bki"],
    package_dir={"osm_bki": "osm_bki"},
    ext_modules=[ext],
    install_requires=["numpy>=1.20.0", "pybind11>=2.6.0"],
    setup_requires=["pybind11>=2.6.0"],
    cmdclass={"build_ext": build_ext, "install": install_then_clean, "clean": clean},
    options={"egg_info": {"egg_base": "."}},
)
