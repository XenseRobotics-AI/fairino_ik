"""Runtime isolation helpers for shells that have sourced ROS.

ROS Humble commonly prepends Python 3.10 packages and its Pinocchio shared
libraries to ``PYTHONPATH``/``LD_LIBRARY_PATH``. Those paths are ABI
incompatible with this Python 3.12 environment. Application entry points can
therefore restart once after removing only the conflicting ROS entries.
"""

from __future__ import annotations

import os
import re
import sys
import sysconfig
from collections.abc import Mapping
from pathlib import Path
from typing import Callable


_SANITIZED_MARKER = "FAIRINO_IK_RUNTIME_SANITIZED"


def _is_conflicting_python_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if "/opt/ros/" in normalized:
        return True
    match = re.search(
        r"/python(\d+)\.(\d+)/(?:site|dist)-packages(?:/|$)", normalized
    )
    if match is None:
        return False
    return (int(match.group(1)), int(match.group(2))) != (
        sys.version_info.major,
        sys.version_info.minor,
    )


def _filter_path_list(
    value: str, predicate: Callable[[str], bool]
) -> tuple[str, tuple[str, ...]]:
    kept: list[str] = []
    removed: list[str] = []
    for entry in value.split(os.pathsep):
        if not entry:
            continue
        if predicate(entry):
            removed.append(entry)
        else:
            kept.append(entry)
    return os.pathsep.join(kept), tuple(removed)


def sanitized_environment(
    source: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Return an environment with incompatible ROS paths removed."""

    environment = dict(os.environ if source is None else source)
    removed: list[str] = []

    python_path, python_removed = _filter_path_list(
        environment.get("PYTHONPATH", ""), _is_conflicting_python_path
    )
    library_path, library_removed = _filter_path_list(
        environment.get("LD_LIBRARY_PATH", ""),
        lambda entry: "/opt/ros/" in entry.replace("\\", "/"),
    )
    removed.extend(f"PYTHONPATH={entry}" for entry in python_removed)
    removed.extend(f"LD_LIBRARY_PATH={entry}" for entry in library_removed)

    if python_path:
        environment["PYTHONPATH"] = python_path
    else:
        environment.pop("PYTHONPATH", None)
    if library_path:
        environment["LD_LIBRARY_PATH"] = library_path
    else:
        environment.pop("LD_LIBRARY_PATH", None)

    if "/opt/ros/" in environment.get("AMENT_PREFIX_PATH", ""):
        removed.append("AMENT_PREFIX_PATH=" + environment["AMENT_PREFIX_PATH"])
        environment.pop("AMENT_PREFIX_PATH", None)

    return environment, tuple(removed)


def _native_library_paths() -> tuple[str, ...]:
    """Return library directories needed by pip-installed native backends.

    ``pin`` and ``eiquadprog`` install their CMake artifacts below
    ``cmeel.prefix``.  The extension's RPATH covers direct dependencies, but
    the dynamic loader may still need these directories for transitive
    dependencies such as urdfdom.  Conda environments can also provide a
    compatible urdfdom version in ``$CONDA_PREFIX/lib``.
    """

    candidates: list[Path] = []
    pinocchio_prefix = os.environ.get("FAIRINO_PINOCCHIO_PREFIX")
    if pinocchio_prefix:
        candidates.append(Path(pinocchio_prefix).expanduser() / "lib")

    try:
        purelib_value = sysconfig.get_paths()["purelib"]
    except (KeyError, TypeError):
        purelib_value = ""
    if purelib_value:
        purelib = Path(purelib_value)
        candidates.append(purelib / "cmeel.prefix" / "lib")

    # Also discover cmeel.prefix when pip/conda exposes its site-packages
    # through a different sys.path entry.
    for entry in sys.path:
        path = Path(entry)
        if path.name != "site-packages":
            continue
        candidates.append(path / "cmeel.prefix" / "lib")
        for parent in path.parents:
            if parent.name == "cmeel.prefix":
                candidates.append(parent / "lib")
                break

    # Put the environment's ordinary lib directory after cmeel.prefix so a
    # system/conda urdfdom cannot shadow the ABI selected by pin.
    candidates.append(Path(sys.prefix) / "lib")

    result: list[str] = []
    for candidate in candidates:
        value = str(candidate.resolve())
        if candidate.is_dir() and value not in result:
            result.append(value)
    return tuple(result)


def _add_native_library_paths(environment: dict[str, str]) -> bool:
    """Prepend native dependency directories and report whether it changed."""

    current = [entry for entry in environment.get("LD_LIBRARY_PATH", "").split(os.pathsep) if entry]
    native = list(_native_library_paths())
    ordered = native + [entry for entry in current if entry not in native]
    value = os.pathsep.join(ordered)
    previous = environment.get("LD_LIBRARY_PATH", "")
    if value:
        environment["LD_LIBRARY_PATH"] = value
    else:
        environment.pop("LD_LIBRARY_PATH", None)
    return environment.get("LD_LIBRARY_PATH", "") != previous


def _preload_native_dependencies() -> None:
    """Load cmeel's transitive libraries before importing the C++ extension.

    Python extensions built by conda inherit an absolute loader path from the
    interpreter.  The project build removes that machine-specific path, so a
    source/editable install needs the cmeel libraries loaded explicitly before
    the extension asks the loader for Pinocchio's transitive urdfdom deps.
    """

    try:
        import ctypes
    except ImportError:  # pragma: no cover - ctypes is part of CPython
        return
    for directory in _native_library_paths():
        for name in ("libpinocchio_default.so.3", "libpinocchio_parsers.so.3"):
            path = Path(directory) / name
            if not path.is_file():
                continue
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue


def reexec_without_ros(module: str) -> None:
    """Restart an application module when ROS paths conflict.

    Changing ``LD_LIBRARY_PATH`` after Python starts is insufficient because
    the dynamic loader may already have cached its search path.
    """

    if os.environ.get(_SANITIZED_MARKER) == "1":
        return
    environment, removed = sanitized_environment()
    library_paths_changed = _add_native_library_paths(environment)
    if not removed and not library_paths_changed and environment == os.environ:
        return
    environment[_SANITIZED_MARKER] = "1"
    reason = "incompatible ROS Pinocchio paths" if removed else "native dependency paths"
    print(
        f"fairino_ik: detected {reason}; restarting with an isolated runtime.",
        file=sys.stderr,
        flush=True,
    )
    os.execve(
        sys.executable,
        [sys.executable, "-m", module, *sys.argv[1:]],
        environment,
    )


__all__ = ["reexec_without_ros", "sanitized_environment", "_preload_native_dependencies"]
