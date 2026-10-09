"""MeshCat rendering for the independent :mod:`fairino_ik` package.

This module parses visual geometry from the packaged Fairino URDF.  Moving-link
poses are supplied by :class:`fairino_ik.solver.FairinoQPModel`; no kinematics
is duplicated in the renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class _Material:
    color: int
    opacity: float


@dataclass(frozen=True)
class _Visual:
    link_name: str
    node_name: str
    geometry_kind: str
    geometry_data: Any
    local_transform: np.ndarray
    material: _Material | None


def _numbers(value: str | None, count: int, default: Sequence[float]) -> np.ndarray:
    if not value:
        return np.asarray(default, dtype=float)
    result = np.fromstring(value, sep=" ", dtype=float)
    if result.shape != (count,):
        raise ValueError(f"expected {count} numeric values, got {value!r}")
    return result


def _rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    """Return the URDF fixed-axis RPY rotation (Rz(yaw) Ry(pitch) Rx(roll))."""
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _origin_transform(element: ET.Element | None) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    if element is None:
        return transform
    transform[:3, :3] = _rpy_matrix(_numbers(element.get("rpy"), 3, (0.0, 0.0, 0.0)))
    transform[:3, 3] = _numbers(element.get("xyz"), 3, (0.0, 0.0, 0.0))
    return transform


def _safe_node_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


class _ResourceResolver:
    def __init__(self, urdf_path: Path, description_root: Path | None) -> None:
        self.urdf_path = urdf_path.resolve()
        self.description_root = description_root.resolve() if description_root else None

    def resolve(self, uri: str) -> Path:
        if uri.startswith("package://"):
            package_and_path = uri[len("package://") :]
            package, separator, relative = package_and_path.partition("/")
            if not separator:
                raise ValueError(f"invalid package URI: {uri!r}")
            candidates: list[Path] = []
            if self.description_root is not None:
                root = self.description_root
                if root.name == package or (root / "package.xml").is_file():
                    candidates.append(root / relative)
                candidates.append(root / package / relative)
            candidates.extend(
                (
                    self.urdf_path.parent / package / relative,
                    self.urdf_path.parent.parent / package / relative,
                )
            )
            # A URDF inside its own ROS package: <package>/urdf/robot.urdf.
            if self.urdf_path.parent.parent.name == package:
                candidates.append(self.urdf_path.parent.parent / relative)
            for candidate in candidates:
                if candidate.is_file():
                    return candidate.resolve()
            searched = "\n  ".join(str(path) for path in candidates)
            raise FileNotFoundError(f"cannot resolve {uri!r}; searched:\n  {searched}")

        if uri.startswith("file://"):
            parsed = urlparse(uri)
            path = Path(unquote(parsed.path))
        else:
            path = Path(uri).expanduser()
            if not path.is_absolute():
                path = self.urdf_path.parent / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"visual mesh does not exist: {path}")
        return path


def _parse_rgba(material: ET.Element | None, named: Mapping[str, _Material]) -> _Material | None:
    if material is None:
        return None
    color = material.find("color")
    if color is not None and color.get("rgba"):
        rgba = np.clip(_numbers(color.get("rgba"), 4, (0.7, 0.7, 0.7, 1.0)), 0.0, 1.0)
        rgb = tuple(int(round(float(channel) * 255.0)) for channel in rgba[:3])
        return _Material((rgb[0] << 16) | (rgb[1] << 8) | rgb[2], float(rgba[3]))
    name = material.get("name")
    return named.get(name) if name else None


def _named_materials(root: ET.Element) -> dict[str, _Material]:
    result: dict[str, _Material] = {}
    for material in root.findall("material"):
        name = material.get("name")
        parsed = _parse_rgba(material, {})
        if name and parsed is not None:
            result[name] = parsed
    return result


def _parse_visuals(urdf_path: str | Path, description_root: str | Path | None) -> list[_Visual]:
    path = Path(urdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"URDF does not exist: {path}")
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ValueError(f"{path} is not a URDF robot document")

    resolver = _ResourceResolver(
        path,
        Path(description_root).expanduser() if description_root is not None else None,
    )
    named_materials = _named_materials(root)
    visuals: list[_Visual] = []
    for link in root.findall("link"):
        link_name = link.get("name")
        if not link_name:
            continue
        for index, visual in enumerate(link.findall("visual")):
            geometry = visual.find("geometry")
            if geometry is None or len(geometry) == 0:
                continue
            shape = next(iter(geometry))
            local = _origin_transform(visual.find("origin"))
            kind = shape.tag
            data: Any
            if kind == "mesh":
                filename = shape.get("filename")
                if not filename:
                    raise ValueError(f"mesh visual on link {link_name!r} has no filename")
                data = resolver.resolve(filename)
                scale = _numbers(shape.get("scale"), 3, (1.0, 1.0, 1.0))
                scale_transform = np.eye(4, dtype=float)
                scale_transform[:3, :3] = np.diag(scale)
                local = local @ scale_transform
            elif kind == "box":
                data = _numbers(shape.get("size"), 3, (1.0, 1.0, 1.0))
            elif kind == "sphere":
                data = float(shape.get("radius", "0"))
            elif kind == "cylinder":
                data = (float(shape.get("length", "0")), float(shape.get("radius", "0")))
                # Three.js cylinders are Y-aligned; URDF cylinders are Z-aligned.
                cylinder_alignment = np.eye(4, dtype=float)
                cylinder_alignment[:3, :3] = _rpy_matrix((math.pi / 2.0, 0.0, 0.0))
                local = local @ cylinder_alignment
            else:
                raise ValueError(
                    f"unsupported visual geometry {kind!r} on link {link_name!r}"
                )

            visual_name = visual.get("name") or f"visual_{index}"
            visuals.append(
                _Visual(
                    link_name=link_name,
                    node_name=f"robot/{_safe_node_name(link_name)}/{_safe_node_name(visual_name)}_{index}",
                    geometry_kind=kind,
                    geometry_data=data,
                    local_transform=local,
                    material=_parse_rgba(visual.find("material"), named_materials),
                )
            )
    if not visuals:
        raise ValueError(f"URDF contains no renderable visual elements: {path}")
    return visuals


def _meshcat_modules():
    try:
        import meshcat
        import meshcat.geometry as geometry
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "MeshCat visualization requires the optional 'meshcat' package; "
            "install it with: pip install meshcat"
        ) from exc
    return meshcat, geometry


def _mesh_geometry(geometry_module: Any, path: Path):
    extension = path.suffix.lower()
    loaders = {
        ".dae": "DaeMeshGeometry",
        ".obj": "ObjMeshGeometry",
        ".stl": "StlMeshGeometry",
    }
    loader_name = loaders.get(extension)
    if loader_name is None or not hasattr(geometry_module, loader_name):
        raise ValueError(f"MeshCat cannot load visual mesh format {extension!r}: {path}")
    return getattr(geometry_module, loader_name).from_file(str(path))


class Visualizer:
    """Render a Fairino model using its Pinocchio frame placements."""

    def __init__(
        self,
        model: Any,
        urdf_path: str | Path,
        description_root: str | Path | None = None,
        *,
        open_browser: bool = True,
        root_node: str = "fairino",
    ) -> None:
        meshcat, geometry = _meshcat_modules()
        self.model = model
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self.description_root = (
            Path(description_root).expanduser().resolve()
            if description_root is not None
            else None
        )
        self._root_node = root_node.strip("/") or "fairino"
        self._visuals = _parse_visuals(self.urdf_path, self.description_root)
        self._viewer = meshcat.Visualizer(zmq_url=None)

        try:
            del self._viewer[self._root_node]
        except Exception:
            pass
        for visual in self._visuals:
            node = self._viewer[f"{self._root_node}/{visual.node_name}"]
            if visual.geometry_kind == "mesh":
                obj = _mesh_geometry(geometry, visual.geometry_data)
            elif visual.geometry_kind == "box":
                obj = geometry.Box(visual.geometry_data)
            elif visual.geometry_kind == "sphere":
                obj = geometry.Sphere(visual.geometry_data)
            elif visual.geometry_kind == "cylinder":
                obj = geometry.Cylinder(*visual.geometry_data)
            else:  # guarded by _parse_visuals
                raise AssertionError(visual.geometry_kind)

            if visual.material is None:
                node.set_object(obj)
            else:
                material = geometry.MeshPhongMaterial()
                material.color = visual.material.color
                material.opacity = visual.material.opacity
                material.transparent = visual.material.opacity < 1.0
                node.set_object(obj, material)

        if open_browser:
            self._viewer.open()

    @property
    def viewer(self):
        return self._viewer

    @property
    def url(self) -> str:
        return str(self._viewer.url())

    def update(self, q: Sequence[float], options: Any = None) -> None:
        """Display a fourteen-joint left-plus-right arm configuration.

        ``options`` supplies the locked torso/head joints; the default
        :class:`fairino_ik.IKOptions` shows the upright zero torso.
        """
        configuration = np.asarray(q, dtype=float)
        if configuration.shape != (int(self.model.nq),):
            raise ValueError(
                f"q must have shape ({int(self.model.nq)},), got {configuration.shape}"
            )
        placements = self.model.frame_placements(configuration, options)
        if not isinstance(placements, Mapping):
            try:
                placements = dict(placements)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "FairinoQPModel.frame_placements(q) must return a name-to-4x4 mapping"
                ) from exc

        missing: set[str] = set()
        for visual in self._visuals:
            placement = placements.get(visual.link_name)
            if placement is None:
                missing.add(visual.link_name)
                continue
            world_from_link = np.asarray(placement, dtype=float)
            if world_from_link.shape != (4, 4):
                raise ValueError(
                    f"placement for {visual.link_name!r} must be 4x4, "
                    f"got {world_from_link.shape}"
                )
            self._viewer[f"{self._root_node}/{visual.node_name}"].set_transform(
                world_from_link @ visual.local_transform
            )
        if missing:
            names = ", ".join(sorted(missing))
            raise KeyError(f"Pinocchio frame placements are missing visual link frames: {names}")

    def play(
        self, trajectory: Sequence[Sequence[float]], dt: float, options: Any = None
    ) -> None:
        """Play a joint trajectory at a fixed sampling period."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        deadline = time.monotonic()
        for q in trajectory:
            self.update(q, options)
            deadline += dt
            time.sleep(max(0.0, deadline - time.monotonic()))

    def draw_path(
        self,
        points_xyz: Sequence[Sequence[float]],
        *,
        node_name: str = "tcp_path",
        color: int = 0x00CC44,
    ) -> None:
        """Draw an end-effector path whose points were obtained from FK."""
        _, geometry = _meshcat_modules()
        points = np.asarray(points_xyz, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points_xyz must have shape (N, 3), got {points.shape}")
        if len(points) < 2:
            return
        line = geometry.Line(
            geometry.PointsGeometry(points.T),
            geometry.LineBasicMaterial(color=color, linewidth=2),
        )
        self._viewer[
            f"{self._root_node}/{_safe_node_name(node_name)}"
        ].set_object(line)


MeshcatVisualizer = Visualizer

__all__ = ["MeshcatVisualizer", "Visualizer"]
