"""Build the shared C++ IK kernel; Python is only the public facade."""
import os
import re
import sys
import sysconfig
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext as _build_ext
from setuptools import setup
from setuptools.command.egg_info import egg_info as _egg_info

ROOT = Path(__file__).resolve().parent
candidates = [Path(os.environ["FAIRINO_PINOCCHIO_PREFIX"])] if os.environ.get("FAIRINO_PINOCCHIO_PREFIX") else []
# pip's isolated build installs the build requirements (pin, eiquadprog) into
# a temporary overlay that is only visible through sys.path, not purelib.
candidates += [Path(entry) / "cmeel.prefix" for entry in sys.path if entry]
candidates += [Path(sysconfig.get_paths()["purelib"]) / "cmeel.prefix", Path(sys.prefix), Path("/usr/local"), Path("/usr")]
prefix = next(
    (
        p
        for p in candidates
        if (p / "include/pinocchio").is_dir()
        and any((p / "lib").glob("libpinocchio_parsers.so*"))
    ),
    None,
)
if prefix is None:
    raise RuntimeError("Install C++ Pinocchio (pip install pin), or set FAIRINO_PINOCCHIO_PREFIX")
if not (prefix / "include/eiquadprog/eiquadprog-rt.hpp").is_file():
    raise RuntimeError("Install the native QP headers in the Pinocchio prefix: pip install eiquadprog")
eigen = Path("/usr/include/eigen3")
if not eigen.is_dir():
    eigen = prefix / "include"

class egg_info(_egg_info):
    def finalize_options(self):
        if self.egg_base is None:
            self.egg_base = str(ROOT / "build/egg-info")
        Path(self.egg_base).mkdir(parents=True, exist_ok=True)
        super().finalize_options()


def _strip_absolute_rpaths(command):
    """Remove interpreter/environment-specific RPATH flags from a linker."""

    if not isinstance(command, str):
        return command
    return re.sub(r"\s-Wl,-rpath(?:-link)?,/[^\s]+", "", command)


# Conda's Python stores its own prefix in LDSHARED/LDCXXSHARED.  Those flags
# are copied into every extension link command, so remove them before the
# compiler is created; changing the compiler object later is too late for
# setuptools' platform-specific linker setup.
for _linker_variable in ("LDSHARED", "LDCXXSHARED"):
    _linker_command = os.environ.get(_linker_variable)
    if _linker_command is None:
        _linker_command = sysconfig.get_config_var(_linker_variable)
    if _linker_command:
        os.environ[_linker_variable] = _strip_absolute_rpaths(_linker_command)


class build_ext(_build_ext):
    """Build an extension whose runtime search path is relocatable."""

    def build_extensions(self):
        # Conda's Python injects its own absolute prefix into LDSHARED.  If
        # left untouched, an editable build works only on the machine where it
        # was compiled.  The extension gets the portable cmeel RPATH below;
        # runtime.py supplies the same path for editable/source-tree runs.
        for name in ("linker_so", "linker_so_cxx"):
            command = getattr(self.compiler, name, None)
            if command:
                setattr(self.compiler, name, _strip_absolute_rpaths(command))
        super().build_extensions()


setup(
    ext_modules=[Pybind11Extension(
        "fairino_ik._native", ["cpp/src/solver.cpp", "cpp/src/bindings.cpp"],
        depends=["cpp/include/fairino_ik/solver.hpp", "cpp/include/fairino_ik/box_qp.hpp"],
        include_dirs=[str(ROOT / "cpp/include"), str(prefix / "include"), str(eigen)],
        library_dirs=[str(prefix / "lib")],
        # cmeel installs its shared libraries beside site-packages.  Keep the
        # wheel relocatable; editable installs are handled by runtime.py,
        # which discovers the active environment's cmeel.prefix at launch.
        runtime_library_dirs=["$ORIGIN/../cmeel.prefix/lib"],
        # Pinocchio's parser library carries its own urdfdom dependencies.
        # Linking those libraries directly can select a different ABI from
        # the one used to build libpinocchio_parsers.so.
        libraries=["pinocchio_default", "pinocchio_parsers"],
        cxx_std=17, extra_compile_args=["-O3"],
    )], cmdclass={"build_ext": build_ext, "egg_info": egg_info}, zip_safe=False,
)
