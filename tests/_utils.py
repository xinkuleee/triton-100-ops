"""Shared helpers for GPU correctness tests."""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
import re
import sys
import types


ROOT = Path(__file__).parents[1]
_NUMBERED_ATTRIBUTE = re.compile(r"^op(?P<number>\d{3})_")
_PACKAGES: dict[str, types.ModuleType] = {}
_MODULES: dict[Path, types.ModuleType] = {}


def _numbered_files(directory: str) -> dict[int, Path]:
    python_directory = ROOT / "categories" / directory / "python"
    files: dict[int, Path] = {}
    for path in sorted(python_directory.glob("[0-9][0-9][0-9]_*.py")):
        number = int(path.name[:3])
        if number in files:
            raise RuntimeError(f"duplicate operator {number:03d} in {python_directory}")
        files[number] = path
    return files


def _load_numbered_module(directory: str, path: Path) -> types.ModuleType:
    cached = _MODULES.get(path)
    if cached is not None:
        return cached

    package_name = f"_triton_100_ops_test_{directory}"
    package = _PACKAGES.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__package__ = package_name
        package.__path__ = [str(path.parent)]
        package.__spec__ = importlib.machinery.ModuleSpec(
            package_name, loader=None, is_package=True
        )
        sys.modules[package_name] = package
        _PACKAGES[package_name] = package

    module_name = f"{package_name}.{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load numbered implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    _MODULES[path] = module
    return module


class _DirectOperatorNamespace:
    """Resolve ``opNNN_*`` from its numbered source file on first use."""

    def __init__(self, directory: str):
        self._directory = directory
        self._files = _numbered_files(directory)

    def __getattr__(self, name: str):
        match = _NUMBERED_ATTRIBUTE.match(name)
        if match is None:
            raise AttributeError(name)
        number = int(match.group("number"))
        path = self._files.get(number)
        if path is None:
            raise AttributeError(name)
        module = _load_numbered_module(self._directory, path)
        try:
            value = getattr(module, name)
        except AttributeError as error:
            raise AttributeError(
                f"{path.name} does not define {name}; tests must use the module's "
                "real public symbol"
            ) from error
        setattr(self, name, value)
        return value


def load_category(directory: str):
    if directory == "01_official_tutorials":
        raise RuntimeError(
            "tutorials 001-011 are complete upstream scripts with top-level examples "
            "and optional checks/benchmarks; inspect them without importing, or run "
            "the numbered script directly for its original runtime behavior"
        )
    return _DirectOperatorNamespace(directory)


def require_gpu(pytest, torch):
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("GPU correctness tests require CUDA or ROCm", allow_module_level=True)


def assert_close(torch, actual, expected, *, dtype=None):
    dtype = dtype or actual.dtype
    if dtype in (torch.float16, torch.bfloat16):
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
