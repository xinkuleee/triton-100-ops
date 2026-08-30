"""GPU-independent checks for the one-file-per-number source layout."""

from __future__ import annotations

import ast
from pathlib import Path
import re


ROOT = Path(__file__).parents[1]
CATEGORIES = ROOT / "categories"
EXPECTED = {
    "01_official_tutorials": range(1, 12),
    "02_elementwise": range(12, 34),
    "03_reductions_normalization": range(34, 47),
    "04_indexing_sparse": range(47, 57),
    "05_linear_algebra": range(57, 61),
    "06_vision": range(61, 69),
    "07_transformer_inference": range(69, 82),
    "08_quantization": range(82, 89),
    "09_losses_optimizers": range(89, 101),
}


def _numbered_files(directory: Path, language: str, suffix: str):
    return sorted((directory / language).glob(f"[0-9][0-9][0-9]_*{suffix}"))


def _public_slots(path: Path):
    slots = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            match = re.match(r"op(\d{3})_", node.name)
            if match:
                slots.append(int(match.group(1)))
    return slots


def _is_triton_jit(decorator: ast.expr) -> bool:
    return (
        isinstance(decorator, ast.Attribute)
        and isinstance(decorator.value, ast.Name)
        and decorator.value.id == "triton"
        and decorator.attr == "jit"
    )


def _direct_kernel_launches(function: ast.FunctionDef):
    """Return names used by ``kernel[grid](...)`` calls in one function."""
    launches = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Subscript):
            continue
        kernel = node.func.value
        if isinstance(kernel, ast.Name):
            launches.append(kernel.id)
    return launches


def test_numbered_python_and_cuda_sources_cover_001_to_100_once():
    python_numbers = []
    cuda_numbers = []
    for name, expected_range in EXPECTED.items():
        directory = CATEGORIES / name
        assert (directory / "README.md").is_file()
        python_files = _numbered_files(directory, "python", ".py")
        cuda_files = _numbered_files(directory, "cuda", ".cu")
        expected = list(expected_range)
        assert [int(path.name[:3]) for path in python_files] == expected
        assert [int(path.name[:3]) for path in cuda_files] == expected
        for path in cuda_files:
            assert f"op{path.name[:3]}_" in path.read_text(), path
        python_numbers.extend(int(path.name[:3]) for path in python_files)
        cuda_numbers.extend(int(path.name[:3]) for path in cuda_files)
    assert python_numbers == list(range(1, 101))
    assert cuda_numbers == list(range(1, 101))
    assert (ROOT / "shared" / "cuda_common.cuh").is_file()


def test_012_to_100_each_define_one_matching_public_function():
    for name, expected_range in list(EXPECTED.items())[1:]:
        directory = CATEGORIES / name
        for path in _numbered_files(directory, "python", ".py"):
            number = int(path.name[:3])
            assert number in expected_range
            assert _public_slots(path) == [number], path


def test_numbered_sources_own_and_directly_launch_their_kernels():
    """Reject numbered files that merely forward to another implementation."""
    for name, expected_range in EXPECTED.items():
        directory = CATEGORIES / name
        for cuda_path in _numbered_files(directory, "cuda", ".cu"):
            number = int(cuda_path.name[:3])
            source = cuda_path.read_text()
            kernels = re.findall(
                rf"__global__\s+void\s+(op{number:03d}_[A-Za-z0-9_]*kernel)\s*\(",
                source,
            )
            assert kernels, f"{cuda_path} must define its own numbered CUDA kernel"
            assert any(
                re.search(rf"\b{re.escape(kernel)}\s*<<<", source)
                or re.search(
                    rf"cudaLaunchKernelEx\s*\([^;]*\b{re.escape(kernel)}\b",
                    source,
                    re.DOTALL,
                )
                for kernel in kernels
            ), f"{cuda_path} must directly launch one of its local CUDA kernels"
            assert not re.search(r'#\s*include\s+["<][^">]*\.cu[">]', source)

        if name == "01_official_tutorials":
            continue
        for python_path in _numbered_files(directory, "python", ".py"):
            number = int(python_path.name[:3])
            tree = ast.parse(python_path.read_text())
            kernels = {
                node.name
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and any(_is_triton_jit(item) for item in node.decorator_list)
                and node.name.startswith(f"_op{number:03d}_")
                and node.name.endswith("_kernel")
            }
            public = [
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name.startswith(f"op{number:03d}_")
            ]
            assert kernels, f"{python_path} must define its own numbered Triton kernel"
            assert len(public) == 1
            assert kernels.intersection(_direct_kernel_launches(public[0])), (
                f"{python_path} public entry must directly launch its local Triton kernel"
            )


def test_no_category_level_implementation_or_package_side_effect_loader_remains():
    for directory in (CATEGORIES / name for name in EXPECTED):
        assert not (directory / "triton_ops.py").exists()
        assert not (directory / "cuda_ops.cu").exists()
        assert not (directory / "python" / "__init__.py").exists()
    source = "\n".join(
        path.read_text()
        for path in (ROOT / "tests").glob("*.py")
        if path != Path(__file__)
    ) + (ROOT / "benchmarks" / "cases.py").read_text()
    assert "triton_ops.py" not in source
    assert "cuda_ops.cu" not in source


def test_official_tutorial_provenance_and_advanced_mechanisms():
    directory = CATEGORIES / "01_official_tutorials"
    assert (directory / "LICENSE").is_file()
    readme = (directory / "README.md").read_text()
    assert "694c0c3bd4da68600ed7028f1be65f72df05a56a" in readme
    source = "\n".join(
        path.read_text() for path in _numbered_files(directory, "python", ".py")
    )
    for token in (
        "tl.rand", "libdevice.asin", "@triton.autotune",
        "class LayerNorm(torch.autograd.Function)", "def _attn_bwd",
        "TensorDescriptor", "tl.dot_scaled", "gdc_wait",
        "gdc_launch_dependents", "launch_pdl",
    ):
        assert token in source


def test_retained_distribution_and_loss_ops():
    manifest = (ROOT / "MANIFEST.md").read_text()
    assert "042|Row LogSoftmax" in manifest
    assert "091|Cross Entropy" in manifest
    assert "093|KL Divergence" in manifest
    numbers = [int(value) for value in re.findall(r"\|(\d{3})\|", manifest)]
    assert sorted(numbers) == list(range(1, 101))


def test_012_to_100_are_named_in_tests_and_unified_benchmarks():
    test_source = "\n".join(
        path.read_text() for path in (ROOT / "tests").glob("test_*.py")
    )
    for number in range(12, 101):
        assert f"op{number:03d}_" in test_source

    cases_path = ROOT / "benchmarks" / "cases.py"
    cases_source = cases_path.read_text()
    assert "triton_ops.py" not in cases_source
    cases_tree = ast.parse(cases_source)
    assignments = [
        node for node in cases_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SUPPORTED_CASES"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    expression = ast.Expression(assignments[0].value)
    compiled = compile(expression, str(cases_path), "eval")
    assert eval(compiled, {"__builtins__": {}}, {"set": set, "range": range}) == set(
        range(12, 101)
    )


if __name__ == "__main__":
    for name, test in sorted(globals().copy().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"{name}: PASS")
