"""Static checks for the complete upstream Triton tutorials 001--011.

The official files intentionally contain top-level examples, correctness checks,
plots, and benchmarks. Importing them here would execute that teaching workload,
so this suite parses their source without importing it. Run a numbered script
directly when its official runtime validation or benchmark is wanted.
"""

from __future__ import annotations

import ast
from pathlib import Path


DIRECTORY = Path(__file__).parents[1] / "categories" / "01_official_tutorials"
PYTHON = DIRECTORY / "python"

EXPECTED = {
    "001_vector_add.py": {"functions": {"add_kernel", "add", "benchmark"}},
    "002_fused_softmax.py": {
        "functions": {"naive_softmax", "softmax_kernel", "softmax", "benchmark"}
    },
    "003_matrix_multiplication.py": {
        "functions": {"matmul_kernel", "leaky_relu", "matmul", "benchmark"}
    },
    "004_low_memory_dropout.py": {
        "functions": {"_dropout", "dropout", "_seeded_dropout", "seeded_dropout"}
    },
    "005_layer_norm.py": {
        "functions": {
            "_layer_norm_fwd_fused", "_layer_norm_bwd_dx_fused",
            "_layer_norm_bwd_dwdb", "test_layer_norm", "bench_layer_norm",
        },
        "classes": {"LayerNorm"},
    },
    "006_fused_attention.py": {
        "functions": {
            "_attn_fwd", "_attn_bwd_preprocess", "_attn_bwd",
            "test_op", "bench_flash_attention",
        },
        "classes": {"_attention"},
    },
    "007_extern_functions.py": {"functions": {"asin_kernel"}},
    "008_grouped_gemm.py": {
        "functions": {
            "grouped_matmul_kernel", "group_gemm_fn",
            "grouped_matmul_tma_kernel", "group_gemm_tma_fn",
            "benchmark_square_matrices", "benchmark_batches",
        }
    },
    "009_persistent_matmul.py": {
        "functions": {
            "matmul_kernel", "matmul_kernel_tma", "matmul_kernel_persistent",
            "matmul_kernel_tma_persistent", "matmul_kernel_descriptor_persistent",
            "validate", "bench",
        }
    },
    "010_block_scaled_matmul.py": {
        "functions": {
            "block_scaled_matmul_kernel", "block_scaled_matmul",
            "validate_block_scaled", "bench_block_scaled",
            "block_scaled_matmul_kernel_cdna4", "block_scaled_matmul_amd",
        }
    },
    "011_programmatic_dependent_launch.py": {
        "functions": {"add_kernel", "add", "validate", "benchmark"}
    },
}


def _top_level_symbols(path: Path):
    tree = ast.parse(path.read_text())
    functions = {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    return tree, functions, classes


def test_official_tutorial_files_and_real_symbols():
    assert {path.name for path in PYTHON.glob("[0-9][0-9][0-9]_*.py")} == set(EXPECTED)
    for filename, expected in EXPECTED.items():
        tree, functions, classes = _top_level_symbols(PYTHON / filename)
        assert expected.get("functions", set()) <= functions, filename
        assert expected.get("classes", set()) <= classes, filename
        assert not any(name.startswith("op") for name in functions), filename
        assert ast.get_docstring(tree), filename


def test_official_tutorials_keep_kernel_test_and_benchmark_mechanisms():
    source = "\n".join((PYTHON / filename).read_text() for filename in EXPECTED)
    for token in (
        "@triton.jit", "@triton.autotune", "@triton.testing.perf_report",
        "tl.rand", "libdevice.asin", "class LayerNorm(torch.autograd.Function)",
        "def _attn_bwd", "TensorDescriptor", "tl.dot_scaled", "gdc_wait",
        "gdc_launch_dependents", "launch_pdl",
    ):
        assert token in source


def test_official_runtime_validation_is_owned_by_the_scripts():
    sources = {filename: (PYTHON / filename).read_text() for filename in EXPECTED}
    for filename in ("001_vector_add.py", "002_fused_softmax.py", "003_matrix_multiplication.py"):
        assert "benchmark.run" in sources[filename]
    assert "test_layer_norm(" in sources["005_layer_norm.py"]
    for filename in (
        "006_fused_attention.py", "009_persistent_matmul.py",
        "010_block_scaled_matmul.py", "011_programmatic_dependent_launch.py",
    ):
        assert "if __name__ == \"__main__\":" in sources[filename] or (
            "if __name__ == '__main__':" in sources[filename]
        )


def test_cuda_counterparts_are_separate_numbered_sources():
    cuda_files = sorted((DIRECTORY / "cuda").glob("[0-9][0-9][0-9]_*.cu"))
    assert [path.name[:3] for path in cuda_files] == [f"{number:03d}" for number in range(1, 12)]
    for path in cuda_files:
        assert f"op{path.name[:3]}_" in path.read_text()


if __name__ == "__main__":
    for name, test in sorted(globals().copy().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"{name}: PASS")
