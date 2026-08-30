# 100 个 GPU 算子：CUDA 与 Triton 对照教材

本项目用 100 个连续编号的算子讲解 GPU 编程。每个编号对应一份 Python/Triton 源码和一份 CUDA 源码。`001–011` 的 Python 文件是固定版本的 Triton 官方完整教程；`012–100` 是本项目直接实现的教学算子。

## 从哪里开始

如果还不了解 GPU，请按以下顺序阅读：

1. 阅读 [GPU、CUDA 与 Triton 基础](docs/00-gpu-cuda-triton-foundations.md)，建立硬件和编程模型。
2. 阅读 [官方教程 001–011](categories/01_official_tutorials/README.md)，学习 Triton 的核心构造。
3. 按领域学习 [012–100](categories/README.md)，对照 CUDA 和 Triton 实现。
4. 查看 [完整算子清单](MANIFEST.md)，按编号查找算子。
5. 查看 [验证状态](VALIDATION.md)，区分静态检查、GPU 测试和未验证内容。

## 教程、概念、实现和验证

| 内容 | 位置 | 作用 |
|---|---|---|
| 概念 | `docs/` | 解释 GPU、CUDA、Triton、MLIR 和性能基础 |
| 教程 | `categories/01_official_tutorials/` | 保存 Triton 官方 11 个完整教程和 CUDA 对照 |
| 实现 | `categories/02_*` 到 `categories/09_*` | 保存 012–100 的分类源码和教学文档 |
| 验证 | `tests/`、`benchmarks/`、`VALIDATION.md` | 静态检查、GPU reference 测试和性能入口 |

## 源码结构

```text
triton-100-ops/
├── README.md
├── MANIFEST.md
├── VALIDATION.md
├── docs/
│   └── 00-gpu-cuda-triton-foundations.md
├── categories/
│   ├── 01_official_tutorials/       # 001–011
│   ├── 02_elementwise/              # 012–033
│   ├── 03_reductions_normalization/ # 034–046
│   ├── 04_indexing_sparse/          # 047–056
│   ├── 05_linear_algebra/           # 057–060
│   ├── 06_vision/                   # 061–068
│   ├── 07_transformer_inference/    # 069–081
│   ├── 08_quantization/             # 082–088
│   └── 09_losses_optimizers/        # 089–100
├── shared/
│   └── cuda_common.cuh
├── tests/
└── benchmarks/
```

每个分类包含 `python/`、`cuda/` 和一个集中教学 `README.md`。每个编号只有一份编号 Python 文件和一份编号 CUDA 文件。分类内公共代码放在可选的 `_common.py` 或 `_common.cuh` 中；跨分类 CUDA reduction helper 位于 `shared/cuda_common.cuh`。

## 运行前准备

固定的官方教程来自 Triton 提交 `694c0c3bd4da68600ed7028f1be65f72df05a56a`，该源码标识为 Triton 3.8.0，并要求 CPython 3.10–3.14（`>=3.10,<3.15`）。要运行完整教程，还需要与该版本兼容的 PyTorch、Triton、NVIDIA CUDA 或 AMD ROCm 环境。

官方教程的附加 Python 依赖包括 `pytest`、`matplotlib`、`pandas` 和 `tabulate`。部分教程还要求特定 GPU 架构，例如 Hopper、Blackwell 或 AMD CDNA4。项目没有提供锁定依赖的安装文件，因此安装组合需要与目标 Triton 版本和硬件匹配。

## 最短可运行路径

不需要 GPU 的静态检查：

```shell
python3 tests/test_catalog_static.py
python3 tests/test_01_official_tutorials.py
python3 benchmarks/run.py --list
```

在已正确安装 Triton 和 GPU 运行时的环境中：

```shell
python3 categories/01_official_tutorials/python/001_vector_add.py
python3 benchmarks/run.py --op 042 --check
```

第一条命令直接执行官方教程。第二条命令先比较 Triton 结果与 PyTorch reference，再分别计时。统一 benchmark 入口只执行 `012–100`；传入 `001–011` 时只打印对应教程的直接运行命令。

## 实现边界

本项目用于学习机制，不等同于 cuBLAS、cuDNN、CUTLASS 或生产级 FlashAttention。`001–011` 的 Triton 侧与固定官方源码逐字一致。CUDA 侧是独立教学实现，其中 003、008、009 和 010 尚未完整覆盖官方教程中的 Tensor Core、TMA、warp specialization 和原生 block-scaled 指令路径。详情见 [验证状态](VALIDATION.md)。
