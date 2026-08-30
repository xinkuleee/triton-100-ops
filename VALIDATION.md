# 验证状态

当前仓库已完成结构与 Python 语法检查，但尚未在本机完成 Triton JIT、CUDA 编译、GPU 数值测试或性能实测。

## 验证级别

| 级别 | 当前状态 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| 文件结构 | 已检查 | 001–100 连续、每个编号各有一份 `.py` 和 `.cu` | kernel 可以编译或正确运行 |
| Python 语法 | 已检查 | Python 文件可以被语法编译 | Triton 后端可以生成设备代码 |
| 官方教程静态检查 | 已检查 | 001–011 文件、顶层符号和指定源码标记存在 | 教程已在 GPU 上执行或高级路径正确 |
| Triton GPU reference 测试 | 已编写，未在本机运行 | 在目标环境运行后可比较 012–100 与 PyTorch reference | 当前主机上的正确性 |
| CUDA 运行时测试 | 未提供 | — | CUDA 001–100 的数值正确性 |
| 性能 benchmark | 已提供入口，未在本机运行 | 在目标环境测量调用闭包的时间 | 当前项目的任何性能结论 |

## 已完成的静态检查

- Python 和 CUDA 编号均为 `001–100`，没有缺号或重复。
- `001–011` 的 Python 文件与固定上游提交 `694c0c3bd4da68600ed7028f1be65f72df05a56a` 一致。
- 静态源码检查确认教程集合中存在 Philox 随机数、autotune、LayerNorm/Attention backward、TensorDescriptor、`tl.dot_scaled` 和 PDL 相关标记。该检查只确认源码标记存在。
- `012–100` 的编号 Python 文件各自定义并直接启动本文件中的 Triton kernel。
- 编号 CUDA 文件各自包含本地 kernel 和直接 launch 语句。该项由文本检查完成，不等价于 CUDA 编译。
- 不再保留分类级中间转发实现或具有导入副作用的 `python/__init__.py`。
- 测试和 benchmark 按编号惰性加载 `012–100` 的实现。

运行静态检查：

```shell
python3 tests/test_catalog_static.py
python3 tests/test_01_official_tutorials.py
python3 benchmarks/run.py --list
```

安装 pytest 后，也可以运行：

```shell
python3 -m pytest -q tests/test_catalog_static.py tests/test_01_official_tutorials.py
```

## 当前环境限制

当前主机没有可用的 Triton 安装、`nvcc` 或 NVIDIA GPU。因此没有完成以下工作：

- Triton JIT 编译；
- CUDA 设备编译；
- `012–100` 的 GPU reference 测试；
- CUDA `001–100` 的运行时测试；
- GPU 性能 benchmark。

Python 语法通过不表示 Triton kernel 已在目标架构运行。静态正则检查通过也不表示 CUDA kernel 能编译、不会越界或不存在同步错误。

## 已知实现边界

- CUDA 003 是 FP32 CUDA Core GEMM，不是 Tensor Core GEMM。
- CUDA 008 只有普通 pointer grouped GEMM，没有 grouped TMA 路径。
- CUDA 009 有普通/持久 shared-memory GEMM 和 tensor-map 编码示例，没有完整 TMA 搬运、MMA 或 warp-specialized pipeline。
- CUDA 010 是 FP4、FP8 和 mixed 软件解码参考，不是原生 block-scaled MMA。
- CUDA `001–011` 没有独立的运行时测试或 benchmark harness。
- TMA 通常需要 Hopper 或更新架构以及相应工具链。官方 block-scaled 路径需要支持的 Blackwell 或 AMD CDNA4 目标。PDL 需要支持它的 CUDA 工具链和 compute capability 9.0 或更高设备。

## 在 GPU 主机上验证

先检查语法和结构：

```shell
PYTHONPYCACHEPREFIX=/tmp/triton-100-ops-pycache \
python3 -m py_compile categories/*/python/*.py tests/*.py benchmarks/*.py

python3 tests/test_catalog_static.py
python3 tests/test_01_official_tutorials.py
```

再运行 GPU reference 测试：

```shell
python3 -m pytest -q tests/test_02_elementwise.py \
  tests/test_03_reductions_normalization.py \
  tests/test_04_indexing_sparse.py \
  tests/test_05_linear_algebra.py \
  tests/test_06_vision.py \
  tests/test_07_transformer_inference.py \
  tests/test_08_quantization.py \
  tests/test_09_losses_optimizers.py
```

这些测试覆盖多种典型 shape、尾块、空输入、部分低精度类型和非法参数，但并非每个算子都覆盖所有 dtype、NaN/Inf、非连续输入或所有边界。

查看 Triton 编译阶段：

```shell
TRITON_ALWAYS_COMPILE=1 \
TRITON_KERNEL_DUMP=1 \
TRITON_DUMP_DIR=/tmp/triton-100-ops-dump \
python3 your_test.py
```
