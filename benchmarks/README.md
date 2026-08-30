# Benchmark 使用说明

统一 benchmark 入口只覆盖 `012–100`。它按编号加载对应 Python 文件，分别计时 Triton 调用闭包和 PyTorch reference 闭包。`001–011` 保留官方教程自己的示例、检查和性能入口。

## 运行命令

```shell
python3 benchmarks/run.py --list
python3 benchmarks/run.py --op 001
python3 benchmarks/run.py --op 042 --size medium
python3 benchmarks/run.py --op 092 --check
```

`--op 001` 不执行 benchmark，只打印官方教程文件的直接运行命令。`--size` 可取 `small`、`medium` 或 `large`。

## 实际输出

runner 使用 `triton.testing.do_bench` 的默认设置，因此输出平均 GPU 时间，而不是中位数。每次运行输出：

- Triton 调用闭包的平均时间，单位为 ms；
- PyTorch reference 调用闭包的平均时间，单位为 ms；
- `reference_time / triton_time`；
- 仅当 case 提供 `bytes_moved` 时，输出 Triton 侧的估算有效 GB/s。

runner 当前不输出 TFLOP/s、TOPS、token/s 或 profiler 指标，也没有命令行选项切换统计量。

## `--check` 做什么

`--check` 在计时前各执行一次 Triton 和 reference，然后递归比较 tensor、tuple 或 list。失败会终止 benchmark。它只适用于 `012–100`。

当前比较器对所有 tensor 使用 `rtol=2e-2` 和 `atol=2e-2`。这对整数结果并非严格相等检查，也可能对 FP32 过于宽松。因此 `--check` 是快速检查，不代替 `tests/` 中按算子设计的断言。

## 计时边界和限制

`do_bench` 使用设备 event，并在正式采样前调用待测函数、估算运行时间和 warmup。首次 JIT 通常在这些预备调用中触发，不进入正式采样；但预备调用和 JIT 仍属于整条 benchmark 命令的端到端开销。

计时对象是整个 Python 闭包，不一定是单个 kernel。部分 case 在闭包中执行输出分配、`clone()`、状态复位、CPU 元数据校验或设备到主机同步。因此：

- 结果不能统一解释为纯 kernel latency；
- in-place、scatter、KV cache、部分稀疏算子和优化器尤其需要检查闭包内容；
- 有效 GB/s 只是基于 case 提供的估算字节数；
- runner 不会自动跳过不支持的架构，导入、分配或 JIT 失败会直接报错。

`cases.py` 集中维护输入、Triton 调用、PyTorch reference 和可选字节估算。要做严谨性能研究，应把分配和同步移出计时区，明确端到端与纯 kernel 两种口径，并记录 GPU、driver、CUDA/ROCm、PyTorch 和 Triton 版本。
