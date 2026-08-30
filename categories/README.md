# 算子分类

本目录按算法领域组织 100 个算子。一个领域共用一个目录和一份集中教学文档，不为每个算子创建单独文件夹。

| 目录 | 编号 | 主题 |
|---|---:|---|
| `01_official_tutorials` | 001–011 | Triton 官方教程 |
| `02_elementwise` | 012–033 | 逐元素、激活和广播 |
| `03_reductions_normalization` | 034–046 | 归约、scan 和归一化 |
| `04_indexing_sparse` | 047–056 | 索引、scatter 和稀疏计算 |
| `05_linear_algebra` | 057–060 | 矩阵和向量运算 |
| `06_vision` | 061–068 | 卷积、池化和 resize |
| `07_transformer_inference` | 069–081 | Attention、KV cache 和解码 |
| `08_quantization` | 082–088 | INT8 和 blockwise 量化 |
| `09_losses_optimizers` | 089–100 | 损失、相似度和优化器更新 |

## 每个分类包含什么

- `python/NNN_name.py`：编号算子的 Triton kernel 和 Python 调用入口。001–011 例外，它们保留官方真实符号。
- `cuda/NNN_name.cu`：独立 CUDA kernel 和 launcher。
- `README.md`：公式、代码映射、硬件行为、数值边界、验证和练习。

`012–100` 的分类可以按需提供 `python/_common.py` 或 `cuda/_common.cuh`。跨分类使用的 CUDA reduction helper 位于 `../shared/cuda_common.cuh`。测试位于 `../tests/`。

## 推荐学习顺序

先学习逐元素算子，再学习需要 CUDA thread 协作的归约。之后进入索引、矩阵计算、视觉和 Transformer。最后学习量化、损失与原地优化器状态更新。完整编号见 [MANIFEST.md](../MANIFEST.md)。
