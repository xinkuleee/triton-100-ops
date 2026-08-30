# 归约、归一化与前缀和（034–046）

本目录用 13 个算子说明 GPU 如何归并一行数据、用统计量归一化数据，以及生成前缀和。Python 文件包含 Triton 实现和输入检查。CUDA 文件是独立的 FP32 对照实现。这些代码用于学习执行模型，不是生产级数值库。

## 基础概念

归约（reduction）把一组输入合并成较少的输出。例如，行求和把 `x[rows, cols]` 的每行变成一个标量：

$$
y_r=\sum_{c=0}^{C-1}x_{r,c}.
$$

归一化先归约出均值、方差或均方，再把统计量应用到原数据。前缀和不把一行压成标量；第 $c$ 个输出包含当前位置及此前的输入：

$$
y_{r,c}=\sum_{j=0}^{c}x_{r,j}.
$$

本文固定使用以下术语：

- **Tensor**：有 shape（各维长度）、dtype（元素类型）和存储的多维数据。
- **GPU kernel**：在 GPU 上执行的设备函数；launcher 是 Host 上配置并启动 CUDA kernel 的函数，wrapper 是 Host 上验证并启动 Triton kernel 的函数。
- **CUDA thread**：执行 GPU kernel 指令流的工作单元。
- **warp**：CUDA 中共同执行的 32 个 CUDA thread；lane 是其中一个 CUDA thread 的物理位置。
- **CUDA thread block（CTA）**：由多个 warp 组成。
- **CUDA grid**：一次 CUDA kernel launch 创建的全部 CUDA thread block（CTA）。
- **Triton program instance**：Triton launch grid 中一个独立逻辑程序实例。
- **Triton block tensor**：一个 Triton program instance 同时表达和计算的一组逻辑值。
- **Shared memory**：同一 CUDA thread block（CTA）共享的片上存储。
- **归约单位元**：不改变结果的填充值。求和用 `0`，最大值用 `-inf`，最小值用 `+inf`。
- **尾部掩码**：阻止无效逻辑元素或 CUDA thread 访问行尾之外的内存。
- **FP32 累加**：把 FP16 或 BF16 输入转为 FP32 后计算和或统计量。
- **inclusive scan**：输出包含当前位置输入的前缀扫描。046 实现这种语义。

### 文件和公共 helper

每个编号都有两个真实源码文件。例如，034 对应 [python/034_row_sum.py](python/034_row_sum.py) 和 [cuda/034_row_sum.cu](cuda/034_row_sum.cu)。其他文件遵循相同的 `python/NNN_name.py` 和 `cuda/NNN_name.cu` 命名。

[python/_common.py](python/_common.py) 检查 Triton 输入，并选择 `BLOCK` 和 warp 数。[cuda/_common.cuh](cuda/_common.cuh) 检查 CUDA 矩阵维度。CUDA 的 `warp_sum`、`warp_max`、`block_sum` 和 `block_max` 实际定义在仓库级 [shared/cuda_common.cuh](../../shared/cuda_common.cuh)；分类 helper 包含该文件。

Python 接口接受连续的 CUDA 或 ROCm tensor，dtype 为 FP16、BF16 或 FP32。CUDA launcher 接受裸 FP32 指针。两套接口的 dtype 能力不同。

## 归约的硬件模型

### CUDA：一个 CUDA thread block（CTA）处理一行

034–043 的 CUDA 实现通常启动 `rows` 个 CUDA thread block（CTA），每个包含 256 个 CUDA thread。每个 CUDA thread 跨步读取一行并生成局部结果：

```cuda
for (int col = threadIdx.x; col < cols; col += blockDim.x)
  local += x[blockIdx.x * cols + col];
```

一个 warp 用 shuffle 合并局部结果。每个 warp 的 lane 0 把部分结果写入 shared memory。第一个 warp 再合并这些结果。树形加法与 CPU 从左到右加法的舍入顺序不同。

### Triton：一个 Triton program instance 处理一个逻辑单位

034–043 和 046 通常为每行启动一个 Triton program instance。该 Triton program instance 创建 `[0, BLOCK)` offsets；`BLOCK` 是不小于 `cols` 的最小二次幂。无效位置使用尾部掩码和单位元。Triton 编译器负责 lowering `tl.sum`、`tl.max`、`tl.min` 和 `tl.cumsum`。

Python helper 把单次归约宽度限制为 65,536 个元素。这个限制只属于 Python/Triton wrapper。CUDA 的 `validate_matrix` 没有该限制；它检查 `rows >= 0`、`cols > 0` 和 `rows * cols <= INT_MAX`。

### BatchNorm Inference 是逐元素映射

044 不计算统计量。它读取给定的 `running_mean` 和 `running_variance`。每个 CUDA thread 处理一个扁平元素；每个 Triton program instance 处理最多 256 个连续元素：

$$
c=i\bmod C,\qquad y_i=(x_i-\mu_c)\frac{\gamma_c}{\sqrt{v_c+\epsilon}}+\beta_c.
$$

因此 044 没有行归约、warp shuffle 或 CUDA thread block（CTA）内统计阶段。

### 共同边界

- Triton 的 `tl.program_id(0)` 和多个地址表达式使用 int32。Python validation 没检查总元素数是否超过 `INT_MAX`。大 tensor 可能使 `row * cols` 或扁平 offset 回绕。
- CUDA helper 允许 `cols == INT_MAX`。多个 kernel 的 int32 循环变量执行 `col += 256` 时可能溢出。
- 65,536 是 Python 接口检查，不保证所有 dtype、GPU 和 Triton 版本都能以可接受资源成功 JIT 和启动。
- 当前测试主要使用普通有限随机数，不能证明 NaN、Inf、极端幅值和超大地址正确。

## 算子指南

### 034 Row Sum

源码：[Triton](python/034_row_sum.py) · [CUDA](cuda/034_row_sum.cu)

#### 概念与用途

Row Sum 把矩阵的每行压缩成一个数。它是平均值、范数和许多损失聚合的基础。

#### 输入输出

输入形状为 `[rows, cols]`，输出形状为 `[rows]`。Triton 接受 FP16、BF16 或 FP32，输出 FP32；CUDA 接受并输出 FP32。

#### 算法

把一行拆成 $T$ 组。第 $t$ 组先计算 $s_t=\sum_kx_{t+kT}$，再把所有 $s_t$ 相加。分组不会改变数学公式，但会改变浮点舍入顺序。

#### Triton 实现

```python
row = tl.program_id(0)
offsets = tl.arange(0, BLOCK)
values = tl.load(x_ptr + row * cols + offsets, mask=offsets < cols, other=0.0).to(tl.float32)
tl.store(out_ptr + row, tl.sum(values, axis=0))
```

`program_id(0)` 选择行。`tl.arange` 生成列号。masked load 对越界逻辑元素填 `0.0`，随后转成 FP32。`tl.sum` 归约 Triton block tensor，最后只写一个行结果。wrapper 为每行启动一个 Triton program instance；零行时不启动 kernel。

#### CUDA 实现

```cuda
for (int col = threadIdx.x; col < cols; col += blockDim.x)
  local += x[blockIdx.x * cols + col];
const float result = block_sum(local);
if (threadIdx.x == 0) out[blockIdx.x] = result;
```

每个 CUDA thread 以 `blockDim.x` 为步长累加。`block_sum` 先做 warp shuffle，再经 shared-memory partial 合并 warp。CUDA thread 0 把最终值写到 `out[blockIdx.x]`。launcher 先验证矩阵，再启动包含 256 个 CUDA thread 的 CUDA thread block（CTA）。

#### GPU 硬件

一个 CUDA thread block（CTA）读取一条连续行。相邻 CUDA thread 首先访问相邻列，便于合并全局内存事务。归约后只有一个 CUDA thread 写出，因此输出流量远小于输入流量。

#### 教程

依次输入全 1、正负交替、257 列和零行，检查总和、尾部掩码和空输出。

#### 验证与限制

NaN 通过加法传播。大幅值可能溢出，抵消会损失精度。超大 tensor 还受本章“共同边界”中说明的 int32 地址限制。

### 035 Row Mean

源码：[Triton](python/035_row_mean.py) · [CUDA](cuda/035_row_mean.cu)

#### 概念与用途

Row Mean 计算矩阵每行的算术平均值。

#### 输入输出

输入形状为 `[rows, cols]`，输出形状为 `[rows]`。Triton 输入支持三种浮点 dtype，输出 FP32；CUDA 是 FP32 接口。

#### 算法

$$
\mu_r=\frac{1}{C}\sum_cx_{r,c}.
$$

先计算 034 的行和，再除以真实列数 $C$。padding 的零只参与存储布局，不计入除数。

#### Triton 实现

```python
values = tl.load(x_ptr + row * cols + offsets, mask=offsets < cols, other=0.0).to(tl.float32)
tl.store(out_ptr + row, tl.sum(values, axis=0) / cols)
```

load、mask 和 FP32 转换与 034 相同。kernel 在 `tl.sum` 后除以运行时参数 `cols`。wrapper 的 `_require_matrix` 拒绝零列，避免除零。

#### CUDA 实现

```cuda
const float result = block_sum(local);
if (threadIdx.x == 0) out[blockIdx.x] = result / cols;
```

CUDA thread 局部和由 `block_sum` 合并。只有 CUDA thread 0 执行最终除法和写回。launcher 与 034 使用相同矩阵验证和 CUDA grid。

#### GPU 硬件

内存访问和归约树与 034 相同。每行只多一次标量除法，不能据此推断实际时间；需要在目标 GPU 上测量。

#### 教程

先用常量行验证均值等于常量，再用单列输入验证输出等于输入。

#### 验证与限制

实现没有补偿求和；均值继承行和的溢出、抵消和舍入误差。

### 036 Row Maximum

源码：[Triton](python/036_row_max.py) · [CUDA](cuda/036_row_max.cu)

#### 概念与用途

Row Maximum 计算每行最大值。它常用于稳定 softmax。

#### 输入输出

输入形状为 `[rows, cols]`，输出形状为 `[rows]`。Triton 输出 dtype 与输入相同，CUDA 为 FP32。

#### 算法

每个 CUDA thread 先找所负责元素的最大值，再合并所有 CUDA thread 的结果。最大值的单位元是 `-inf`，因为 `max(x, -inf) = x`。

#### Triton 实现

```python
values = tl.load(x_ptr + row * cols + offsets, mask=offsets < cols, other=-float("inf"))
tl.store(out_ptr + row, tl.max(values, axis=0))
```

masked load 把无效逻辑元素填为 `-inf`。`tl.max(values, axis=0)` 归约整行。wrapper 创建同 dtype 的一维输出。

#### CUDA 实现

```cuda
local = fmaxf(local, x[blockIdx.x * cols + col]);
const float result = block_max(local);
if (threadIdx.x == 0) out[blockIdx.x] = result;
```

CUDA thread 用 `fmaxf` 更新局部最大值。仓库级 `block_max` 先做 warp 归约，再合并 shared partial。CUDA thread 0 写回。

#### GPU 硬件

读流与 034 相同，但归约操作从加法变为比较选择。每个 CUDA thread block（CTA）仍独立处理一行，无跨 CUDA thread block（CTA）通信。

#### 教程

依次测试全负行、重复最大值、`-inf` 和 257 列，检查最大值和尾部掩码。

#### 验证与限制

当前 `fmaxf` 与 `tl.max` 忽略 NaN，而测试 reference `torch.max` 传播 NaN；含 NaN 输入存在已知语义差异。

### 037 Row Minimum

源码：[Triton](python/037_row_min.py) · [CUDA](cuda/037_row_min.cu)

#### 概念与用途

Row Minimum 计算每行最小值。

#### 输入输出

输入形状为 `[rows, cols]`，输出形状为 `[rows]`。dtype 规则与 036 相同：Triton 输出 dtype 与输入相同，CUDA 为 FP32。

#### 算法

算法把最大值比较换成最小值比较。单位元是 `+inf`，因为 `min(x, +inf) = x`。

#### Triton 实现

```python
values = tl.load(x_ptr + row * cols + offsets, mask=offsets < cols, other=float("inf"))
tl.store(out_ptr + row, tl.min(values, axis=0))
```

无效逻辑元素载入 `+inf`。`tl.min` 归约整行，wrapper 为每行启动一个 Triton program instance。

#### CUDA 实现

```cuda
value = fminf(value, __shfl_down_sync(0xffffffffu, value, offset));
if (lane == 0) partial[warp] = value;
__syncthreads();
```

该编号文件定义 `warp_min_local` 和 `block_min_local`。warp 使用 shuffle 和 `fminf`，lane 0 写 shared partial，第一个 warp 完成第二级归约。

#### GPU 硬件

一个 CUDA thread block（CTA）对应一行。两级归约只在 CUDA thread block（CTA）内同步，不需要原子操作或全局临时数组。

#### 教程

依次测试全正行、重复最小值、`+inf` 和非二次幂列数，检查最小值和尾部掩码。

#### 验证与限制

CUDA `fminf` 与 Triton `tl.min` 忽略 NaN，但 `torch.min` reference 传播 NaN。

### 038 Row L1 Norm

源码：[Triton](python/038_l1_norm.py) · [CUDA](cuda/038_l1_norm.cu)

#### 概念与用途

Row L1 Norm 衡量一行元素绝对值的总量，常用于距离和稀疏正则。

#### 输入输出

输入形状为 `[rows, cols]`，输出形状为 `[rows]`。Triton 和 CUDA 都返回 FP32。

#### 算法

$$
\lVert x_r\rVert_1=\sum_c|x_{r,c}|.
$$

绝对值把正负输入变成非负数，随后问题退化为 034 的行求和。

#### Triton 实现

```python
values = tl.load(x_ptr + row * cols + offsets, mask=offsets < cols, other=0.0).to(tl.float32)
tl.store(out_ptr + row, tl.sum(tl.abs(values), axis=0))
```

masked load 以零填尾，并转 FP32。`tl.abs` 逐元素取绝对值，`tl.sum` 完成归约。

#### CUDA 实现

```cuda
local += fabsf(x[blockIdx.x * cols + col]);
const float result = block_sum(local);
```

每个 CUDA thread 用 `fabsf` 后累加，`block_sum` 合并结果，CUDA thread 0 写回。

#### GPU 硬件

绝对值是归约前的逐元素操作，可与 load 和局部累加融合，不产生中间绝对值 tensor。

#### 教程

先用正负镜像输入验证符号不影响结果，再测试零、Inf 和 NaN。

#### 验证与限制

NaN 传播；非负项不断累加，大行仍可能 FP32 溢出。

### 039 Row L2 Norm

源码：[Triton](python/039_l2_norm.py) · [CUDA](cuda/039_l2_norm.cu)

#### 概念与用途

Row L2 Norm 是一行在欧几里得空间中的长度，常用于距离计算和向量归一化。

#### 输入输出

输入形状为 `[rows, cols]`，输出形状为 `[rows]`。Triton 和 CUDA 都返回 FP32。

#### 算法

$$
\lVert x_r\rVert_2=\sqrt{\sum_cx_{r,c}^2}.
$$

先平方消除符号，再归约平方和，最后只对行标量开平方。

#### Triton 实现

```python
values = tl.load(x_ptr + row * cols + offsets, mask=offsets < cols, other=0.0).to(tl.float32)
tl.store(out_ptr + row, tl.sqrt(tl.sum(values * values, axis=0)))
```

输入转 FP32 后在寄存器表达式中计算 `values * values`。`tl.sum` 归约，`tl.sqrt` 处理最终标量。

#### CUDA 实现

```cuda
const float value = x[blockIdx.x * cols + col];
local += value * value;
const float result = block_sum(local);
if (threadIdx.x == 0) out[blockIdx.x] = sqrtf(result);
```

每个 CUDA thread 读取值并累加 `value * value`。`block_sum` 合并平方和，CUDA thread 0 调用 `sqrtf`。

#### GPU 硬件

平方与局部累加融合。开平方只执行每行一次，主体仍是读流和归约。

#### 教程

先用勾股数组 `[3, 4]` 验证结果为 5，再测试极大数和极小数。

#### 验证与限制

实现没有缩放范数算法；平方可能溢出或下溢。

### 040 Row Variance

源码：[Triton](python/040_variance.py) · [CUDA](cuda/040_variance.cu)

#### 概念与用途

Row Variance 描述一行围绕均值的离散程度。它为归一化等算法提供统计量。该算子计算总体方差，分母是 $C$ 而不是 $C-1$。

#### 输入输出

输入形状为 `[rows, cols]`，输出形状为 `[rows]`。Triton 和 CUDA 都输出 FP32。

#### 算法

$$
v_r=\frac{1}{C}\sum_c(x_{r,c}-\mu_r)^2.
$$

两遍算法先求均值，再读取同一行求平方差。padding 逻辑元素的中心化值必须置零，否则它们会把 `0 - mean` 计入方差。

#### Triton 实现

```python
mean = tl.sum(values, axis=0) / cols
centered = tl.where(mask, values - mean, 0.0)
tl.store(out_ptr + row, tl.sum(centered * centered, axis=0) / cols)
```

kernel 生成 `mask`，载入并转 FP32。第一次 `tl.sum` 得均值。`tl.where(mask, values - mean, 0.0)` 排除 padding，第二次 `tl.sum` 得平方差和。

#### CUDA 实现

```cuda
const float sum = block_sum(local);
if (threadIdx.x == 0) mean = sum / cols;
__syncthreads();
local += delta * delta;
```

第一次 `block_sum` 后，CUDA thread 0 把均值写入 shared scalar。barrier 让所有 CUDA thread 看到均值。CUDA thread 再次遍历行并归约平方差。公共 helper 内的保护 barrier 允许连续复用 shared partial。

#### GPU 硬件

一个 CUDA thread block（CTA）两次读取同一行。第二遍可能从 cache 受益，但本仓库没有 profiler 证据。CUDA thread block（CTA）内两个归约阶段需要同步。

#### 教程

先验证常量行和单列输入都得到 0，再用“大常量 + 小扰动”比较 `torch.var(correction=0)`。

#### 验证与限制

CUDA 朴素 FP32 均值可产生灾难性抵消；实现没有 Welford 或更高精度累加。

### 041 Row Argmax

源码：[Triton](python/041_argmax.py) · [CUDA](cuda/041_argmax.cu)

#### 概念与用途

Row Argmax 返回每行最大元素的位置而不是最大值。它可用于分类预测和 Top-1 选择。有限值重复时，接口目标是最左索引。

#### 输入输出

输入形状为 `[rows, cols]`，输出是 int64 `[rows]`。Triton 输入支持 FP16、BF16 或 FP32；CUDA 输入为 FP32。

#### 算法

归约项是二元组 `(value, index)`。先比较 value；value 相等时比较 index，较小 index 胜出。单位元为 `(-inf, INT_MAX)`。

#### Triton 实现

```python
values = tl.load(x_ptr + row * cols + offsets, mask=offsets < cols, other=-float("inf"))
index = tl.argmax(values, axis=0, tie_break_left=True)
tl.store(out_ptr + row, index)
```

masked load 对尾部填 `-inf`。`tl.argmax(..., tie_break_left=True)` 返回 Triton block tensor 内索引。wrapper 分配 int64 输出，store 时完成类型转换。

#### CUDA 实现

```cuda
return (right.value > left.value ||
        (right.value == left.value && right.index < left.index))
    ? right : left;
```

`better_max` 实现 value-index 比较。`warp_argmax` 同时 shuffle value 和 index。`block_argmax` 经 shared `ValueIndex` 数组合并 warp，CUDA thread 0 写 int64 输出。

#### GPU 硬件

与标量最大值相比，每条归约边同时传递一个 float 和一个 int。所有通信仍在 CUDA thread block（CTA）内完成。

#### 教程

先用 `[7, 7, 1]` 验证返回 0，再测试跨 warp 的重复最大值。

#### 验证与限制

Triton 忽略 NaN。CUDA 比较器未实现 PyTorch 的 NaN 优先规则；全 NaN 行可能输出 `INT_MAX` 哨兵。

### 042 Row LogSoftmax

源码：[Triton](python/042_log_softmax.py) · [CUDA](cuda/042_log_softmax.cu)

#### 概念与用途

Row LogSoftmax 把每行 logits 转成对数概率。logit 是模型在归一化前输出的任意实数分数。LogSoftmax 常与负对数似然或交叉熵一起使用。

#### 输入输出

输入和输出形状均为 `[rows, cols]`。Triton 输出 dtype 与输入相同，CUDA 输入和输出均为 FP32。对有限输入，`exp(output).sum(-1)` 应接近 1。

#### 算法

$$
y_{r,c}=x_{r,c}-\left(m_r+\log\sum_je^{x_{r,j}-m_r}\right).
$$

减去最大值后，有限输入的最大指数为 1，从而避免普通指数上溢。

#### Triton 实现

```python
maximum = tl.max(values, axis=0)
log_denominator = maximum + tl.log(tl.sum(tl.exp(values - maximum), axis=0))
tl.store(out_ptr + row * cols + offsets, values - log_denominator, mask=mask)
```

masked load 对尾部填 `-inf` 并转 FP32。`tl.max` 得 $m_r$。`tl.exp` 和 `tl.sum` 得分母，`tl.log` 得对数分母。最后 masked store 整行。

#### CUDA 实现

```cuda
const float row_max = block_max(local_max);
local_sum += expf(x[blockIdx.x * cols + col] - maximum);
const float row_sum = block_sum(local_sum);
```

第一次归约得到 maximum，并通过 shared scalar 广播。第二次归约指数和，CUDA thread 0 计算 shared `log_denominator`。同步后所有 CUDA thread 再次跨步写输出。

#### GPU 硬件

一个 CUDA thread block（CTA）或 Triton program instance 融合三次行遍历和两个归约，避免把最大值或指数中间 tensor 写回全局内存。

#### 教程

先验证 `exp(y).sum(-1)` 接近 1，再给整行加同一常量，结果应近似不变。

#### 验证与限制

NaN 会污染结果；全 `-inf` 或含 `+inf` 的行可形成 `inf - inf` 并产生 NaN。

### 043 RMSNorm

源码：[Triton](python/043_rms_norm.py) · [CUDA](cuda/043_rms_norm.cu)

#### 概念与用途

RMSNorm 用均方根缩放每行，再乘逐列权重 `gamma`。它常用于 Transformer 层。与 LayerNorm 不同，它不减均值，也没有 `beta`。

#### 输入输出

输入 `x` 和输出的形状均为 `[rows, cols]`，权重 `gamma` 的形状为 `[cols]`。Triton 输出 dtype 与输入相同，CUDA 输入和输出均为 FP32。

#### 算法

$$
y_{r,c}=x_{r,c}\gamma_c\left(\frac{1}{C}\sum_jx_{r,j}^2+\epsilon\right)^{-1/2}.
$$

先归约平方和，除以列数得到均方，再计算一次逆平方根供整行复用。

#### Triton 实现

```python
mean_square = tl.sum(values * values, axis=0) / cols
gamma = tl.load(gamma_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
result = values * tl.rsqrt(mean_square + eps) * gamma
```

masked load 转 FP32，`tl.sum(values * values)` 得均方。kernel masked load `gamma`，计算 `values * rsqrt(...) * gamma` 并写回原 dtype。wrapper 验证 gamma 的形状、设备和 dtype。

#### CUDA 实现

```cuda
const float square_sum = block_sum(local);
if (threadIdx.x == 0) inverse_rms = rsqrtf(square_sum / cols + eps);
__syncthreads();
```

CUDA thread 归约平方和。CUDA thread 0 把 `rsqrtf(square_sum / cols + eps)` 写入 shared `inverse_rms`。同步后 CUDA thread 重读输入，并按列读取 gamma 写输出。

#### GPU 硬件

每行一个 CUDA thread block（CTA）或 Triton program instance。一个 shared 标量代替每个 CUDA thread 重复计算逆平方根。没有跨行通信。

#### 教程

先令 `gamma=1` 并与直接公式比较，再改变一个 `gamma` 元素，检查对应列是否被缩放。

#### 验证与限制

平方可能溢出或下溢；`eps=0` 且整行为零时可能产生 NaN。

### 044 BatchNorm Inference

源码：[Triton](python/044_batch_norm_inference.py) · [CUDA](cuda/044_batch_norm_inference.cu)

#### 概念与用途

BatchNorm Inference 使用训练阶段保存的通道均值和方差执行推理变换。它不重新计算当前 batch 的统计量，因此不是训练期 BatchNorm。

#### 输入输出

输入 `x` 和输出的形状均为 `[rows, cols]`。`running_mean`、`running_variance`、`gamma` 和 `beta` 的形状均为 `[cols]`。Triton 输出 dtype 与 `x` 相同；CUDA 输入和输出均为 FP32。

#### 算法

对扁平元素 $i$，通道为 $c=i\bmod C$。每个输出只依赖一个输入元素和该通道的四个参数，因此无需归约。

#### Triton 实现

```python
offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
channels = offsets % cols
result = (x - mean) * tl.rsqrt(variance + eps) * gamma + beta
```

Triton program instance 生成 256 个连续 offsets 和总长度 mask。`offsets % cols` 得 channels。kernel 分别 masked load 五个输入，转 FP32，应用公式并 masked store。

#### CUDA 实现

```cuda
const int offset = blockIdx.x * blockDim.x + threadIdx.x;
if (offset >= total) return;
const int channel = offset % cols;
out[offset] = (x[offset] - mean[channel]) * rsqrtf(variance[channel] + eps) * gamma[channel] + beta[channel];
```

每个 CUDA thread 计算一个 offset。越界 CUDA thread 立即返回。有效 CUDA thread 计算 channel，读取通道参数并调用 `rsqrtf`。launcher 的 CUDA grid 覆盖 `rows * cols`。

#### GPU 硬件

这是普通逐元素 CUDA grid，不是“一个 CUDA thread block（CTA）处理一行”。相邻 CUDA thread 读取连续 x；通道参数按行重复访问，是否命中 cache 需要 profiler 验证。

#### 教程

令 `mean=0`、`variance=1`、`gamma=1`、`beta=0`。当 `eps` 很小时，输出应近似输入。

#### 验证与限制

代码不检查 `running_variance >= 0`；`variance + eps < 0` 时结果为 NaN。大扁平 offset 受 int32 限制。

### 045 GroupNorm

源码：[Triton](python/045_group_norm.py) · [CUDA](cuda/045_group_norm.cu)

#### 概念与用途

GroupNorm 把通道分组，并在每个 batch-group 内计算统计量。它不依赖其他 batch 样本，常用于小 batch 模型。

#### 输入输出

输入和输出布局固定为 `[batch, channels, spatial]`。`gamma` 和 `beta` 的形状为 `[channels]`。Triton 输出 dtype 与输入相同；CUDA 输入和输出均为 FP32。

#### 算法

每组通道数为 $C/G$，元素数为 $K=(C/G)S$。先对这 $K$ 个元素计算均值和总体方差，再按原通道应用 gamma 和 beta。

#### Triton 实现

```python
program = tl.program_id(0)
group = program % groups
batch = program // groups
base = batch * channels * spatial + group * count
mean = tl.sum(values, axis=0) / count
```

Triton program instance id 解码为 batch 和 group。连续的 group 数据从 `base` 开始加载。两次 `tl.sum` 得均值和方差。`offsets // spatial` 恢复组内通道，再加载 affine 参数。

#### CUDA 实现

```cuda
const int group = blockIdx.x % groups;
const int batch = blockIdx.x / groups;
const int base = batch * channels * spatial + group * count;
```

`blockIdx.x % groups` 得 group，除法得 batch。CUDA thread 两遍跨步读取 group。两个 shared scalar 保存 mean 和 inverse_std。第三遍读取输入并写回归一化结果。

#### GPU 硬件

一个 CUDA thread block（CTA）或 Triton program instance 对应一个 batch-group。组在内存中连续，因此线性 index 可覆盖通道和 spatial。不同组互不通信。

#### 教程

依次设置 `groups=1`、`groups=channels` 和中间分组，并与 `F.group_norm` 比较。

#### 验证与限制

Triton 每组最多 65,536 个元素。CUDA 未检查多个 int32 乘法溢出；两套统计也继承 040 的抵消风险。

### 046 Inclusive Cumsum

源码：[Triton](python/046_cumsum.py) · [CUDA](cuda/046_cumsum.cu)

#### 概念与用途

Inclusive Cumsum 返回每个位置及其之前所有元素的和。它用于累计计数、序列偏移和部分并行算法。当前元素包含在输出中，因此是 inclusive scan。

#### 输入输出

输入和输出形状均为 `[rows, cols]`。Triton 输出 dtype 与输入相同；CUDA 输入和输出均为 FP32。

#### 算法

Hillis–Steele 算法在距离 1、2、4、8 等位置读取左侧部分和。经过 $\log_2T$ 轮后，tile 内每个位置都有前缀和。后续 tile 还需加前一 tile 的总和。

#### Triton 实现

```python
values = tl.load(x_ptr + row * cols + offsets, mask=mask, other=0.0).to(tl.float32)
inclusive_prefix = tl.cumsum(values, axis=0)
tl.store(out_ptr + row * cols + offsets, inclusive_prefix, mask=mask)
```

Triton program instance masked load 整行并转 FP32。`tl.cumsum(values, axis=0)` 生成 Triton block tensor 的前缀。masked store 丢弃 padding，并转换回输入 dtype。

#### CUDA 实现

```cuda
const float left = threadIdx.x >= offset ? scan[threadIdx.x - offset] : 0.0f;
__syncthreads();
scan[threadIdx.x] += left;
__syncthreads();
```

CUDA thread block（CTA）初始化 shared `carry=0`。每轮把最多 256 个值写入 `scan`，在每个 offset 前后同步。所有 CUDA thread 快照旧 carry 后写输出；最后一个 CUDA thread 累加 tile 总和，再同步进入下一 tile。

#### GPU 硬件

一个 CUDA thread block（CTA）独占一行。tile 内 256 个 CUDA thread 并行，但多个 tile 在同一 CUDA thread block（CTA）中串行。这避免跨 CUDA thread block（CTA）carry 协议，代价是宽行并行度受限且 barrier 较多。

#### 教程

用全 1 输入检查输出为 `[1,2,3,...]`，并分别测试 255、256、257、512 和 513 列。

#### 验证与限制

长行会累积舍入误差。CUDA 的 `tile`、`col` 和 `row_base` 使用 int32。

## 入门实验

以下是建议实验，不代表仓库已在当前机器完成运行。需要 PyTorch、Triton、pytest 和可用的 CUDA 或 ROCm GPU。

### 运行一个可靠的算子检查

```shell
python3 benchmarks/run.py --op 042 --size medium --check
```

从项目根目录运行。命令返回码为 0，并打印 Triton 与 PyTorch reference 的计时行，表示这个输入先通过快速数值比较；它不编译 CUDA，也不证明其他 shape 正确。

整套分类测试可用下面的命令调查，但当前宽度上限用例访问 `ops.MAX_REDUCTION_SIZE`，而动态 namespace 只解析 `opNNN_*`，因此会在调用算子前抛 `AttributeError`。在修复该测试前，不能把整套测试描述为可靠 happy path。

```shell
python3 -m pytest -q tests/test_03_reductions_normalization.py
```

### 检查 benchmark case

```shell
python3 benchmarks/run.py --op 046 --size large --check
```

`--check` 在计时前比较一次 Triton 输出与 PyTorch reference。这些是建议命令，不是性能结果。case 定义位于 [benchmarks/cases.py](../../benchmarks/cases.py)。当前比较 helper 对 int64 索引也使用浮点相对容差，因此 041 的接近但错误索引可能被判为通过。

### 增加边界实验

1. 在不同列放置 NaN，比较 036、037 和 041。
2. 用大常量加小扰动测试 040 和 045。
3. 用全零行和 `eps=0` 测试 043。
4. 用全 `-inf`、含 `+inf` 和含 NaN 的行测试 042。
5. 用 255、256、257、512 和 513 列测试 046。
6. 在显存足够的设备上测试总元素数跨越 `INT_MAX` 的地址计算。

比较 NaN 时应显式允许对应位置同时为 NaN。当前 `tests/_utils.py` 的 `assert_close` 没有设置 `equal_nan=True`。整数索引应精确比较。

### 分析性能

用 profiler 检查带宽、归约延迟、shared-memory 同步、寄存器、spill 和 occupancy。首次 Triton JIT 编译时间不属于稳态 kernel 时间。没有实测数据时，不应声明某实现更快。

## 验证状态

### 已由源码和静态检查确认

- 034–046 各有一份命名一致的 Triton/Python 源码和 CUDA 源码。
- Python 源码、分类测试和 benchmark case 可通过 Python 语法编译。
- CUDA 公共归约 helper 的真实位置是 `shared/cuda_common.cuh`。
- 044 是逐元素推理变换，不计算统计量。
- 046 的 CUDA barrier 对 CUDA thread block（CTA）内 CUDA thread 一致；代码审查未发现支持范围内的 barrier deadlock。
- 代码审查已识别 NaN 语义、int32 地址、CUDA 040 数值稳定性和 CUDA 045 溢出限制。

### 尚未验证或当前已知失败

- 当前环境未完成 Triton JIT、CUDA 编译、GPU correctness test 或性能 benchmark。
- 分类测试的宽度上限用例存在 namespace 常量访问错误。
- 036、037 和 041 的 NaN 行为与当前 PyTorch reference 不一致。
- 041 的 CUDA 全 NaN 行可能输出 `INT_MAX`。
- 超过 int32 地址范围的大 tensor 没有安全保证。
- CUDA 040 和 045 没有生产级稳定方差算法。
- Python 的 65,536 元素宽度未在所有 dtype 和目标 GPU 上完成资源验证。
- 分类测试只调用 Triton callable，不直接编译或运行 CUDA 编号文件。

在修复并于目标 GPU 上运行前，准确表述是“源码存在且完成静态审查”，不是“所有算子已经验证正确或达到某项性能”。
