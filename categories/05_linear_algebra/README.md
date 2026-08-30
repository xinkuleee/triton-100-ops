# 05 线性代数：057–060

本章解释批量矩阵乘、矩阵向量乘、外积和带偏置线性层。`python/` 文件包含 Triton kernel 与 Python 包装函数；kernel 是在 GPU 上执行的函数，包装函数在 CPU 上检查输入、分配输出并启动 kernel。`cuda/` 文件是独立的 FP32 教学实现，不调用 NVIDIA 的高性能线性代数库 cuBLAS。现有自动化设备测试只面向 Triton/Python 路径。

## 共同概念

**Scalar（标量）**是一个数，**vector（向量）**是一维数列，**matrix（矩阵）**是二维数表，**tensor（张量）**是它们向更多维的推广。**Shape（形状）**记录各维长度；`[M,K]` 表示 M 行 K 列。**Dtype（数据类型）**规定每个元素的编码和精度。FP16、BF16 和 FP32 都是浮点格式：FP16 和 BF16 各占 16 位，FP32 占 32 位；BF16 用较少的有效数字换取接近 FP32 的指数范围。

**Dot product（点积）**把两个等长向量逐元素相乘后求和。矩阵乘让 A 的每一行与 B 的每一列做点积；K 是参与每个点积的内维。**Accumulator（累加器）**保存乘加过程中的部分和。**Reduction（归约）**把一组值合并成更少的值，例如求和。**Broadcast（广播）**让长度为 1 的维度在运算中逻辑扩展，不必复制数据。

**Tile（分块）**是矩阵的一块矩形区域。分块让片上数据被多个计算复用。**Mask（掩码）**标记边缘分块中的有效元素，防止读取或写入矩阵范围之外。大 O 记号只描述规模增长趋势。例如，$O(MK)$ 表示工作量随 $M\times K$ 成比例增长，不表示精确运行时间。

本章中的矩阵按 row-major（行优先）顺序连续存储：同一行的元素相邻，最后一个维度变化最快。**Pointer（指针）**保存内存地址；地址表达式把多维坐标换算成一维存储位置。**Contiguous（连续）**表示 tensor 的逻辑顺序与这段连续存储一致。

**CUDA thread** 执行标量指令；一组可同步的 CUDA thread 组成 **CUDA thread block（CTA）**；全部 CUDA thread block（CTA）组成 **CUDA grid**。NVIDIA GPU 通常以 32 个 CUDA thread 组成的 **warp** 为调度单位。**Barrier（屏障）**要求同一 CUDA thread block（CTA）中的 CUDA thread 都到达指定位置后再继续。

**Global memory（全局内存）**容量大但访问延迟高。**Register（寄存器）**是单个 CUDA thread 使用的片上存储。**Shared memory（共享内存）**是同一 CUDA thread block（CTA）共享的片上存储。硬件 cache（缓存）自动保留最近访问的数据。CUDA 057 显式使用 shared memory 复用输入 tile；CUDA 058–060 使用更直接的基线。

**Triton program instance** 是 Triton launch grid 中的独立逻辑实例；**Triton block tensor** 是该实例同时表达的一组逻辑值。Triton 后端把 kernel 编译成目标 GPU 的机器指令。NVIDIA 和 AMD GPU 的可用指令不同；ROCm 是 AMD GPU 的软件栈。

Triton launch grid 的每个维度都规定要启动多少个 Triton program instance。`tl.program_id(axis)` 返回当前 Triton program instance 在指定维度中从 0 开始的 ID。`tl.arange` 创建一组连续的逻辑坐标。运行时参数在每次调用时才确定；`tl.constexpr` 标记编译 kernel 时已知的编译期参数。CUDA 中的 `blockIdx` 和 `threadIdx` 分别给出 CUDA thread block（CTA）与 CUDA thread 的坐标。

`tl.load` 从指针指定的地址读取值，`tl.store` 把值写到指定地址，`tl.dot` 完成两个 Triton block tensor 的矩阵乘加，`tl.sum` 完成求和归约。CUDA 的 `__shared__` 声明 shared memory，`__syncthreads()` 实现 CUDA thread block（CTA）内的 barrier。原子操作把一次读、改、写作为不可分割的操作；本章实现不需要原子操作。

CUDA core 执行通用算术指令。Tensor Core 是面向小矩阵乘加的专用硬件；MMA 表示矩阵乘加操作。TF32（TensorFloat-32）是一种 Tensor Core 计算格式，保留 FP32 的指数范围但减少乘法输入的有效位。IEEE 754 是浮点运算标准；文中的 IEEE FP32 指按该标准的 FP32 精度执行输入乘法。

Memory bandwidth（内存带宽）表示单位时间能传输多少数据；arithmetic throughput（算术吞吐量）表示单位时间能完成多少计算。读取数据耗时占主导的 kernel 受内存带宽限制；计算耗时占主导的 kernel 受算术吞吐量限制。

`INT_MAX` 是 32 位有符号整数的最大值。整数溢出表示计算结果超出数据类型可表示的范围；用溢出的结果计算指针可能访问错误地址。文中的地址上限用于说明实现边界，不代表 GPU 内存本身只有 32 位地址。

验证代码把 kernel 结果与 **reference（参考实现）**的预期结果比较。**Benchmark（性能基准）**测量预设输入的运行表现，不等于完整正确性测试。JIT（just-in-time，即时编译）在运行期间编译 Triton kernel；`nvcc` 是 NVIDIA CUDA 编译器。

`python/_common.py` 的 `require_tensors` 检查连续、同设备、同 dtype 的 GPU 浮点 tensor。支持 FP16、BF16、FP32。`MAX_DOT_K` 只限制 057/060 的 K，`MAX_GEMV_K` 只限制 058 的 K；它们不限制 M、N、batch 或总元素数。

`cuda/_common.cuh` 定义 `kTile=16`、`kMaxDotK=65536` 和 `matrix_dims_valid`。`matrix_dims_valid` 只检查非负维度和 `m*n <= INT_MAX`，不证明 `m*k`、`k*n`、batch 乘积或全部设备地址安全。

## 057 — Batched Matmul

### 概念与用途

Batched Matmul（批量矩阵乘）一次处理 B 组互不依赖的矩阵乘。B 是 batch（批次）维，表示一批结构相同的数据。常见用途包括同时处理多个样本，以及同时计算多个注意力头的矩阵乘。

### 输入输出

输入 A 的 shape 是 `[B,M,K]`，输入 B 的 shape 是 `[B,K,N]`，输出 C 的 shape 是 `[B,M,N]`、dtype 是 FP32。两个输入的 batch 数必须相同，A 的 K 必须等于 B 的倒数第二维。K 是点积长度；M 和 N 决定每个输出矩阵的行数和列数。K=0 时没有乘加项，因此输出全为零。

### 算法

对每个 batch 独立计算：

$$
C_{b,m,n}=\sum_{q=0}^{K-1}A_{b,m,q}B_{b,q,n}.
$$

每个输出元素读取 K 对输入并完成 K 次乘加。直接算法的工作量为 $O(BMNK)$，输出元素数为 $BMN$。实现把输出分为 tile，并沿 K 维分段累加。

### Triton 实现

`python/057_batched_matmul.py` 使用三维 Triton launch grid。包装函数设置 `BM=32`、`BN=32`、`BK=32`，因此每个 Triton program instance 负责一个 32x32 输出 tile，并让 K 每轮前进 32 个元素。`tl.program_id(0)` 是运行时 batch ID，不是 `tl.constexpr`；M、N、K 和 tile 大小才是编译期参数。

FP16/BF16 输入由 tl.dot 用 FP32 accumulator 累加。FP32 输入未指定 input_precision=ieee；NVIDIA 后端默认可能使用 TF32，ROCm 默认路径不同。因此，支持 FP32 不表示跨后端逐位一致。

```python
batch = tl.program_id(0)
rows = tl.program_id(1) * BM + tl.arange(0, BM)
cols = tl.program_id(2) * BN + tl.arange(0, BN)
for k0 in range(0, K, BK):
    a = tl.load(a_ptr + batch * M * K + rows[:, None] * K + inner[None, :],
                mask=(rows[:, None] < M) & (inner[None, :] < K), other=0.0)
    b = tl.load(b_ptr + batch * K * N + inner[:, None] * N + cols[None, :],
                mask=(inner[:, None] < K) & (cols[None, :] < N), other=0.0)
    accumulator = tl.dot(a, b, accumulator)
```

三个 `tl.program_id` 返回的 Triton program instance ID 分别选 batch、M tile、N tile。`rows[:,None]` 与 `inner[None,:]` 构造 BM x BK 地址块；B 使用 BK x BN 地址块。

地址包含 batch*M*K、batch*K*N、batch*M*N。当前包装函数没有为这些乘积设置显式地址上限。

#### 代码阅读

第一段用三个 `tl.program_id` 返回的 Triton program instance ID 选择 batch、行 tile 和列 tile。第二段创建 FP32 accumulator。K 循环构造 A 的 BM x BK tile 与 B 的 BK x BN tile，mask 把尾部补零；tl.dot 将它们累加。最后一段只写 M x N 有效区域。batch ID 参与地址计算，但不会成为一个额外的 Triton block tensor 维度。

### CUDA 实现

`cuda/057_batched_matmul.cu` 使用 16x16 CUDA thread block（CTA）和两个 shared-memory tile，`blockIdx.z` 选择 batch。地址使用 `int` 乘法。launcher（启动函数）未验证 `m*k`、`k*n`、`batch*m*k`、`batch*k*n` 或 `batch*m*n`，也未单独验证 batch 是否符合 CUDA grid 的 z 维限制。溢出可能导致越界。

```cuda
__shared__ float a_tile[kTile][kTile];
__shared__ float b_tile[kTile][kTile];
a_tile[threadIdx.y][threadIdx.x] = row < m && a_col < k ? batch_a[row * k + a_col] : 0.0f;
b_tile[threadIdx.y][threadIdx.x] = b_row < k && col < n ? batch_b[b_row * n + col] : 0.0f;
__syncthreads();
accumulator += a_tile[threadIdx.y][inner] * b_tile[inner][threadIdx.x];
```

CUDA thread block（CTA）合作填充两个 tile。第一道 barrier 保证 tile 完整；源码中的第二道 barrier 防止下一轮提前覆盖。

#### 代码阅读

每个 CUDA thread 对应一个输出元素。每轮先把一个 A/B tile 合作搬入 shared memory，barrier 后从 shared memory 执行 16 次乘加，再用第二个 barrier 防止下一轮覆盖仍在使用的数据。

### GPU 硬件映射

shared memory 减少同一 tile 的重复全局读取。`tl.dot` 表达 tile 矩阵乘加，后端根据输入 dtype 和目标 GPU 选择底层指令。CUDA 路径是 FP32 CUDA core 基线；源码没有显式 Tensor Core MMA。

### 教程

跟踪一个 16x16 tile：每轮把 A/B 子块搬入 shared memory，执行 16 次乘加，再处理下一个 K tile。Triton 表达相同数据依赖，但由 tl.dot 选择底层指令。练习：对 B=2、M=35、N=29、K=17 写出 Triton launch grid，并指出哪些 tile 需要 mask。

### 验证与限制

测试覆盖一组非整 tile FP16 shape、K=0 和一例 shape 不匹配。reference torch.bmm(a,b).float() 先按输入 dtype/后端语义计算，再转 FP32，不是先把输入转 FP32。未覆盖 BF16、FP32/TF32、空 M/N/B、最大 K 或大地址。benchmark 只用 FP16。CUDA 未编译或运行。

## 058 — GEMV

### 概念与用途

GEMV 是 General Matrix-Vector Multiplication（通用矩阵向量乘）的缩写。它让矩阵的每一行与同一个向量做点积，常用于线性代数迭代、较小批次推理和只处理一个向量的线性变换。

### 输入输出

输入 matrix 的 shape 是 `[M,K]`，输入 vector 的 shape 是 `[K]`，两者的 K 必须相同。输出 shape 是 `[M]`、dtype 是 FP32；每个输出元素对应 matrix 的一行。当前 Triton 和 CUDA 包装路径要求 $1\le K\le\text{MAX_GEMV_K}$，因此拒绝 K=0。

### 算法

矩阵向量乘计算：

$$
y_m=\sum_{q=0}^{K-1}A_{m,q}x_q.
$$

一个 Triton program instance 或 CUDA thread block（CTA）处理一行。CUDA thread 先计算部分和，再做块内归约。直接算法的工作量是 $O(MK)$，读取约 $MK$ 个矩阵元素。

### Triton 实现

`python/058_gemv.py` 使用长度不小于 K 的二次幂 Triton block tensor，并要求 $1 \le K \le \text{MAX_GEMV_K}$。二次幂是 $2^n$ 形式的整数，例如 128、256。数学和 PyTorch 都定义 K=0 时每行输出 0，但当前包装函数拒绝该输入。

```python
row = tl.program_id(0)
col = tl.arange(0, BLOCK_K)
mask = col < cols
matrix = tl.load(matrix_ptr + row * cols + col, mask=mask, other=0.0)
vector = tl.load(vector_ptr + col, mask=mask, other=0.0)
tl.store(out_ptr + row, tl.sum(matrix.to(tl.float32) * vector, axis=0))
```

一个 Triton program instance 覆盖一行。二次幂 Triton block tensor 中的多余逻辑元素由同一个 `mask` 补零。

#### 代码阅读

`tl.program_id` 返回的 Triton program instance ID 选择行，tl.arange 生成列。mask 把二次幂 Triton block tensor 中超出 K 的逻辑元素补零。矩阵值转为 FP32，与向量相乘后用 tl.sum 归约，再把标量逻辑结果写入该行输出。

### CUDA 实现

`cuda/058_gemv.cu` 每行启动一个 CUDA thread block（CTA），使用 `block_sum`，也拒绝 K=0。地址 `row*cols+col` 使用 `int`；launcher 未检查 `rows*cols <= INT_MAX`。

```cuda
float partial = 0.0f;
for (int col = threadIdx.x; col < cols; col += blockDim.x)
  partial += matrix[row * cols + col] * vector[col];
const float sum = block_sum(partial);
if (threadIdx.x == 0) out[row] = sum;
```

每个 CUDA thread 先在寄存器累加，`block_sum` 再合并整个 CUDA thread block（CTA）的部分和。

#### 代码阅读

每个 CUDA thread 以 `blockDim.x` 为步长遍历列，并在寄存器 `partial` 中保存部分和。`block_sum` 先用 warp shuffle 合并一个 warp 内的值，再通过 shared memory 合并各 warp。一个 CUDA thread block（CTA）只产生一个标量。

### GPU 硬件映射

Warp shuffle 让同一 warp 的 CUDA thread 直接交换寄存器值，不必为每一步归约都访问 shared memory。vector 可被不同 CUDA thread block（CTA）从硬件 cache 复用，而 matrix 通常只读一次。矩阵数据的复用很少，因此 GEMV 常由 global memory 带宽而不是算术吞吐量限制。

### 教程

256 个 CUDA thread 分别读取 q=threadIdx.x、q+256 等位置，然后把部分和归约为一个输出标量。练习：说明 K=193 时 CUDA thread 0 和 CUDA thread 200 各读取哪些列，以及哪些 CUDA thread 只有一个有效元素。

### 验证与限制

测试只覆盖一组非二次幂 FP32 K。未覆盖 FP16、BF16、K=0、空行、上限或地址溢出。benchmark 使用方阵。CUDA 未运行。

## 059 — Outer Product

### 概念与用途

Outer Product（外积）把两个向量的所有元素两两相乘，生成一个矩阵。它常用于秩 1 矩阵更新、梯度构造和两个特征轴的组合。与矩阵乘不同，外积没有归约轴，因此不同输出元素完全独立。

### 输入输出

输入 x 的 shape 是 `[M]`，输入 y 的 shape 是 `[N]`。输出 Z 的 shape 是 `[M,N]`、dtype 是 FP32。输入必须是同设备、同 dtype 的连续 GPU 浮点 tensor。任一向量为空时，Triton 包装函数返回对应的空输出。

### 算法

外积计算：

$$
Z_{m,n}=x_m y_n.
$$

每个输出元素执行一次乘法。工作量和输出存储都是 $O(MN)$。算法通过广播把 x 看作 M 行 1 列，把 y 看作 1 行 N 列，然后逐元素相乘。

### Triton 实现

`python/059_outer_product.py` 使用 32x32 tile 和广播。任一向量为空时返回对应空 shape。

```python
row = tl.program_id(0) * BM + tl.arange(0, BM)
col = tl.program_id(1) * BN + tl.arange(0, BN)
x = tl.load(x_ptr + row, mask=row < rows, other=0.0)
y = tl.load(y_ptr + col, mask=col < cols, other=0.0)
tl.store(out_ptr + row[:, None] * cols + col[None, :], x[:, None] * y[None, :], mask=mask)
```

两个一维向量通过增加长度为 1 的维度广播成 BM x BN tile；没有归约轴。

#### 代码阅读

两个 `tl.program_id` 返回的 Triton program instance ID 选择行/列 tile。kernel 分别加载 BM 个 x 和 BN 个 y，再用 x[:,None] 与 y[None,:] 广播生成 BM x BN 结果。二维 mask 只写有效矩形。

### CUDA 实现

`cuda/059_outer_product.cu` 使用 16x16 CUDA thread block（CTA）。launcher 允许零维，但只检查非负维度，没有调用 `matrix_dims_valid`；`M*N > INT_MAX` 时 `row*cols+col` 可能溢出。

```cuda
const int row = blockIdx.y * blockDim.y + threadIdx.y;
const int col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < rows && col < cols)
  out[row * cols + col] = x[row] * y[col];
```

二维 CUDA thread block（CTA）直接映射二维输出；条件处理边缘 tile。

#### 代码阅读

`blockIdx` 和 `threadIdx` 直接形成二维坐标。有效 CUDA thread 读取一个 x 元素和一个 y 元素，完成一次乘法，并写一个 out 元素。

### GPU 硬件映射

同一 CUDA thread block（CTA）中的多个 CUDA thread 会重复引用相同的 x 或 y 元素。硬件 cache 可以合并或复用其中一部分 global memory 访问；源码没有显式把输入复制到 shared memory。这个实现没有 CUDA thread 间通信，也不需要 barrier 或原子操作。

### 教程

x=[a,b]、y=[c,d,e] 时，每个输出行是一个 x 元素乘完整 y。没有 CUDA thread 间求和或原子操作。练习：写出完整 2x3 输出，并标出 Triton 广播产生的两个逻辑输入 tile。

### 验证与限制

测试覆盖一组 FP32 向量和 dtype 不匹配，未覆盖空向量、低精度或大输出。benchmark 只用等长 FP32 向量。CUDA 未运行。

## 060 — Linear + Bias

### 概念与用途

Linear + Bias（带偏置线性层）先做线性变换，再给每个输出特征加一个 bias（偏置）。线性变换用权重将 K 个输入特征组合成 N 个输出特征；偏置给每个输出特征增加一个可学习常数。它是全连接神经网络和 Transformer 投影层的基础操作。**Fusion（融合）**把原本需要多个 kernel 的步骤放入一个 kernel，从而减少中间数据在 global memory 中的读写。

### 输入输出

输入 x 的 shape 是 `[M,K]`，weight 的 shape 是 `[N,K]`，bias 的 shape 是 `[N]`。x 与 weight 的 K 必须相同，bias 长度必须等于 N。输出 Y 的 shape 是 `[M,N]`、dtype 是 FP32。K=0 时没有乘加项，因此每一行都等于 bias。

### 算法

线性层计算：

$$
Y_{m,n}=b_n+\sum_{q=0}^{K-1}X_{m,q}W_{n,q}.
$$

矩阵乘使用 FP32 accumulator 形成 tile，再在唯一一次写回前加 bias。直接算法的工作量是 $O(MNK)$；bias 融合增加 $MN$ 次加法，但省去单独 bias kernel 对整个输出的一次读取和一次写入。

### Triton 实现

`python/060_linear_bias.py` 使用 32x32x32 tile。weight 保持 `[N,K]` 存储；指针表达式形成 `weight.T` 的寄存器 tile，不物化转置。这里的转置交换矩阵的行和列；不物化表示不创建另一份转置后的 global memory 数据。

与 057 相同，FP32 输入未指定 IEEE 精度，NVIDIA 上可能使用 TF32。包装函数只限制 K，没有限制 M*K、N*K 或 M*N。

```python
rows = tl.program_id(0) * BM + tl.arange(0, BM)
cols = tl.program_id(1) * BN + tl.arange(0, BN)
x = tl.load(x_ptr + rows[:, None] * K + inner[None, :],
            mask=(rows[:, None] < M) & (inner[None, :] < K), other=0.0)
weight = tl.load(weight_ptr + cols[None, :] * K + inner[:, None],
                 mask=(cols[None, :] < N) & (inner[:, None] < K), other=0.0)
accumulator = tl.dot(x, weight, accumulator)
bias = tl.load(bias_ptr + cols, mask=cols < N, other=0.0)
tl.store(out_ptr + rows[:, None] * N + cols[None, :], accumulator + bias[None, :],
         mask=(rows[:, None] < M) & (cols[None, :] < N))
```

x tile 是 BM x BK。weight 地址把 `inner` 放在第一维、`cols` 放在第二维，直接形成 BK x BN 的逻辑转置；`bias[None,:]` 沿 M 维广播。

#### 代码阅读

行/列 `tl.program_id` 返回的 Triton program instance ID 选输出 tile。K 循环加载 x 的 BM x BK tile。weight 的地址按 cols*K+inner 读取，得到逻辑 BK x BN 的转置视图。tl.dot 累加后只加载 BN 个 bias，并广播到 BM 行再写回。

### CUDA 实现

`cuda/060_linear_bias.cu` 每个 CUDA thread 串行计算一个点积，没有 shared-memory tile。`matrix_dims_valid` 只验证 `m*n`；`row*k+inner` 和 `col*k+inner` 仍可能溢出。

```cuda
float accumulator = bias[col];
for (int inner = 0; inner < k; ++inner)
  accumulator += x[row * k + inner] * weight[col * k + inner];
out[row * n + col] = accumulator;
```

每个 CUDA thread 先装入自己的 bias，再串行遍历 K。CUDA thread block（CTA）内没有合作加载或 shared-memory 复用。

#### 代码阅读

二维 CUDA thread 坐标选择一个输出。accumulator 先初始化为 `bias[col]`，随后由单个 CUDA thread 串行遍历 K。

### GPU 硬件映射

Triton 用 `tl.dot` 表达 tile 矩阵乘加，并在 accumulator 写回 global memory 前广播和加入 bias。CUDA 实现没有 CUDA thread block（CTA）内合作或输入 tile 复用，因此容易重复读取 x 和 weight。它用于展示公式和融合位置，不代表高性能 GEMM，也没有显式使用 Tensor Core MMA。

### 教程

若先写 GEMM 结果再启动 bias kernel，需要额外读写整个输出。当前 Triton kernel 在 accumulator 写回前加 bias，省掉该中间往返。练习：对 M=35、N=27、K=19 写出 Triton launch grid，并解释 weight 地址为何等价于 weight.T。

### 验证与限制

测试覆盖非整 tile FP16 shape 和 K=0。reference F.linear(...).float() 先按输入 dtype 与后端语义完成线性层，再转 FP32，不代表纯 IEEE FP32 运算。未覆盖 BF16、FP32/TF32、空 M/N、大地址或最大 K。benchmark 只用 FP16 方阵。CUDA 未运行。

## 运行验证

```shell
python3 -m pytest -q tests/test_05_linear_algebra.py
python3 benchmarks/run.py --op 057 --size medium --check
```

--check 只比较一组预设输入。small/medium/large 不是完整 shape、dtype 或边界扫描。静态测试只检查编号、文件、符号和直接 kernel launch，不编译 CUDA。当前项目记录的环境没有 nvcc 或可用 GPU，因此 CUDA 编译、Triton JIT、GPU 数值和性能均未验证。
