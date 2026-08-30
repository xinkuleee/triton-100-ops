# 逐元素算子 012–033

本章解释 22 个逐元素 GPU 算子的当前实现。Python 文件提供 Triton 接口；CUDA 文件提供独立的 FP32 CUDA 实现。两套代码互不调用，也不保证所有特殊值完全一致。

当前实现主要面向连续张量和普通有限输入。大张量索引、NaN clamp 边界、多 GPU current device、部分非有限标量和 CUDA 空指针仍有已知限制。不要把本章的验证建议当作现有测试已经覆盖的事实。

## 源码地图

每个编号的两份文件实现同一个算子。Python 文件包含 Triton kernel 和包装函数；CUDA 文件包含独立 kernel 和 launcher。

| 编号 | Triton | CUDA |
|---:|---|---|
| 012 | [`python/012_sub.py`](python/012_sub.py) | [`cuda/012_sub.cu`](cuda/012_sub.cu) |
| 013 | [`python/013_mul.py`](python/013_mul.py) | [`cuda/013_mul.cu`](cuda/013_mul.cu) |
| 014 | [`python/014_div.py`](python/014_div.py) | [`cuda/014_div.cu`](cuda/014_div.cu) |
| 015 | [`python/015_scalar_add.py`](python/015_scalar_add.py) | [`cuda/015_scalar_add.cu`](cuda/015_scalar_add.cu) |
| 016 | [`python/016_axpby.py`](python/016_axpby.py) | [`cuda/016_axpby.cu`](cuda/016_axpby.cu) |
| 017 | [`python/017_relu.py`](python/017_relu.py) | [`cuda/017_relu.cu`](cuda/017_relu.cu) |
| 018 | [`python/018_leaky_relu.py`](python/018_leaky_relu.py) | [`cuda/018_leaky_relu.cu`](cuda/018_leaky_relu.cu) |
| 019 | [`python/019_sigmoid.py`](python/019_sigmoid.py) | [`cuda/019_sigmoid.cu`](cuda/019_sigmoid.cu) |
| 020 | [`python/020_tanh.py`](python/020_tanh.py) | [`cuda/020_tanh.cu`](cuda/020_tanh.cu) |
| 021 | [`python/021_gelu.py`](python/021_gelu.py) | [`cuda/021_gelu.cu`](cuda/021_gelu.cu) |
| 022 | [`python/022_silu.py`](python/022_silu.py) | [`cuda/022_silu.cu`](cuda/022_silu.cu) |
| 023 | [`python/023_softplus.py`](python/023_softplus.py) | [`cuda/023_softplus.cu`](cuda/023_softplus.cu) |
| 024 | [`python/024_elu.py`](python/024_elu.py) | [`cuda/024_elu.cu`](cuda/024_elu.cu) |
| 025 | [`python/025_hard_sigmoid.py`](python/025_hard_sigmoid.py) | [`cuda/025_hard_sigmoid.cu`](cuda/025_hard_sigmoid.cu) |
| 026 | [`python/026_hard_swish.py`](python/026_hard_swish.py) | [`cuda/026_hard_swish.cu`](cuda/026_hard_swish.cu) |
| 027 | [`python/027_square.py`](python/027_square.py) | [`cuda/027_square.cu`](cuda/027_square.cu) |
| 028 | [`python/028_sqrt.py`](python/028_sqrt.py) | [`cuda/028_sqrt.cu`](cuda/028_sqrt.cu) |
| 029 | [`python/029_exp.py`](python/029_exp.py) | [`cuda/029_exp.cu`](cuda/029_exp.cu) |
| 030 | [`python/030_log.py`](python/030_log.py) | [`cuda/030_log.cu`](cuda/030_log.cu) |
| 031 | [`python/031_clamp.py`](python/031_clamp.py) | [`cuda/031_clamp.cu`](cuda/031_clamp.cu) |
| 032 | [`python/032_where.py`](python/032_where.py) | [`cuda/032_where.cu`](cuda/032_where.cu) |
| 033 | [`python/033_row_bias_add.py`](python/033_row_bias_add.py) | [`cuda/033_row_bias_add.cu`](cuda/033_row_bias_add.cu) |

## 核心概念

### 张量、shape、dtype 和 device

**张量**是按一个或多个维度排列的数值集合。本章把它存放在 GPU 内存中。**shape** 描述每个维度的长度；例如 shape 为 `[2, 3]` 的二维张量有 2 行、3 列，共 6 个元素。

**dtype** 是每个元素的数据类型。本章 Python 接口支持 FP16、BF16 和 FP32 三种浮点格式。FP16 和 BF16 每个元素占 16 bit，FP32 每个元素占 32 bit；格式不同会改变可表示范围、精度和内存流量。**device** 表示张量所在的 GPU。参与同一个算子的张量必须位于同一 device，才能由同一次 kernel launch 直接访问。CUDA 是 NVIDIA GPU 的编程平台；ROCm 是 AMD GPU 的软件栈。

### Kernel、包装函数和 launcher

**Kernel** 是在 GPU 上并行执行的函数。它读取输入、进行计算并写出结果。Python **包装函数**在 CPU 端检查 PyTorch 张量、分配输出并启动 Triton kernel。CUDA **launcher** 是 CPU 端的 C++ 函数；它检查标量参数、配置 CUDA grid 并启动 CUDA kernel。

`load` 表示从 GPU 内存读取数据，`store` 表示写回数据。**Mask** 是与一组索引对应的布尔条件；最后一个计算块不足 256 个元素时，mask 让 Triton 只访问真实存在的地址。CUDA kernel 用 `index < n` 完成同样的地址保护。

### 逐元素计算

逐元素算子独立处理每个位置。例如减法满足 $z_i=x_i-y_i$。位置 $i$ 不依赖其他位置，所以 GPU 可以并行处理大量元素。033 还根据列号读取 bias，但输出元素之间仍没有依赖。

### CUDA thread、CUDA thread block（CTA）和 CUDA grid

**CUDA thread** 是执行一个元素计算的工作单元。本章通常让一个 CUDA thread 处理一个线性索引。

**CUDA thread block（CTA）** 是一组一起调度的 CUDA thread。本章固定每个 CUDA thread block（CTA）使用 256 个 CUDA thread。

**CUDA grid** 是一次 kernel launch 创建的全部 CUDA thread block（CTA）。处理 $N$ 个元素时，CUDA grid 包含 $(N+255)/256$ 个 CUDA thread block（CTA），除法取整。最后一个 CUDA thread block（CTA）可能没有填满，所以 kernel 用 index 小于 n 的条件保护尾部。

### Triton program instance 和 Triton block tensor

**Triton program instance** 是一次 Triton kernel 中独立执行的工作单元。本章每个 Triton program instance 处理 256 个连续位置。

**Triton block tensor** 是一个 Triton program instance 同时计算的一组值。共同索引形式如下：

~~~python
offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
mask = offsets < n_elements
~~~

load 和 store 使用 mask，避免尾部越界。Triton block tensor 是一组抽象逻辑值，不是 CUDA thread block（CTA），也不保证物理上始终位于寄存器；lowering 可能使用寄存器、shared memory，资源不足时还可能 spill 到设备内存。

### 连续张量

Python 接口只接受连续的 CUDA 或 ROCm 张量。连续张量的相邻逻辑元素按固定顺序存放，可以用 0 到 `numel-1` 的线性索引访问；`numel` 表示元素总数。除 033 外，kernel 不解释输入维度，只保留原 shape。

### 广播和激活函数

**广播**表示把一个标量或较小张量的值复用于多个输出位置。015 把一个 `scalar` 用于全部元素；033 根据列号让每一行复用同一个 bias。广播描述索引关系，不表示先在全局内存中复制出一个同 shape 张量。

**激活函数**是神经网络中逐元素改变数值的非线性函数。017–026 包含 ReLU、Sigmoid、GELU 等激活函数。它们控制哪些信号被保留、缩放或平滑门控。HardSigmoid 和 HardSwish 使用分段线性近似，避免 Sigmoid 所需的指数计算。

### 浮点特殊值

浮点格式除普通有限数外还包含 **Inf**、**NaN**、正零和负零。Inf 表示超出有限范围的无穷结果；NaN 表示没有有效实数结果，例如 `0/0`。正零和负零数值比较相等，但部分运算会保留符号。**Subnormal** 是比最小正规数更接近零的非零值，其精度和硬件处理可能不同。文档在相关算子的“验证与限制”中明确这些边界。

### GPU 内存和合并访问

**全局内存**是 GPU 上供所有 CUDA thread block（CTA）访问的大容量内存。它的访问延迟高于片上存储，所以简单逐元素 kernel 的性能通常取决于单位时间能读取和写回多少字节；这个速率称为**内存带宽**。相邻 CUDA thread 访问相邻地址时，硬件可以把请求合并为较少的内存事务，这称为**合并访问**。Triton 源码表达连续的逻辑地址，编译器负责把它 lowering 为硬件访问。

**shared memory** 是一个 CUDA thread block（CTA）内部共享的片上存储。本章算子没有跨输出元素的数据依赖，因此当前 CUDA 源码不声明 shared memory，也不需要 CUDA thread block（CTA）内部同步。Triton 源码同样不显式要求跨 Triton program instance 通信；编译器仍可根据 lowering 和资源需求选择实际存储位置。

## 共同算法和硬件映射

每个 Python 接口执行相同的基本流程：

1. 验证输入类型、dtype、shape、device 和连续性。
2. 用 `torch.empty_like` 分配新输出。
3. 空张量直接返回，不 launch kernel。
4. 非空张量启动 Triton kernel。
5. 每个 Triton program instance 加载 Triton block tensor、计算并 masked store。

两套实现采用不同的启动路径。下图只表示各自的层级，不表示一个 Triton program instance 必然对应一个 CUDA thread block（CTA）。

~~~mermaid
flowchart LR
    P[Python 包装函数] --> TC[Triton 启动配置]
    TC --> PI[Triton program instance]
    PI --> BT[Triton block tensor]
    BT --> M[GPU 全局内存]

    C[CUDA 调用方] --> L[CUDA launcher]
    L --> G[CUDA grid]
    G --> B[CUDA thread block（CTA）]
    B --> T[CUDA thread]
    T --> M
~~~

Python 公共文件 [python/_common.py](python/_common.py) 实际拥有以下符号：

- `BLOCK_SIZE`，值为 256；
- `require_float_tensor`；
- `require_pair`；
- `grid` 启动配置函数；
- Triton JIT helper `stable_sigmoid`；
- Triton JIT helper `stable_softplus`。

CUDA 公共代码分两层：

- [../../shared/cuda_common.cuh](../../shared/cuda_common.cuh) 属于整个项目，定义 `gpu_ops::kThreads`、`gpu_ops::ceil_div_int`、`CUDA_CHECK` 和归约 helper。
- [cuda/_common.cuh](cuda/_common.cuh) 属于本章，在 `gpu_ops::elementwise_detail` namespace 中定义 `validate_n`、`stable_sigmoid` 和 `stable_softplus`。

本章 CUDA launcher 和 kernel 位于 gpu_ops namespace；文件内 kernel 还位于匿名 namespace。CUDA 接口只接受 float 指针，即 FP32。Python/Triton 接口接受 FP16、BF16 和 FP32。

二元 Python 算子要求两个输入具有相同 shape、device 和 dtype。032 的 condition 必须是连续的 bool 或 uint8 GPU 张量，并匹配输入 shape 和 device。033 要求 x 的 shape 为 rows×cols，bias 长度为 cols。

CUDA launcher 使用 int 计数。033 检查 rows 和 cols 非负，并检查 rows×cols 不超过 INT_MAX。当前 CUDA launcher 对零工作量直接返回成功；对非零工作量不检查数据指针是否为空。

## 012 · Subtract

### 概念与用途

从一个张量逐元素减去另一个张量。

### 输入与输出

Python 接口输入连续的 `x` 和 `y`。两者必须具有相同 shape、device 和 dtype，dtype 可为 FP16、BF16 或 FP32。输出是新分配的同 shape、同 device、同 dtype 张量。CUDA launcher 输入三个 FP32 指针和元素数 `n`，结果写入调用方提供的 `out`。

### 算法

$z_i=x_i-y_i$。

### Triton 实现

require_pair 验证两个输入。kernel 加载两个 Triton block tensor，执行减法，再写入新输出。

~~~python
offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
mask = offsets < n_elements
a = tl.load(a_ptr + offsets, mask=mask)
b = tl.load(b_ptr + offsets, mask=mask)
tl.store(out_ptr + offsets, a - b, mask=mask)
~~~

第一行把 Triton program instance 编号转换为本块的 256 个线性位置。第二行标记真实元素。两个 load 读取相同位置的 `x` 和 `y`；最后一行只为有效位置写入差值。Python 入口先调用 `require_pair`，再用 `grid(x.numel())` 创建 Triton 启动配置。

### CUDA 实现

一个 CUDA thread 计算一个 FP32 差值。launcher 是 gpu_ops::launch_op012_sub。

~~~cuda
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < n) out[index] = x[index] - y[index];
~~~

`blockIdx.x` 选择 CUDA thread block（CTA），`threadIdx.x` 选择该 CUDA thread block（CTA）内的 CUDA thread。两者合成全局 index。边界判断让最后一个 CUDA thread block（CTA）的多余 CUDA thread 不访问内存。launcher 用 `ceil_div_int(n, kThreads)` 创建 CUDA grid，`kThreads` 固定为 256。

### GPU 硬件

每个有效输出需要从全局内存读取两个元素并写回一个元素，只执行一次减法。CUDA thread 的连续地址有利于合并访问；Triton 的连续逻辑地址为后端生成合并访问提供条件，物理映射由 lowering 决定。该 kernel 通常受内存带宽限制，而不是受减法吞吐量限制。

### 教程

先用长度 1 理解单元素结果，再用长度 256 观察完整计算块，最后用长度 257 观察尾部 mask。依次与 `torch.sub` 比较。

### 验证与限制

再加入 signed zero、Inf 和 NaN；Inf-Inf 产生 NaN。当前测试覆盖常规随机尾块，但不覆盖特殊值。Triton 支持三种 dtype，CUDA 只支持 FP32。

## 013 · Multiply

### 概念与用途

计算两个张量的 Hadamard 乘积。

### 输入与输出

Python 接口输入连续的 `x` 和 `y`。两者必须具有相同 shape、device 和 dtype，dtype 可为 FP16、BF16 或 FP32。输出是新分配的同 shape、同 device、同 dtype 张量。CUDA launcher 输入三个 FP32 指针和元素数 `n`，结果写入 `out`。

### 算法

$z_i=x_i y_i$。

### Triton 实现

加载两个同 shape Triton block tensor 并直接相乘。

~~~python
offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
mask = offsets < n_elements
a = tl.load(a_ptr + offsets, mask=mask)
b = tl.load(b_ptr + offsets, mask=mask)
tl.store(out_ptr + offsets, a * b, mask=mask)
~~~

索引、mask 和两个 load 与 012 相同。唯一的算术变化是 store 前的 a*b。入口仍用 require_pair 拒绝 shape、device 或 dtype 不一致的输入。

### CUDA 实现

一个 CUDA thread 执行一次 FP32 乘法。

~~~cuda
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < n) out[index] = x[index] * y[index];
~~~

每个 CUDA thread 读取两个 FP32 值并写一个 FP32 值。launcher 先用 validate_n 拒绝负 n，n 为零时不创建 CUDA grid，否则启动包含 256 个 CUDA thread 的 CUDA thread block（CTA）。

### GPU 硬件

每个有效输出读取两个元素、执行一次乘法并写回一个元素。连续访问可以合并为较少的全局内存事务；算术工作很少，因此普通大张量通常主要消耗内存带宽。

### 教程

先计算两个短向量的逐位置乘积，再把长度改为 257，确认最后一个 Triton program instance 和最后一个 CUDA thread block（CTA）只处理一个有效元素。

### 验证与限制

加入零、最大有限值、subnormal、Inf 和 NaN。零乘 Inf 产生 NaN，大幅值可能溢出。低精度 Triton 比 FP32 CUDA 更早溢出；当前测试只使用普通随机值。

## 014 · Divide

### 概念与用途

逐元素计算商。

### 输入与输出

Python 接口输入连续的被除数 `x` 和除数 `y`。两者必须具有相同 shape、device 和 dtype，dtype 可为 FP16、BF16 或 FP32。输出是新分配的同 shape、同 device、同 dtype 张量。CUDA launcher 使用 FP32 指针；接口不要求除数非零。

### 算法

$z_i=x_i/y_i$。

### Triton 实现

加载两个 Triton block tensor并使用除法。FP16 和 BF16 除法提升到 FP32 计算，再存回输入 dtype。

~~~python
offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
mask = offsets < n_elements
a = tl.load(a_ptr + offsets, mask=mask)
b = tl.load(b_ptr + offsets, mask=mask)
tl.store(out_ptr + offsets, a / b, mask=mask)
~~~

Triton block tensor 的每个逻辑元素计算同一位置的商。mask 只防止地址越界，不检查 b 是否为零。Triton 的类型规则把 FP16/BF16 除法提升为 FP32；store 再转换为 out 的 dtype。

### CUDA 实现

一个 CUDA thread 执行一次 FP32 除法。

~~~cuda
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < n) out[index] = x[index] / y[index];
~~~

CUDA 路径不做除零分支，也不改变 IEEE 特殊值。launcher 的 CUDA grid 和尾部规则与 012 相同。

### GPU 硬件

内存流量仍是两个输入加一个输出，但除法比加法或乘法具有更高延迟和更低吞吐量。大量独立 CUDA thread 或 Triton block tensor 的逻辑元素让 GPU 可以用其他就绪工作隐藏部分除法延迟；Triton 到硬件的物理映射由 lowering 决定。

### 教程

先用非零分母验证普通商，再把同一组分子分别除以正零、负零和极小非零数，观察 IEEE 754 结果。

### 验证与限制

显式构造正零、负零、极小分母、Inf 和 NaN。接口不拒绝零分母；非零数除以正零或负零可产生不同符号的 Inf，0/0 和 Inf/Inf 产生 NaN。当前随机测试没有这些定向输入。

## 015 · Scalar Add

### 概念与用途

给每个元素加同一个标量。

### 输入与输出

Python 接口输入连续的 FP16、BF16 或 FP32 张量 `x` 和 runtime 标量 `scalar`，输出是新分配的同 shape、同 device、同 dtype 张量。CUDA launcher 输入 FP32 指针、FP32 `scalar` 和元素数 `n`，结果写入 `out`。

### 算法

$z_i=x_i+s$。

### Triton 实现

scalar 是 runtime kernel 参数并广播到 Triton block tensor。wrapper 只验证 x，不提前验证 scalar 类型。

~~~python
values = tl.load(x_ptr + offsets, mask=mask)
tl.store(out_ptr + offsets, values + scalar, mask=mask)
~~~

offsets 和 mask 来自共同模板。values 是包含 256 个逻辑元素的 Triton block tensor；scalar 是一个 runtime 标量，Triton 自动把它广播到每个逻辑元素。入口用 torch.empty_like(x) 保持 shape 和 dtype。

### CUDA 实现

scalar 按 FP32 value 传入，每个 CUDA thread 执行一次加法。

~~~cuda
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < n) out[index] = x[index] + scalar;
~~~

scalar 由所有 CUDA thread 共享同一个值，但每个 CUDA thread 读取不同的 x[index]。launcher 将 scalar 作为 kernel 参数按值传递。

### GPU 硬件

每个有效输出只读取一个张量元素并写回一个元素。`scalar` 作为 kernel 参数传入，不需要为每个位置从另一个数组取值；该简单 kernel 通常受全局内存带宽限制。

### 教程

先给短向量加 1.25，确认标量广播到每个位置。再用空输入确认包装函数不会启动 CUDA grid 或 Triton kernel。

### 验证与限制

测试 0、负数、Inf、NaN 和空输入。NaN `scalar` 使有效结果为 NaN；不支持的 Python 对象在 Triton launch 参数专门化阶段报错。当前测试只使用 1.25。

## 016 · AXPBY

### 概念与用途

一次完成两个缩放和一次相加，避免创建中间张量。

### 输入与输出

Python 接口输入连续的同 shape、同 device、同 dtype 张量 `x` 和 `y`，以及 runtime 标量 `alpha` 和 `beta`。张量 dtype 可为 FP16、BF16 或 FP32。输出是新分配的同 shape、同 device、同 dtype 张量。CUDA launcher 的张量数据和两个系数均为 FP32。

### 算法

$z_i=alpha*x_i+beta*y_i$。

### Triton 实现

加载 x 和 y，在一个表达式中计算结果。wrapper 不验证 alpha 和 beta 的有限性。

~~~python
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
tl.store(out_ptr + offsets, alpha * x + beta * y, mask=mask)
~~~

alpha 和 beta 都是 runtime 标量，并分别广播到 x 和 y 的 Triton block tensor。表达式在一个 kernel 内完成，所以不会把两个缩放结果写成中间张量。

### CUDA 实现

每个 CUDA thread 源码包含两个乘法和一次加法。编译器是否生成 FMA 取决于工具链和选项，源码没有保证。

~~~cuda
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < n)
  out[index] = alpha * x[index] + beta * y[index];
~~~

每个 CUDA thread 读取 x 和 y 各一次并写出一次。alpha 和 beta 按值传入。源码表达式允许编译器优化，但文档不假定具体指令。

### GPU 硬件

一个 kernel 读取 `x` 和 `y` 后直接写出结果。若拆成两个缩放 kernel 和一个加法 kernel，还要读写中间张量；当前融合表达式减少全局内存流量和 kernel launch 次数。

### 教程

先令 `alpha=beta=1`，确认结果等于 `x+y`。再改变两个系数，检查同一输入位置如何经过两次缩放和一次相加得到输出。

### 验证与限制

测试系数 0、1、负数、Inf 和 NaN，并构造大幅值抵消。系数为零仍可能遇到零乘 Inf 等于 NaN。低精度 Triton 与 FP32 CUDA 的舍入不同；当前测试只使用两组有限系数。

## 017 · ReLU

### 概念与用途

把负数置零，保留非负数。

### 输入与输出

Python 接口输入一个连续的 FP16、BF16 或 FP32 GPU 张量 `x`，输出是新分配的同 shape、同 device、同 dtype 张量。CUDA launcher 输入 FP32 `x`、`out` 指针和元素数 `n`。

### 算法

当 $x<0$ 时输出 0，否则输出 x。

### Triton 实现

使用 tl.where(x < 0, 0, x)。NaN 比较为 false，因此保留 NaN。

~~~python
x = tl.load(x_ptr + offsets, mask=mask)
result = tl.where(x < 0.0, 0.0, x)
tl.store(out_ptr + offsets, result, mask=mask)
~~~

比较产生一个布尔 Triton block tensor。tl.where 按逻辑元素选择零或原值，不是 Python 控制流。masked store 仍负责尾部；算术条件不代替地址 mask。

### CUDA 实现

使用相同的小于零条件，每个 CUDA thread 处理一个 FP32 值。

~~~cuda
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < n)
  out[index] = x[index] < 0.0f ? 0.0f : x[index];
~~~

外层 if 是内存边界条件；三元表达式才是 ReLU 条件。两者作用不同。

### GPU 硬件

每个位置只需要一次读取、一次比较、一次选择和一次写回。选择通常可通过谓词执行，不要求不同 CUDA thread 跳转到两个独立 kernel 路径；连续读写仍使该 kernel 主要受内存带宽限制。

### 教程

用同时包含负数、零和正数的短向量运行两套实现。把地址边界条件与 ReLU 的数值条件分开观察：前者阻止越界，后者决定输出值。

### 验证与限制

验证负值、正负零、Inf 和 NaN。负零走保留分支，所以保留符号；这不同于某些不传播 NaN 的 `max` 写法。当前测试只使用有限数值采样。

## 018 · LeakyReLU

### 概念与用途

保留正侧，用 slope 缩放负侧。

### 输入与输出

Python 接口输入连续的 FP16、BF16 或 FP32 GPU 张量 `x`，以及默认值为 0.01 的 runtime 标量 `slope`。输出与 `x` 的 shape、device 和 dtype 相同。CUDA launcher 只处理 FP32 数据和 FP32 `slope`。

### 算法

当 $x>=0$ 时输出 x，否则输出 $s*x$。

### Triton 实现

使用 tl.where(x >= 0, x, slope*x)。默认 slope 为 0.01，wrapper 不限制符号或有限性。

~~~python
x = tl.load(x_ptr + offsets, mask=mask)
tl.store(out_ptr + offsets,
         tl.where(x >= 0.0, x, slope * x), mask=mask)
~~~

slope 广播到整个 Triton block tensor。比较为每个逻辑元素生成谓词；tl.where 选择原值或乘积。入口把默认 slope 设为 0.01，并直接作为 runtime 参数传入。

### CUDA 实现

一个 CUDA thread 使用相同条件选择 FP32 分支。

~~~cuda
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < n)
  out[index] = x[index] >= 0.0f ? x[index] : slope * x[index];
~~~

每个 CUDA thread 独立判断符号。slope 按 FP32 value 传入；launcher 不限制 slope。

### GPU 硬件

每个位置进行一次符号比较，并只在负侧使用一次乘法。Triton 的 `tl.where` 和 CUDA 的条件表达式表达逐位置选择；连续的一读一写使普通大张量通常受内存带宽限制。

### 教程

先用 `slope=0.2` 运行包含负数、零和正数的短向量，确认正侧不变、负侧缩小。再把 `slope` 改为 0 和 1，观察它分别接近 ReLU 和恒等映射。

### 验证与限制

测试负 `slope`、Inf、NaN 和正负零。当前定义让零直接走 `x` 分支；`slope` 为 Inf 或 NaN 时，PyTorch LeakyReLU 在正负零上得到 NaN，而当前实现返回原零。当前测试只用有限 `slope=0.2`。

## 019 · Sigmoid

### 概念与用途

把实数映射到 0 和 1 之间，并避免朴素公式在大负数上指数上溢。

### 输入与输出

Python 接口输入一个连续的 FP16、BF16 或 FP32 GPU 张量 `x`，输出是新分配的同 shape、同 device、同 dtype 张量。CUDA launcher 输入 FP32 指针和元素数 `n`。

### 算法

令 $e=exp(-|x|)$。非负分支计算 $1/(1+e)$，负分支计算 $e/(1+e)$。

### Triton 实现

python/_common.py 的 stable_sigmoid 在 FP32 中实现该分段公式。输入先提升到 FP32。

~~~python
@triton.jit
def stable_sigmoid(value):
    exponential = tl.exp(-tl.abs(value))
    return tl.where(
        value >= 0.0,
        1.0 / (1.0 + exponential),
        exponential / (1.0 + exponential),
    )
~~~

kernel 先把 load 结果转换为 tl.float32，再调用 helper。abs 保证指数参数不为正，两个分支共享 exponential。`tl.where` 按 Triton block tensor 的逻辑元素选择稳定公式，结果由 masked store 转回输出 dtype。

### CUDA 实现

gpu_ops::elementwise_detail::stable_sigmoid 使用 expf 和 fabsf 实现同一公式。

~~~cuda
const float exponential = expf(-fabsf(value));
return value >= 0.0f ? 1.0f / (1.0f + exponential)
                     : exponential / (1.0f + exponential);
~~~

每个 CUDA thread 把 x[index] 传给该 device helper。helper 位于 elementwise_detail namespace，不是项目级 shared header 的符号。

### GPU 硬件

指数、绝对值和除法比普通加法需要更多计算。实现先读取一个输入，复用 `exponential` 局部值，再写一个输出；局部复用避免第二次指数计算和额外全局内存访问。实际指数指令和精度由编译后端与 GPU 决定。

### 教程

从 `x=0` 开始确认输出为 0.5，再比较 `x=20` 和 `x=-20`。最后查看 `x=1000` 与 `x=-1000`，理解分段公式为什么不会计算 `exp(1000)`。

### 验证与限制

验证 0、正负 20、正负 1000、Inf、NaN 和三种 dtype。内部指数不会向正无穷上溢，NaN 传播，FP16 与 PyTorch 可能有最后一位舍入差异。当前 extreme 测试没有 Inf/NaN。

## 020 · Tanh

### 概念与用途

把实数映射到 -1 和 1 之间。

### 输入与输出

Python 接口输入一个连续的 FP16、BF16 或 FP32 GPU 张量 `x`，输出是新分配的同 shape、同 device、同 dtype 张量。CUDA launcher 输入 FP32 指针和元素数 `n`。

### 算法

当前 Triton 使用 $tanh(x)=2*sigmoid(2x)-1$。

### Triton 实现

复用 stable_sigmoid，并使用 FP32 临时值。

~~~python
x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
tl.store(out_ptr + offsets,
         2.0 * stable_sigmoid(2.0 * x) - 1.0, mask=mask)
~~~

内层 2*x 把输入送入 sigmoid 恒等式；外层乘二再减一得到 tanh。所有操作在 FP32 Triton block tensor 中执行，最后转换为输出 dtype。

### CUDA 实现

CUDA 不复用 sigmoid helper，而是直接调用 tanhf。

~~~cuda
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < n) out[index] = tanhf(x[index]);
~~~

每个 CUDA thread 调用一次 CUDA 数学函数。它与 Triton 的 sigmoid 恒等式是不同计算路径。

### GPU 硬件

Triton 路径复用 sigmoid helper，因此执行绝对值、指数、除法和选择；CUDA 路径调用 `tanhf`。两者都让每个 CUDA thread 或 Triton block tensor 的逻辑元素独立处理一个输入，但会使用不同的数学指令序列；Triton 到硬件的物理映射由 lowering 决定。

### 教程

先比较输入 -2、0 和 2，确认结果范围。再把输入缩小到正负 $10^{-8}$，同时打印结果和 signed zero，观察恒等式中的减法。

### 验证与限制

在零附近密集采样并检查 signed zero。两套实现不是同一算法：Triton 在零附近发生相减消去；FP32 `x=-1e-8` 可得到正零，并且负零不保留符号。CUDA `tanhf` 没有这条特定缺陷。当前线性网格不足以暴露该问题。

## 021 · GELU

### 概念与用途

使用 tanh 近似的 Gaussian Error Linear Unit。

### 输入与输出

Python 接口输入一个连续的 FP16、BF16 或 FP32 GPU 张量 `x`，输出与 `x` 的 shape、device 和 dtype 相同。CUDA launcher 输入 FP32 指针和元素数 `n`。本实现固定使用 tanh 近似，不提供 exact 模式参数。

### 算法

$f(x)=0.5*x*[1+tanh(0.7978845608*(x+0.044715*x^3))]$。

### Triton 实现

x 的三次方需要两次乘法，外层再乘一次 x；此外还有常数缩放和 sigmoid 形式的 tanh。不能把整条表达式简单描述成只有三次乘法，因为生成指令还取决于常数乘法和融合。

~~~python
x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
tanh_inner = 2.0 * stable_sigmoid(2.0 * inner) - 1.0
tl.store(out_ptr + offsets,
         0.5 * x * (1.0 + tanh_inner), mask=mask)
~~~

第一行保证中间计算为 FP32。第二行构造 tanh 的输入；第三行复用 020 的 sigmoid 恒等式；最后一行把 gate 乘回 x。每一行都作用于整个 Triton block tensor。

### CUDA 实现

CUDA 直接调用 tanhf，而不是 stable_sigmoid。

~~~cuda
const float value = x[index];
const float inner = 0.7978845608028654f *
    (value + 0.044715f * value * value * value);
out[index] = 0.5f * value * (1.0f + tanhf(inner));
~~~

每个 CUDA thread 先缓存自己的输入，再构造 inner 并调用 tanhf。缓存 value 避免反复书写全局内存表达式；实际寄存器和指令由编译器决定。

### GPU 硬件

GELU 的乘法、加法和 tanh 近似都在一个 kernel 中完成，因此中间 `inner` 和 `tanh_inner` 不写入全局内存。代价是每个有效位置比简单逐元素算子使用更多算术指令和临时值。

### 教程

先对 -2、0 和 2 计算公式中的 `inner`，再计算 tanh gate，最后乘回输入。用这三个中间步骤理解一个 Triton block tensor 如何保留整组临时值。

### 验证与限制

与 `torch.nn.functional.gelu(..., approximate="tanh")` 比较，重点采样零附近、饱和区、Inf 和 NaN。两套实现共享近似公式，但 tanh 路径不同。负无穷得到 NaN，正无穷得到 Inf，与 PyTorch tanh approximate 参考一致。当前测试只覆盖有限范围。

## 022 · SiLU

### 概念与用途

用 sigmoid 门控输入，也称 Swish。

### 输入与输出

Python 接口输入一个连续的 FP16、BF16 或 FP32 GPU 张量 `x`，输出与 `x` 的 shape、device 和 dtype 相同。CUDA launcher 输入 FP32 指针和元素数 `n`。

### 算法

$f(x)=x*sigmoid(x)$。

### Triton 实现

输入提升到 FP32，调用 stable_sigmoid，再与 x 相乘。

~~~python
x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
tl.store(out_ptr + offsets, x * stable_sigmoid(x), mask=mask)
~~~

同一个 FP32 Triton block tensor 既作为 sigmoid 输入，也直接参与最终乘法。一个 kernel 完成 gate 和乘法，不产生中间全局张量。

### CUDA 实现

调用 elementwise_detail::stable_sigmoid 后执行 FP32 乘法。

~~~cuda
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < n)
  out[index] = x[index] * elementwise_detail::stable_sigmoid(x[index]);
~~~

每个 CUDA thread 读取自己的输入，调用本章 CUDA helper，再写回一个结果。

### GPU 硬件

sigmoid 和最终乘法融合在一个 kernel 中。GPU 只需对输入执行一次全局内存读取并写回一次输出，不需要先把 sigmoid 结果保存为中间张量。指数和除法使算术成本高于 ReLU。

### 教程

先分别求 `sigmoid(x)` 和 `x*sigmoid(x)`，再运行融合实现并比较。这样可以区分数学上的两个步骤与硬件上的单次 kernel launch。

### 验证与限制

测试零附近、正负 20、Inf、NaN 和三种 dtype。负 Inf 乘零得到 NaN，与 PyTorch SiLU 一致。FP16 可能有最后一位差异；当前测试只覆盖有限数值采样。

## 023 · Softplus

### 概念与用途

提供 ReLU 的平滑版本。

### 输入与输出

Python 接口输入一个连续的 FP16、BF16 或 FP32 GPU 张量 `x`，输出与 `x` 的 shape、device 和 dtype 相同。CUDA launcher 输入 FP32 指针和元素数 `n`。当前接口没有可配置的 `beta` 或 `threshold` 参数。

### 算法

$f(x)=max(x,0)+log(1+exp(-|x|))$。

### Triton 实现

stable_softplus 实际使用 tl.log(1+tl.exp(...))。名称表达设计意图，但 log(1+u) 在很小的 u 上仍会发生消去。

~~~python
@triton.jit
def stable_softplus(value):
    return tl.maximum(value, 0.0) + \
        tl.log(1.0 + tl.exp(-tl.abs(value)))
~~~

kernel 将输入转换为 FP32，再调用 helper。maximum 提供大正数的线性部分；abs 让指数参数非正；log 项提供平滑修正。这里的 log(1+u) 正是负尾误差来源。

### CUDA 实现

elementwise_detail::stable_softplus 使用 log1pf(expf(...))，负尾更准确。

~~~cuda
return fmaxf(value, 0.0f) +
       log1pf(expf(-fabsf(value)));
~~~

每个 CUDA thread 把 x[index] 传入 helper。log1pf 专门计算 log(1+u)，因此小 u 不会先在加法中消失。

### GPU 硬件

最大值、绝对值、指数和对数在同一个 kernel 中执行。CUDA 的 `log1pf` 和 Triton 的 `tl.log(1+u)` 不是同一数值路径；硬件和数学库实现会影响吞吐量，但当前代码首先存在可观察的算法精度差异。

### 教程

先计算 `x=0` 和 `x=20`，再把输入改为 -10、-20 和 -100。分别打印 Triton、CUDA 参考公式和高精度参考，观察负尾何时变为零。

### 验证与限制

用高精度参考检查 -10 到 -100 的负尾，不要只用宽松绝对容差。Triton FP32 在 `x=-20` 可返回 0，而 PyTorch Softplus 约为 $2.06*10^{-9}$。这是已知错误，两套 helper 并不数值等价；当前测试的 `atol` 会掩盖错误归零。

## 024 · ELU

### 概念与用途

正侧保持线性，负侧使用指数曲线。

### 输入与输出

Python 接口输入连续的 FP16、BF16 或 FP32 GPU 张量 `x`，以及默认值为 1.0 的 runtime 标量 `alpha`。输出与 `x` 的 shape、device 和 dtype 相同。CUDA launcher 只处理 FP32 数据和 FP32 `alpha`。

### 算法

正数输出 x；非正数输出 $alpha*(exp(x)-1)$。

### Triton 实现

使用 tl.exp(x)-1，不是 expm1。输入提升到 FP32。

~~~python
x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
result = tl.where(x > 0.0, x,
                  alpha * (tl.exp(x) - 1.0))
tl.store(out_ptr + offsets, result, mask=mask)
~~~

比较生成正侧谓词。正侧选择 x；非正侧先计算指数差，再广播乘 alpha。tl.where 的两个表达式都已构造，不能把它理解为 Python 的短路 if。

### CUDA 实现

使用 expm1f，因此零附近通常更准确。

~~~cuda
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < n)
  out[index] = x[index] > 0.0f
      ? x[index] : alpha * expm1f(x[index]);
~~~

外层 if 保护地址；三元表达式选择 ELU 分支。alpha 是所有 CUDA thread 共用的 FP32 runtime 参数。

### GPU 硬件

正侧只保留输入，负侧还计算指数。分段选择在一个 kernel 内完成，不需要把正值和负值拆成两个数组；不同数值路径仍可能造成不同 CUDA thread 或 Triton block tensor 的逻辑元素执行成本不同。Triton 到硬件的物理映射由 lowering 决定。

### 教程

先用 `alpha=0.7` 比较负数、零和正数。再把输入收窄到零的两侧，并分别使用 Triton 的 `exp(x)-1` 与 CUDA 的 `expm1f(x)` 公式。

### 验证与限制

检查 signed zero，并测试有限、Inf 和 NaN `alpha`。Triton 对普通 `alpha=1` 和负零会丢失负号。`alpha` 为 Inf 或 NaN 且 `x` 为正负零时，当前两套分段公式得到 NaN；PyTorch ELU 保留输入零。当前测试只使用 `alpha=0.7`。

## 025 · HardSigmoid

### 概念与用途

用分段线性函数近似 sigmoid。

### 输入与输出

Python 接口输入一个连续的 FP16、BF16 或 FP32 GPU 张量 `x`，输出与 `x` 的 shape、device 和 dtype 相同。CUDA launcher 输入 FP32 指针和元素数 `n`。

### 算法

$f(x)=clip(x/6+0.5,0,1)$。

### Triton 实现

输入提升到 FP32，计算仿射值，再用嵌套 tl.where 截断。

~~~python
x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
value = x / 6.0 + 0.5
result = tl.where(value < 0.0, 0.0,
                  tl.where(value > 1.0, 1.0, value))
tl.store(out_ptr + offsets, result, mask=mask)
~~~

value 是未截断 gate。外层选择下界零；内层选择上界一或 value。两个比较和选择都按 Triton block tensor 的逻辑元素执行。

### CUDA 实现

每个 CUDA thread 使用相同的两个比较和三段选择。

~~~cuda
const float value = x[index] / 6.0f + 0.5f;
out[index] = value < 0.0f ? 0.0f
           : (value > 1.0f ? 1.0f : value);
~~~

代码位于 index<n 的边界分支内。每个 CUDA thread 只处理自己的 FP32 value。

### GPU 硬件

每个位置执行仿射变换、两次比较和选择。它不调用指数、对数等特殊数学函数；除法 `x/6` 可由编译器按目标硬件降低，但本项目未检查生成指令。连续的一读一写使大张量通常受内存带宽限制。

### 教程

先输入 -4、-3、0、3 和 4，逐步计算 `x/6+0.5`，再应用上下界。重点观察 -3 和 3 是三个线性区间的连接点。

### 验证与限制

围绕 -3 和 3 的两侧采样，并加入 Inf 和 NaN。负 Inf 输出 0，正 Inf 输出 1，NaN 穿透。当前测试只使用有限数值采样。

## 026 · HardSwish

### 概念与用途

用 hard sigmoid 门控输入。

### 输入与输出

Python 接口输入一个连续的 FP16、BF16 或 FP32 GPU 张量 `x`，输出与 `x` 的 shape、device 和 dtype 相同。CUDA launcher 输入 FP32 指针和元素数 `n`。

### 算法

$f(x)=x*clip(x/6+0.5,0,1)$。

### Triton 实现

输入提升到 FP32，先计算 gate，再执行 x*gate。

~~~python
x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
value = x / 6.0 + 0.5
gate = tl.where(value < 0.0, 0.0,
                tl.where(value > 1.0, 1.0, value))
tl.store(out_ptr + offsets, x * gate, mask=mask)
~~~

前两步与 025 相同。变化是保留 gate Triton block tensor，并在 store 前按逻辑元素乘回 x。

### CUDA 实现

每个 CUDA thread 按 FP32 计算同一表达式。

~~~cuda
const float value = x[index] / 6.0f + 0.5f;
const float gate = value < 0.0f ? 0.0f
                 : (value > 1.0f ? 1.0f : value);
out[index] = x[index] * gate;
~~~

与 025 相比，CUDA 多保存一个 gate 局部值并增加一次乘法。代码仍位于 index<n 分支中。

### GPU 硬件

HardSwish 把 HardSigmoid 的 gate 和最终乘法融合在同一 kernel 中。中间 gate 保留在 Triton 临时值或 CUDA 局部值中，不写入全局内存；因此仍只有一次输入读取和一次输出写入。

### 教程

复用 025 的 -4、-3、0、3 和 4，先得到 gate，再逐位置乘回输入。对比 025 可以直接看到多出的一次乘法如何改变输出。

### 验证与限制

测试 -3 和 3 两侧、Inf、NaN 和三种 dtype。负 Inf 乘零产生 NaN，正 Inf 乘一产生 Inf。FP16 与 PyTorch 的等价重排可能产生最后一位差异；当前 dtype 测试不包含特殊值。

## 027 · Square

### 概念与用途

逐元素求平方。

### 输入与输出

Python 接口输入一个连续的 FP16、BF16 或 FP32 GPU 张量 `x`，输出与 `x` 的 shape、device 和 dtype 相同。CUDA launcher 输入 FP32 指针和元素数 `n`。

### 算法

$z_i=x_i^2$。

### Triton 实现

直接计算 x*x，没有显式提升到 FP32。

~~~python
x = tl.load(x_ptr + offsets, mask=mask)
tl.store(out_ptr + offsets, x * x, mask=mask)
~~~

load 保留输入元素类型，所以 FP16、BF16 和 FP32 使用各自的 Triton 计算类型。两个乘数引用同一个 Triton block tensor。

### CUDA 实现

每个 CUDA thread 执行一次 FP32 乘法。

~~~cuda
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < n) out[index] = x[index] * x[index];
~~~

每个 CUDA thread 读一个逻辑元素并写一个平方结果。编译器可以把重复读取保存在寄存器中，但接口流量仍是一读一写。

### GPU 硬件

每个位置只执行一读、一乘和一写。没有跨位置依赖，也没有复杂数学函数；连续访问和很低的算术强度使普通大张量通常受全局内存带宽限制。

### 教程

先输入 -2、-0、0 和 2，确认符号与平方的关系。再把长度改为 257，检查尾部 mask 不改变有效元素的计算。

### 验证与限制

测试最大有限值、极小值、signed zero、Inf 和 NaN。FP16 和 BF16 可能更早溢出；正负 Inf 输出 Inf，NaN 传播，负零平方为正零。当前测试只覆盖有限范围。

## 028 · Square Root

### 概念与用途

逐元素求平方根。

### 输入与输出

Python 接口输入一个连续的 FP16、BF16 或 FP32 GPU 张量 `x`，输出与 `x` 的 shape、device 和 dtype 相同。CUDA launcher 输入 FP32 指针和元素数 `n`。两套接口都接受负输入，但浮点结果为 NaN。

### 算法

$z_i=sqrt(x_i)$。

### Triton 实现

输入提升到 FP32，再调用 tl.sqrt。

~~~python
x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
tl.store(out_ptr + offsets, tl.sqrt(x), mask=mask)
~~~

显式转换统一了三种输入 dtype 的中间精度。sqrt 对 Triton block tensor 的每个逻辑元素计算；store 再转换为输出 dtype。

### CUDA 实现

每个 CUDA thread 调用 sqrtf。

~~~cuda
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < n) out[index] = sqrtf(x[index]);
~~~

CUDA 接口本来就是 FP32，因此无需显式提升。没有定义域检查，负输入直接交给 sqrtf。

### GPU 硬件

每个位置独立调用平方根运算。平方根通常比乘加具有更高延迟；大量 CUDA thread 或 Triton block tensor 的逻辑元素提供并行工作，帮助 GPU 隐藏部分延迟。Triton 到硬件的物理映射由 lowering 决定。

### 教程

先输入 0、1、4 和 9，确认完全平方数。再加入负数，区分“接口允许输入”与“数学结果是有限实数”这两件事。

### 验证与限制

测试负数、正负零、完全平方数、Inf 和 NaN。负有限值和负 Inf 产生 NaN，负零保留符号，正 Inf 输出 Inf。接口不 clamp 负输入；当前测试与 benchmark 主动把输入改为正数。

## 029 · Exponential

### 概念与用途

逐元素计算自然指数。

### 输入与输出

Python 接口输入一个连续的 FP16、BF16 或 FP32 GPU 张量 `x`，输出与 `x` 的 shape、device 和 dtype 相同。CUDA launcher 输入 FP32 指针和元素数 `n`。

### 算法

$z_i=e^{x_i}$。

### Triton 实现

输入提升到 FP32，使用 tl.exp。NVIDIA 后端可把 FP32 tl.exp 降低为近似指数指令；它不是源码级 expf 等价保证。

~~~python
x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
tl.store(out_ptr + offsets, tl.exp(x), mask=mask)
~~~

一个 Triton program instance 对包含 256 个逻辑元素的 Triton block tensor 计算指数。显式 FP32 中间值避免直接对低精度 Triton block tensor 调用数学函数。

### CUDA 实现

每个 CUDA thread 调用 expf。

~~~cuda
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < n) out[index] = expf(x[index]);
~~~

每个 CUDA thread 独立调用 CUDA 数学函数。launcher 不设置溢出阈值，也不检查输出是否有限。

### GPU 硬件

每个位置独立执行指数函数。NVIDIA 后端可以把 Triton FP32 `tl.exp` 降低为近似指数指令，而 CUDA 使用 `expf`；源码不保证两条路径得到逐位相同的指令或结果。

### 教程

先输入 -1、0 和 1，建立自然指数的直观结果。然后逐渐增大正负输入，观察有限值何时溢出为 Inf 或下溢为零。

### 验证与限制

围绕 FP32 上溢和下溢边界采样，并测试 Inf/NaN。大正数溢出为 Inf，大负数下溢为零，NaN 传播。Triton 和 CUDA 的精确舍入可能不同；当前测试只覆盖温和有限范围。

## 030 · Logarithm

### 概念与用途

逐元素计算自然对数。

### 输入与输出

Python 接口输入一个连续的 FP16、BF16 或 FP32 GPU 张量 `x`，输出与 `x` 的 shape、device 和 dtype 相同。CUDA launcher 输入 FP32 指针和元素数 `n`。两套接口都不要求输入为正。

### 算法

$z_i=log(x_i)$。

### Triton 实现

输入提升到 FP32，再调用 tl.log。

~~~python
x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
tl.store(out_ptr + offsets, tl.log(x), mask=mask)
~~~

mask 只保护地址，不代表 `x` 为正。`tl.log` 对 Triton block tensor 的每个有效逻辑元素执行，定义域行为由浮点数学语义决定。

### CUDA 实现

每个 CUDA thread 调用 logf。

~~~cuda
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < n) out[index] = logf(x[index]);
~~~

CUDA 同样不做正值检查。调用方会收到 logf 对零、负数、Inf 和 NaN 的结果。

### GPU 硬件

每个位置独立执行对数函数。复杂数学函数的吞吐量低于普通加法，但这里仍没有跨位置通信；CUDA thread 和 Triton block tensor 的逻辑元素可以并行处理。Triton 到硬件的物理映射由 lowering 决定。

### 教程

先输入 1 和 $e$，确认自然对数。再加入零和负数，观察地址 mask 只防止越界，并不会修正对数定义域。

### 验证与限制

测试负数、正负零、1、subnormal、Inf 和 NaN。正负零输出负 Inf，负有限值和负 Inf 产生 NaN，正 Inf 输出 Inf。接口不添加 epsilon，也不 clamp；当前测试与 benchmark 都把输入改为 `abs(x)+0.01`。

## 031 · Clamp

### 概念与用途

把值限制在闭区间 low 到 high。

### 输入与输出

Python 接口输入连续的 FP16、BF16 或 FP32 GPU 张量 `x`，以及 runtime 标量 `low` 和 `high`。输出与 `x` 的 shape、device 和 dtype 相同。CUDA launcher 只处理 FP32 数据。当前两套接口都要求 `low <= high`，但不拒绝 NaN bound。

### 算法

当前接口要求 low 不大于 high，普通有限边界下执行先下限、再上限截断。

### Triton 实现

wrapper 在 low>high 时抛 ValueError。kernel 使用两个比较和嵌套 tl.where，没有调用 tl.clamp。

~~~python
x = tl.load(x_ptr + offsets, mask=mask)
result = tl.where(x < low, low,
                  tl.where(x > high, high, x))
tl.store(out_ptr + offsets, result, mask=mask)
~~~

low 和 high 是 runtime 标量，并广播到 Triton block tensor。第一个比较处理下界，第二个处理上界；两者都不处理 NaN bound。wrapper 的 low>high 检查发生在输出分配前。

### CUDA 实现

launcher 在 n<0 或 low>high 时返回 cudaErrorInvalidValue，kernel 使用相同嵌套比较。

~~~cuda
const float value = x[index];
out[index] = value < low ? low
           : (value > high ? high : value);
~~~

该代码位于 index<n 内。launcher 同时验证 n 和有限区间顺序，但不验证 bound 是否为 NaN。

### GPU 硬件

`low` 和 `high` 作为 kernel 参数广播到所有位置。每个位置只读一次输入，并用两次比较和选择完成截断；无需跨 CUDA thread 或跨 Triton program instance 通信。

### 教程

先用 `low=-1`、`high=1` 处理小向量，分别跟踪低于区间、位于区间和高于区间的值。再尝试相等边界与反向边界，观察 wrapper 的检查时机。

### 验证与限制

分别测试 NaN 输入、NaN `low`、NaN `high`、相等边界和反向边界。这不是完整 `torch.clamp` 语义：任一 bound 为 NaN 时，检查被绕过，kernel 退化为单边截断或保留输入，而 PyTorch 输出 NaN。当前接口还拒绝 `low>high`，而 PyTorch 会输出 `high`。NaN 输入本身会穿透当前比较。当前测试只覆盖普通有限边界和拒绝反向边界。

## 032 · Where

### 概念与用途

根据 condition 从 x 或 y 逐元素选择。

### 输入与输出

Python 接口输入连续的 `condition`、`x` 和 `y`。`condition` 必须是 bool 或 uint8，`x` 和 `y` 必须是同 shape、同 device、同浮点 dtype；三者的 shape 和 device 必须一致。输出与 `x` 的 shape、device 和 dtype 相同。CUDA launcher 的 condition 是 byte 指针，候选和输出是 FP32 指针。

### 算法

condition 非零时输出 x，否则输出 y。

### Triton 实现

加载 condition、x 和 y 三个 Triton block tensor，再执行 tl.where(condition != 0, x, y)。wrapper 接受 bool 和 uint8 condition。

~~~python
condition = tl.load(condition_ptr + offsets, mask=mask)
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
tl.store(out_ptr + offsets,
         tl.where(condition != 0, x, y), mask=mask)
~~~

三个 load 使用同一 `offsets` 和地址 mask。`condition != 0` 产生选择谓词；`tl.where` 按 Triton block tensor 的逻辑元素选择候选。即使某个逻辑元素最终只选一个候选，源码仍加载了两个候选。

### CUDA 实现

condition 是 const unsigned char 指针。每个 CUDA thread 读取一个 byte condition 和两个 FP32 候选。

~~~cuda
const int index = blockIdx.x * blockDim.x + threadIdx.x;
if (index < n)
  out[index] = condition[index] != 0 ? x[index] : y[index];
~~~

任何非零 byte 都选择 x。CUDA 参数类型不能区分 PyTorch bool 与 uint8；它只看到一个 byte 数组。

### GPU 硬件

Triton 源码对每个位置读取一个 byte condition、两个候选并写一个输出；FP32 的显式流量是每元素 13 byte。CUDA 条件表达式在语言语义上只求值被选候选，编译器可使用分支或谓词化选择，实际读取哪些候选要以生成代码为准。两条路径的连续地址都有利于合并全局内存访问。

### 教程

用交替真假的短 condition 运行两套实现，逐位置检查选择结果。再把 condition 改为 uint8 的 0 和非零值，确认非零即为真。

### 验证与限制

在候选的不同位置放置 NaN 和 Inf。未选中的 NaN 不进入结果；Triton 源码仍显式加载两个候选，CUDA 源码不保证读取未选候选。`condition` 不是 Tensor 时，wrapper 当前因访问 `dtype` 抛 `AttributeError`，而不是明确的 `TypeError`。当前测试只使用 bool condition 和普通有限候选。

## 033 · Row Bias Add

### 概念与用途

给二维矩阵每一行加同一个列 bias。

### 输入与输出

Python 接口输入二维连续张量 `x` 和一维连续张量 `bias`。两者必须位于同一 device、使用相同的 FP16、BF16 或 FP32 dtype，且 `bias` 长度必须等于 `x` 的列数。输出与 `x` 的 shape、device 和 dtype 相同。CUDA launcher 输入 FP32 指针以及 `rows`、`cols`。

### 算法

$z_{r,c}=x_{r,c}+b_c$。

### Triton 实现

矩阵被线性化，用 offsets % cols 得到列号。wrapper 要求 x 二维、bias 一维、bias 长度等于列数，并要求 device 和 dtype 相同。

~~~python
offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)
bias = tl.load(bias_ptr + offsets % cols, mask=mask)
tl.store(out_ptr + offsets, x + bias, mask=mask)
~~~

线性 `offset` 除以 `cols` 的余数就是列号。多个 Triton block tensor 逻辑元素可以读取同一个 bias 元素，但各自写不同输出。入口只在 `x.numel()` 非零时 launch，因此零列不会执行取余。

### CUDA 实现

kernel 用 index % cols 选择 bias。launcher 检查负维度和 rows×cols 的 int 溢出；零行或零列提前返回。

~~~cuda
const int index = blockIdx.x * blockDim.x + threadIdx.x;
const int n = rows * cols;
if (index < n)
  out[index] = x[index] + bias[index % cols];
~~~

launcher 在乘法前验证 rows 不超过 INT_MAX/cols，然后才计算 n。由于零列提前返回，kernel 中的模运算不会以零为除数。

### GPU 硬件

每个输出读取一个 `x` 元素和一个 bias 元素。相邻位置连续读取 `x`，但 bias 索引每行循环；同一 bias 会被多行重复使用，能否命中 cache 取决于矩阵宽度和硬件。整数取余也比纯线性索引增加计算。

### 教程

先使用 2 行 3 列矩阵和长度 3 的 bias，手工展开线性 `offset` 与列号 `offset % cols`。再使用 257 列，观察同一映射如何跨过计算块边界。

### 验证与限制

测试零行、零列、非 256 倍数列宽、NaN/Inf bias 和错误 shape/device/dtype。Python 的零列矩阵配空 bias 可通过 shape 检查，并因 `numel` 为零安全返回；CUDA 同样对零列提前返回。非零工作量下所有 CUDA 指针都必须有效，但 launcher 没有检查。当前测试覆盖零行、257 列和错误 bias 长度。

## 测试实际覆盖范围

[tests/test_02_elementwise.py](../../tests/test_02_elementwise.py) 只执行 Python/Triton 接口。它覆盖：

- 012–014 的长度 1、31、256、257 和 1003；
- 015–016 的一组有限参数；
- 017–030 的有限输入参考比较；
- 019 和 023 的有限大幅值；
- 012、016、026 和 029 的三种 dtype；
- 若干空张量和无效 shape/dtype。

它不编译或运行 CUDA 文件。[tests/test_catalog_static.py](../../tests/test_catalog_static.py) 只检查编号、局部 kernel 定义和源码中存在直接 launch；它不能证明 CUDA 可以编译或正确执行。

测试 helper assert_close 对 FP16/BF16 使用 rtol=2e-2、atol=2e-2，对其他 dtype 使用 rtol=2e-4、atol=2e-5。它没有设置 equal_nan=True。因此 actual 和 expected 同位置都为 NaN 时仍失败。不能声称现有 helper 已支持 NaN 等价比较。

本章“教程”中的验证步骤是建议，不代表自动测试已经覆盖。

## Benchmark 实际能力

统一 benchmark 位于 [benchmarks/run.py](../../benchmarks/run.py) 和 [benchmarks/cases.py](../../benchmarks/cases.py)。它只加载 Python/Triton 实现，不编译或运行 CUDA 文件。

~~~shell
python3 benchmarks/run.py --op 23 --size medium --check
~~~

size 可选 small、medium 或 large，对应长度 $2^{14}$、$2^{20}$ 和 $2^{24}$。check 在计时前执行一次 Triton 与 PyTorch reference 比较；没有 check 时不检查正确性。

计时使用 triton.testing.do_bench，分别测 Triton callable 和 PyTorch reference。speedup 只表示当前环境中这两个 callable 的计时比，不代表 CUDA 对比、端到端应用加速或跨设备保证。

带宽是估算值。032 当前按 4*x.nbytes 计数。对 FP32 x、y、out 和 byte condition，显式流量是每元素 13 byte，而公式按 16 byte，报告带宽约高估 23%。

Benchmark 输入主要是普通 FP32 随机数据。028 和 030 主动改成正数。它不能验证特殊值、低精度或大张量限制。

## 已知限制

以下问题来自当前代码审查，尚未在源码中修复：

1. **Triton 大张量索引溢出。** 012–033 的 Triton program instance ID 和 arange 形成 int32 offset。线性 offset 超过 INT_MAX 时可能回绕，产生错误结果或越界访问；wrapper 不限制 numel。
2. **CUDA 不检查非零工作量的空指针。** launch 可能先返回成功，再异步报告非法访问。零工作量提前返回，所以允许空指针。
3. **Triton Softplus 负尾错误归零。** stable_softplus 使用 log(1+u)，不是 log1p(u)。CUDA helper 不受此特定问题影响。
4. **Clamp 的 NaN bound 错误。** Python 和 CUDA 都只检查 low>high；NaN bound 绕过检查且不会传播到所有输出。
5. **多 GPU current device 未处理。** wrapper 验证输入同 device，但没有进入该 device 上下文。输入在 cuda:1、current device 为 cuda:0 时可能用错误 device 或 stream launch。
6. **Triton Tanh 零附近消去。** sigmoid 恒等式损失小量并把负零变为正零。
7. **非有限 activation 参数存在零点差异。** LeakyReLU 的非有限 slope 和 ELU 的非有限 alpha 在正负零上与 PyTorch 不一致。
8. **ELU Triton 零附近精度较弱。** Triton 使用 exp(x)-1，CUDA 使用 expm1f。
9. **Where 非 Tensor 错误不受控。** condition 不是 Tensor 时抛 AttributeError。
10. **CUDA 缺少动态验证。** 当前测试不编译 CUDA，也不执行 launcher 错误路径、特殊值和边界。
11. **NaN reference 比较未启用。** 公共 helper 没有 equal_nan=True。
12. **Where 带宽估算偏高。** benchmark 把 byte condition 按 FP32 大小计入。

## 建议验证顺序

1. 先运行 012，观察长度 257 的尾块。
2. 再运行 032，理解三个输入 Triton block tensor 和逐位置选择。
3. 运行 033，理解线性 offset 如何映射到列号。
4. 对 019、020、023 和 024 增加零附近与极值输入。
5. 最后为 CUDA launcher 添加独立编译与运行测试，不要用 Triton 测试结果替代 CUDA 验证。
