# GPU、CUDA、Triton 与 MLIR 基础

GPU 通过大量并行执行单元处理数据。CUDA 让程序员直接组织 CUDA thread 和片上资源；Triton 让程序员描述块级张量计算，再由编译器完成更多硬件映射。先建立本页的术语，再进入 100 个算子，能避免把 Triton 的逻辑张量误当成固定数量的物理 CUDA thread。

## 0. 阅读前先固定术语

| 名称 | 含义 | 在本项目中的作用 |
|---|---|---|
| **CPU** | Central Processing Unit，中央处理器 | 运行 Python、检查参数、分配输出并启动 GPU 工作 |
| **GPU** | Graphics Processing Unit，图形处理器 | 并行执行算术、访存、归约和矩阵运算 |
| **Host** | 发起设备工作的主机端，通常指 CPU 及其内存 | 执行 Python wrapper 和 CUDA launcher |
| **Device** | 执行加速程序的设备端，本项目指 GPU 及其内存 | 执行编译后的 kernel |
| **Tensor** | 具有 shape、dtype 和存储的一组多维数据 | 保存算子的输入、输出、权重和状态 |
| **Operator** | 把输入按一条数学规则变成输出的计算 | 本项目中的 Vector Add、Softmax、Matmul 等编号条目 |
| **Kernel** | 在 GPU 上执行的设备函数 | 一次 launch 让许多并行执行单元处理一个算子的工作 |
| **Wrapper** | 在 Host 上准备并调用 kernel 的函数 | 检查输入、分配输出、按后端计算 CUDA grid 或 Triton launch grid，并传入参数 |
| **CUDA** | NVIDIA 的 GPU 编程平台和编程模型 | 显式描述 CUDA thread、CUDA thread block（CTA）、内存和同步 |
| **Triton** | 面向并行张量计算的语言和编译器 | 用 Triton program instance 与 Triton block tensor 描述计算 |
| **MLIR** | Multi-Level Intermediate Representation，多层中间表示基础设施 | 承载 Triton 从高层张量语义到目标代码的分阶段编译 |

**Shape** 是 tensor 各维的长度，例如 `[rows, cols]`。**Dtype** 是每个元素的数据类型，例如 FP32、FP16 或 INT8。FP 表示 floating point（浮点数），INT 表示 integer（整数），后面的数字表示位数。

## 1. 先理解 GPU 为什么快

CPU 擅长低延迟和复杂控制流。GPU 用更多执行单元换取高吞吐，适合对大量数据重复执行相似运算。**吞吐**表示单位时间完成的总工作量；**延迟**表示一次工作从开始到结束的时间。**Reduction（归约）**把多个值合并为较少结果，例如求和。**Activation（激活函数）**对神经网络中的数值施加非线性变换。**Attention（注意力）**用 query 与 key 的相关性对 value 加权。矩阵乘、归约、激活函数和 Attention 都符合 GPU 的吞吐导向特点。

一次 GPU kernel 启动包含以下层级：

```text
Host → CUDA grid → CUDA thread block（CTA）→ warp → CUDA thread
                         ↓
                         在某个 SM 上执行
```

- **Host**：通常是 CPU，负责分配 tensor、设置参数和启动 kernel。
- **CUDA grid**：一次 kernel launch 创建的全部 CUDA thread block（CTA）。
- **CUDA thread block（CTA）**：可以通过 shared memory 和 barrier 协作的一组 CUDA thread。
- **Warp**：NVIDIA GPU 上由 32 个 lane 组成，是调度和指令发射的重要单位；lane 是 warp 中一个 CUDA thread 的位置。
- **SM**：Streaming Multiprocessor，包含 warp scheduler、CUDA Core、寄存器和 shared memory；支持矩阵加速的较新架构还包含 Tensor Core。

**Global memory** 是容量较大的 GPU 设备内存，tensor 通常存放在这里。**Shared memory** 是同一 CUDA thread block（CTA）共享的片上存储。**Register** 是每个 CUDA thread 私有的片上存储。**Barrier** 要求一组执行者都到达某点后再继续。**Atomic operation** 把一次读改写作为不可分割的更新，用于多个执行者竞争同一地址的场景。

普通 kernel 中的 `__syncthreads()` 只能同步同一 CUDA thread block（CTA）。跨 CUDA thread block（CTA）协作通常需要 atomic、多个 kernel，或重新划分数据所有权。Cooperative launch（协作式启动）也可在满足硬件和启动约束时提供 CUDA grid 级同步。

## 2. CUDA 和 Triton 的编程模型

CUDA 是 CUDA thread 中心模型。程序员通常先写一个 CUDA thread 对一个或几个标量做什么，再决定 CUDA thread block（CTA）大小、shared memory 和同步。

```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
if (i < n) {
  out[i] = x[i];
}
```

Triton 是块级张量模型。**Triton program instance** 是 Triton launch grid 中一个独立的逻辑程序实例。**Triton block tensor** 是该实例一次表达和计算的一组值。`tl.arange` 创建逻辑索引 Triton block tensor，并不等于创建同样数量的物理 CUDA thread。

```python
pid = tl.program_id(0)
offsets = pid * BLOCK + tl.arange(0, BLOCK)
mask = offsets < n
values = tl.load(x_ptr + offsets, mask=mask)
tl.store(out_ptr + offsets, values, mask=mask)
```

| Triton 构造 | CUDA 近似对应 | 关键区别 |
|---|---|---|
| `kernel[grid]` | kernel launch | 都把工作提交给设备 stream；stream 是按序提交 GPU 工作的队列 |
| Triton program instance | 默认 `num_ctas=1` 时近似一个 CUDA thread block（CTA） | 固定 NVIDIA 后端仅在 SM90+ 上支持 `num_ctas>1` 的 CUDA thread block cluster |
| `tl.program_id(0)` | `blockIdx.x` | 选择当前工作块 |
| `tl.arange(0, B)` | 一组 CUDA thread 索引 | 它是逻辑索引张量，不是 B 个 CUDA thread |
| mask | 边界判断或 predication | 防止尾块越界访存 |
| `tl.sum`、`tl.max` | warp/shared reduction | 具体 lowering 取决于 layout 和目标 |
| `tl.dot` | FMA、MMA、WGMMA 或 MFMA | dtype、shape、layout 和架构共同决定指令 |
| `tl.constexpr` | 模板参数 | 在编译期专门化 kernel |

默认 `num_ctas=1` 时，一个 Triton program instance 通常近似对应一个 CUDA thread block（CTA）。固定 NVIDIA 后端仅在 SM90+ 上允许 `num_ctas>1`，此时一个 Triton program instance 可使用由多个 CUDA thread block（CTA）组成的 cluster。其他后端只有在自身报告支持 multi-CTA launch 时才能使用相应能力。这个对应关系只用于理解执行粒度，不保证生成逐指令相同的代码。

表中的 **predication（谓词执行）**表示用真假条件控制一条指令是否生效。FMA 是 fused multiply-add（融合乘加）；MMA、WGMMA 和 MFMA 是不同目标上的矩阵乘加指令族。Layout 描述逻辑元素如何分布到硬件和存储。Lowering 表示把高层操作逐步转换为更接近硬件的低层操作。

## 3. Triton 最核心的能力

Triton 的核心不是用 Python 改写 CUDA 语法，而是让程序员保留块级计算结构。Tile 是一块待处理数据；broadcast 把较小值扩展到多个逻辑位置；reduction 合并多个值；scan 为每个位置保留前缀结果；pointer 表示内存地址；dot 表示点积或块矩阵乘。编译器再把这些结构映射到 warp、物理 lane、寄存器、shared memory 和目标指令。

这使调优重点从“每个 CUDA thread 执行哪一行标量代码”转为以下问题：

- 一个 Triton program instance 处理多大的 tile；
- 连续逻辑维度是否对应连续内存；
- 中间值能否在片上复用；
- reduction 或 dot 使用多少 warp；
- pipeline（让搬运与计算重叠的软件流水）有多少 stage；
- 不同 shape 应选择哪组编译期参数。

## 4. MLIR 在 Triton 中的作用

编译器不会直接把 Python 文本逐行翻译成 GPU 指令。它先建立中间表示，简称 IR。IR 是一种适合分析和改写的程序形式。

MLIR 是构建多层 IR 的基础设施。它提供三个重要概念：

- **方言（dialect）**：一组有明确语义的操作和类型。不同方言描述不同抽象层。
- **pass**：读取一层 IR，分析或改写其中的操作。例如折叠常量、选择 layout 或安排流水。
- **lowering**：把高层操作逐步变成更接近目标硬件的低层操作。

Triton 使用 MLIR 的价值是延迟丢失结构信息。若过早把 `tl.dot` 拆成大量标量乘加，后续 pass 就更难识别矩阵 tile、选择矩阵指令或安排数据搬运。Triton 因此在较高层保留 tensor shape、layout、reduction axis 和 memory operation，再逐步确定硬件细节。

### 4.1 从调用到机器代码

第一次调用一个新 kernel 配置时，Triton JIT（just-in-time，即运行时按需编译）会结合函数代码、`tl.constexpr` 参数、dtype 和目标架构生成专用版本。相同编译键之后通常可复用缓存；改变 shape 相关元参数、dtype 或目标可能触发新的编译。

固定版本先把 `@triton.jit` kernel 变成 TTIR（Triton IR），再变成含 GPU layout 的 TTGIR（Triton GPU IR），然后变成 LLIR（LLVM IR）。NVIDIA 后端继续生成 PTX 虚拟指令和 cubin 二进制；AMD 后端生成 AMDGCN 和 HSACO code object。准确阶段来自当前固定 Triton 源码中的 backend stage 注册，不应把 NVIDIA 的 PTX/cubin 路径套到 AMD。

```mermaid
flowchart LR
  A[@triton.jit kernel 源码] --> B[JIT 编译键]
  P[Python wrapper] -->|参数、Triton launch grid、触发编译| B
  B --> C[TTIR]
  C --> D[TTGIR]
  D --> E[LLIR]
  E --> F{目标后端}
  F -->|NVIDIA| G[PTX → cubin]
  F -->|AMD| H[AMDGCN → HSACO]
  G --> I[运行时加载并 launch]
  H --> I
  P -->|运行时参数| I
```

Python wrapper 不在设备上执行。它负责检查 shape 和 dtype、分配输出、计算 Triton launch grid，并把运行时参数传给已编译 kernel。设备只执行编译后的 GPU 程序。

### 4.2 每层决定什么

| 层 | 仍能看见的信息 | 典型决定 |
|---|---|---|
| Python wrapper | tensor shape、dtype、device、用户参数 | 输入契约、输出分配、Triton launch grid、meta-parameter |
| TTIR | Triton block tensor 运算和内存语义 | 常量折叠、广播和高层运算简化 |
| TTGIR | GPU layout、warp、CUDA thread block（CTA）、shared memory | 数据如何分给 lane、是否需要 layout conversion、pipeline |
| LLIR | 低层控制流、地址和目标 intrinsic | 目标相关优化和指令选择准备 |
| PTX/AMDGCN | 接近硬件的目标指令 | 寄存器、访存、同步和矩阵指令形式 |
| cubin/HSACO | 可加载二进制 | 在具体 GPU 上执行 |

### 4.3 一条 Triton 表达式如何变化

以 `tl.sum(values, axis=0)` 为例：

1. TTIR 知道它是沿某个 tensor 轴的 reduction。
2. TTGIR 已有 layout，因此能判断数据分布在哪些 lane 和 warp。
3. lowering 可选择寄存器内操作、warp shuffle、shared-memory partial 或其他目标实现。
4. 低层 IR 不再需要保留一条抽象的“tensor sum”，而是包含具体通信和算术操作。

`tl.dot` 也遵循这个过程。高层 IR 保留矩阵块关系；TTGIR 结合 layout 和目标能力；后端在条件合法时选择 MMA、WGMMA 或 MFMA，否则使用其他 dot/FMA 实现。

### 4.4 编译期和运行期的分界

`tl.constexpr` 值在编译期已知，可以控制循环展开、tile 形状和分支删除。普通 tensor 数据在运行期才存在。下面两种参数承担不同角色：

- `BLOCK_SIZE=256`：编译期 meta-parameter，改变 IR 形状并可能产生一个新 kernel 版本。
- `n_elements`：运行期标量，用来生成 mask，同一已编译版本可处理多个实际长度。

Autotune 会为同一问题试运行多组 meta-parameter，然后缓存较合适的配置。它优化的是候选集合中的实现，不保证找到全局最优方案。

### 4.5 MLIR 不负责什么

MLIR 和 Triton pass 不会自动修复错误算法、错误地址公式或不合理 benchmark。它们也不保证中间值一定放在寄存器、不发生 spill，或 `tl.dot` 一定使用 Tensor Core。最终结果仍取决于：

- 输入 dtype 和 shape；
- tensor layout 与地址连续性；
- `num_warps`、`num_stages` 和 tile 大小；
- 寄存器与 shared-memory 压力；
- GPU 架构、driver 和工具链版本。


## 5. GPU 存储层级

| 存储 | 主要范围 | 特点 | 常见用途 |
|---|---|---|---|
| Register | 单个 CUDA thread | 延迟低、容量有限 | 局部值和 accumulator |
| Shared memory | 单个 CUDA thread block（CTA） | 显式协作、有 bank | tile 和跨 warp reduction |
| L1 cache | 单 SM | 缓存 global load | 时间和空间局部性 |
| L2 cache | 全 GPU | 所有 SM 共享 | CUDA thread block（CTA）之间的数据复用 |
| HBM/GDDR | 全 GPU | 容量大、延迟高 | 输入、输出和模型参数 |

Cache（缓存）自动保留近期数据，L1 更靠近单个 SM，L2 供全 GPU 共享。寄存器过多会减少一个 SM 能同时驻留的 warp。发生 register spill（寄存器溢出）时，编译器把部分值放到所谓 local memory；它实际位于设备内存，延迟会明显增加。

## 6. 合并访存和 shared-memory bank

同一 warp 的 lane 访问连续地址时，硬件可以把访问合并为较少的内存事务。大 stride 或随机索引会增加事务数和无效传输。Triton 能分析 pointer tensor 的连续性，但程序员仍应让最快变化的逻辑维度对应连续地址。

Shared memory 被分为多个可并行服务的 bank。同一 warp 的多个 lane 同时访问不同地址却命中同一 bank 时，访问可能串行。CUDA 常用 padding（增加不用的元素）或 swizzle（重排地址映射）减少这种 bank conflict：

```cuda
__shared__ float tile[32][33];
```

Triton 后端可以通过 layout 和 shared encoding 安排存储，但最终是否避免冲突应查看生成代码或 profiler，而不是只根据源码推测。

## 7. 分支、mask 和同步

同一 warp 的 lane 走不同长分支时，硬件可能串行执行不同路径，这称为 warp divergence。简单边界条件通常可降低为 predication。

```cuda
if (i < n) out[i] = x[i];
```

```python
tl.store(out + offsets, values, mask=offsets < n)
```

mask 只保护相应操作，不是 CUDA thread block（CTA）barrier，也不能同步不同 Triton program instance。

## 8. Occupancy 和延迟隐藏

**Occupancy（占用率）**是一个 SM 上活跃 warp 数与该架构允许的最大活跃 warp 数之比。一个 SM 能同时驻留多少 CUDA thread block（CTA），受 CUDA thread 数、warp 数、寄存器、shared memory 和架构上限共同约束。下面各项都表示一种资源允许的 CUDA thread block（CTA）数量上限：

$$
N_{\text{resident}}=\min\left(
\left\lfloor\frac{T_{\text{SM}}}{T_{\text{CTA}}}\right\rfloor,
\left\lfloor\frac{R_{\text{SM}}}{R_{\text{CTA}}}\right\rfloor,
\left\lfloor\frac{S_{\text{SM}}}{S_{\text{CTA}}}\right\rfloor,
N_{\text{CTA,arch}}
\right)
$$

$T$、$R$、$S$ 分别表示 CUDA thread、寄存器和 shared memory 的资源量；下标 `SM` 表示每个 SM 可用量，下标 `CTA` 表示每个 CUDA thread block（CTA）的用量。

当一个 warp 等待设备内存时，scheduler 可以运行另一个 ready warp。较高 occupancy 有助于隐藏延迟，但不是越高越好。为了提高 occupancy 而过度减少寄存器，可能导致 spill。

Triton 常用调参包括：

- `BLOCK_SIZE` 或 `BLOCK_M/BLOCK_N/BLOCK_K`：单个 Triton program instance 的工作量；
- `num_warps`：单个 Triton program instance 的并行资源；
- `num_stages`：软件流水深度；
- `num_ctas`：固定 NVIDIA 后端在 SM90+ 上的 CUDA thread block cluster 配置；其他目标必须由后端报告支持。

## 9. Roofline 和算术强度

Roofline 是用计算峰值与内存带宽上限判断性能瓶颈的模型。BW 表示 bandwidth（带宽），即单位时间可传输的字节数。

算术强度是计算次数与设备内存传输字节数之比：

$$
I=\frac{\text{浮点运算次数}}{\text{设备内存传输字节数}}
$$

可达到的性能近似受计算峰值和内存带宽共同限制：

$$
P\leq\min\left(P_{\text{compute peak}},I\cdot BW_{\text{memory}}\right)
$$

逐元素算子通常更容易受带宽限制。足够大的 GEMM 通常更容易受计算吞吐限制。fusion 的主要价值是避免中间张量往返设备内存；中间值具体位于寄存器、shared memory 还是发生 spill，取决于 lowering 和资源压力。

## 10. Reduction 和 scan

Reduction 把多个元素合并为较少结果，例如 sum、max 和 norm。CUDA 常按以下层级实现：

1. 每个 CUDA thread 局部累加。
2. warp 内使用 shuffle。
3. 每个 warp 把 partial 写入 shared memory。
4. 一个 warp 合并这些 partial。

Triton 用 `tl.sum` 或 `tl.max` 表达归约轴。编译器根据 tensor layout 实现物理 lane 和 warp 间通信。跨 Triton program instance 的 reduction 仍需要 atomic 或第二个 kernel。

Scan 产生每个前缀的结果，例如 inclusive cumsum。它与 reduction 不同：reduction 只保留最终合并值，scan 保留每个位置的前缀值。

## 11. Tensor Core 和分块 GEMM

GEMM 是 General Matrix Multiplication（通用矩阵乘），计算矩阵乘积并可累加已有矩阵。Tensor Core 是面向小矩阵乘加的专用硬件执行单元。其核心运算可写为：

$$
D=A\times B+C
$$

高性能 GEMM 将矩阵分成 CUDA thread block（CTA）tile、warp tile 和指令 tile。它还要重叠 global load、shared-memory copy 和矩阵乘加。CUDA 通常需要显式处理 operand layout、异步 copy、barrier 和 fragment。Triton 用 `tl.dot` 保留矩阵运算语义，再由目标后端选择合法指令。

使用 `tl.dot` 不等于必然使用 Tensor Core。必须同时满足目标架构、dtype、tile 和 layout 条件。

## 12. 正确性和性能验证

一个可靠测试应根据算子契约选择边界，而不是机械要求所有算子覆盖同一组输入。常见检查包括：

- 非整块尺寸和尾 mask；
- 空输入是否允许；
- dtype 和精度容差；
- NaN、Inf、零、极值和重复索引；
- shape、stride、device 和非法参数；
- 与 PyTorch reference 的数值比较。

不是每个算子都支持非连续输入或所有浮点 dtype。先阅读对应 wrapper 的约束。

benchmark 前应先验证正确性，再完成 warmup，并使用 CUDA event 或 `triton.testing.do_bench`。必须说明计时范围是否包含输出分配、数据复制、Host 同步和元数据检查。优化顺序是：先正确，再测量，然后定位瓶颈，最后修改实现。
