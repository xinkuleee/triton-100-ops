# 04 索引与稀疏：047–056

本章的核心问题是：索引如何改变 GPU 的读写地址，以及不规则地址如何影响合并访存、缓存、原子操作和负载均衡。本章用十个教学算子回答这个问题。

## 共同概念

### 软件与硬件如何配合

CPU 是 Host，负责检查输入、分配输出并启动工作。GPU 是 Device，负责并行执行 **GPU kernel**。Python 包装函数启动 Triton kernel；CUDA launcher 启动 CUDA kernel。Triton 使用 JIT（Just-In-Time，即首次调用时即时编译）把 Python 中的 kernel 描述编译为 GPU 代码。CUDA 文件包含 FP32 kernel 与轻量 launcher，通常由 CUDA 编译器 `nvcc` 编译。CUDA launcher 接收裸指针，不知道 tensor 的实际分配长度。

### 数据、索引与内存

**Tensor** 是有 shape（各维长度）、dtype（元素类型）和存储的多维数据。FP16 是 16 位 IEEE 半精度浮点格式，BF16 是保留较宽指数范围的 16 位浮点格式，FP32 是 32 位 IEEE 单精度浮点格式。int32、int64 和 `int64_t` 分别表示 32 位或 64 位有符号整数。`x[R,C]` 表示 x 是 R 行、C 列的二维 tensor；方括号在文档中描述 shape，不是一次源码索引操作。

**Index（索引）**是选择 tensor 元素的整数；索引决定读写地址。连续索引让相邻 CUDA thread 访问相邻地址，GPU 可以把请求合并为较少的显存事务，这称为**合并访存**。随机索引通常需要更多事务。GPU cache 保存近期读取的数据；重复索引可能提高缓存命中率。

### 并行执行单位

**CUDA thread** 执行标量工作；可同步并共享片上 shared memory 的一组 CUDA thread 称为 **CUDA thread block（CTA）**；一次 CUDA kernel launch 的全部 CUDA thread block（CTA）称为 **CUDA grid**。**Triton program instance** 是 Triton launch grid 中的独立逻辑实例；**Triton block tensor** 是它同时表达的一组逻辑值。`mask` 只让有效逻辑元素访存，不提供同步。

**warp** 是 NVIDIA GPU 上通常共同执行指令的 32 个 CUDA thread；warp lane 是其中一个 CUDA thread 的物理位置。Shared memory 是同一 CUDA thread block（CTA）可共享的片上存储。Barrier（屏障）让一个 CUDA thread block（CTA）中的 CUDA thread 等待到指定同步点。Triton 源码中的变量名 `lane` 仅表示 Triton block tensor 的逻辑偏移，不表示物理 warp lane。

### 索引操作与稀疏格式

**Gather** 按索引从多个源位置读取。**Scatter** 按索引向多个目标位置写入。多个更新写同一目标时会发生数据竞争；atomic operation（原子操作）把一次更新变成不可分割的操作，但浮点原子加法的顺序仍不固定。**Alias（别名）**表示两个 tensor 共享存储。Reduction（归约）把多个值合并为一个值，例如求和或求最大值。

下面的字符图强调 Gather 与 Scatter 的地址方向。图规模很小，字符图比流程图更直接。

```text
Gather:  source[index[p]] --read-->  out[p]
Scatter: src[p]           --write-> out[index[p]]
```

**CSR（Compressed Sparse Row，压缩稀疏行）**用 `row_ptr[r]:row_ptr[r+1]` 这个左闭右开的半开区间定位第 r 行的非零元素；半开区间包含起点、不包含终点。**COO（Coordinate，坐标格式）**为每个非零元素直接保存行列坐标。**nnz** 是 non-zero entries（已存储非零项）的数量。Embedding Bag 的 `offsets` 也用相邻边界表示每个 bag 的半开区间。

复杂度记号 $O(f(n))$ 描述输入增长时工作量的增长量级，不表示精确运行时间。例如，$O(kC)$ 表示工作量与 k 和 C 的乘积成比例。`INT_MAX` 是 int32 能表示的最大正整数；超过范围的地址计算可能溢出并指向错误位置。测试检查结果是否满足预期，benchmark 测量运行时间或吞吐量，profiler（性能分析工具）记录更细的硬件执行指标。

[python/_common.py](python/_common.py) 的真实 helper 是 `require_device_tensor`、`require_float_tensor`、`require_same_device` 和 `row_launch_meta`。前三个 helper 检查输入；`row_launch_meta` 选择 Triton block tensor 的宽度和使用的 warp 数。`MAX_ROW` 只限制单行向量化宽度，`MAX_BAG` 只限制单个 bag 的循环次数。CUDA 复用 [shared/cuda_common.cuh](../../shared/cuda_common.cuh) 的 `kThreads`、`ceil_div_int` 和 `block_sum`。

## 047 — Embedding Lookup

### 概念与用途

Embedding Lookup（嵌入查找）把整数 ID 映射为固定宽度的向量。语言模型用它把 token ID 转成向量，推荐系统用它读取用户或物品特征。

### 输入与输出

输入 `weight[V,D]` 包含 V 个 ID 的 D 维向量，`indices[I]` 包含 I 个待查询 ID。输出 `out[I,D]`。非法 ID 产生零行。ID（identifier）是标识一个对象的整数。

### 算法

对每个输出位置 `(item, col)`，先读取 `indices[item]`。ID 位于 `[0, V)` 时复制 `weight[ID, col]`；否则写 0。

### Triton 实现

[python/047_embedding.py](python/047_embedding.py) 为每个 ID 启动一个 Triton program instance，用 ID mask 保护读取、列 mask 处理尾部。支持 FP16、BF16、FP32 权重和 int64 ID；`row_launch_meta(D)` 要求 `1 <= D <= 65536`。

```python
index = tl.load(indices_ptr + item).to(tl.int64)
valid_index = (index >= 0) & (index < vocab)
values = tl.load(weight_ptr + index * width + col,
                 mask=valid_index & (col < width), other=0.0)
tl.store(out_ptr + item * width + col, values, mask=col < width)
```

`index` 是一次标量间接索引；`valid_index` 保护源行，`col < width` 同时保护源列与输出尾部。

包装函数先验证 shape、dtype、连续性和设备，再分配输出。`tl.program_id` 返回的 Triton program instance ID 选择输出行，Triton block tensor 中的逻辑元素选择列。合法 ID 分支读取 weight，非法分支使用零；最后只有列 mask 内的逻辑元素写输出。

### CUDA 实现

[cuda/047_embedding.cu](cuda/047_embedding.cu) 为每个 ID 启动一个 CUDA thread block（CTA）。接口固定为 FP32/int64_t。launcher 未验证 `count*width` 的地址范围，极大尺寸可能发生 32 位地址计算溢出。

```cuda
const int64_t index = indices[blockIdx.x];
for (int col = threadIdx.x; col < width; col += blockDim.x)
  out[blockIdx.x * width + col] =
      index >= 0 && index < vocab ? weight[index * width + col] : 0.0f;
```

`blockIdx.x` 选择输出行；`threadIdx.x` 让 CUDA thread block（CTA）中的 CUDA thread 跨步复制列。

### GPU 硬件映射

同一 Triton program instance 或 CUDA thread block（CTA）处理一行。同一行的列访问连续，适合合并访存；不同实例的源行由 ID 决定，缓存命中取决于 ID 是否重复。

### 教程

比较合法、负数和越上界 ID。零行是本项目契约，不是 PyTorch embedding 的默认越界行为。练习：构造重复 ID，观察输出正确性不变，再用 profiler 比较重复行与随机行的缓存行为。

### 验证与限制

测试覆盖合法、负数和越上界三类 ID；benchmark 只用合法 ID。CUDA 未自动编译或运行。

## 048 — Gather Rows

### 概念与用途

Gather（聚集）按索引从源 tensor 读取数据。Gather Rows 为每个输入行选择一组列，适合读取稀疏特征或重排部分元素。

### 输入与输出

输入 `x[R,C]` 是 R 行 C 列的矩阵，`indices[R,I]` 为每行提供 I 个列号。输出 `out[R,I]`。非法列写零。源码中的 `count` 等于 I。

### 算法

把输出 `[R,I]` 展平成 `R*I` 个位置。对线性位置 `offset`，用 `offset // I` 得到行号，再读取 `indices[offset]` 得到源列号。合法列读取 `x[row, index]`，非法列写 0。

### Triton 实现

[python/048_gather_rows.py](python/048_gather_rows.py) 让 Triton block tensor 中的每个逻辑元素写一个输出。包装函数没有为 kernel 的 32 位 `tl.program_id` 返回的 Triton program instance ID 和地址表达式设置总规模上限。

```python
offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
row = offset // count
index = tl.load(indices_ptr + offset, mask=offset < rows * count, other=0)
value = tl.load(x_ptr + row * cols + index, mask=valid, other=0.0)
```

`offset` 同时定位索引和输出；整数除法恢复行，间接 `index` 选择列。

包装函数确认 indices 第一维等于 R。kernel 将二维输出展平成一维：先得到 `offset`，再计算 `row` 和 `index`，最后组成源地址。

### CUDA 实现

[cuda/048_gather_rows.cu](cuda/048_gather_rows.cu) 只证明 `rows*count` 能放入 int32，不能证明 `row*cols+index` 安全。较大的 cols 配合较小的 count 仍可能溢出。

```cuda
const int offset = blockIdx.x * blockDim.x + threadIdx.x;
if (offset >= total) return;
const int row = offset / count;
const int64_t index = indices[offset];
out[offset] = index >= 0 && index < cols ? x[row * cols + index] : 0.0f;
```

边界分支保护线性输出；三元表达式只在合法列读取 `x`。

### GPU 硬件映射

连续逻辑元素或 CUDA thread 连续读取 indices 和写 out，因此索引流和输出通常可以合并访存。源 x 地址可能分散，数据读取能否合并取决于每行的索引模式。

### 教程

分别推导输出元素数 rows*count 和源地址 row*cols+index；两者需要不同的上界检查。练习：分别使用递增、逆序和随机列索引，解释为何计算量相同而显存事务不同。

### 验证与限制

测试覆盖正常和非法索引；benchmark 只用合法索引。CUDA 未运行。

## 049 — Scatter Rows

### 概念与用途

Scatter（分散）按索引把源值写到目标位置。Scatter Rows 用普通写入更新矩阵中的指定列，适合目的位置互不重复的覆盖操作。

### 输入与输出

输入 `src[R,I]` 提供更新值，`indices[R,I]` 提供目的列号，`out[R,C]` 是被原地修改的目标。普通 store（写入）用 `src` 覆盖 `out` 的指定列。非法列被忽略。

### 算法

对每个 `src[row, item]`，读取对应列索引。索引合法时写入 `out[row, index]`，否则跳过。算法不合并重复目的位置。

### Triton 实现

[python/049_scatter_rows.py](python/049_scatter_rows.py) 的 mask 只能防越界，不能规定重复目的位置的写入顺序。多个逻辑元素写同一位置时，最终保留哪个值未指定。包装函数也不检查 src/out 别名；写入可能改变尚未读取的源值。

```python
index = tl.load(indices_ptr + offset, mask=mask, other=0).to(tl.int64)
value = tl.load(src_ptr + offset, mask=mask)
tl.store(out_ptr + row * cols + index, value,
         mask=mask & (index >= 0) & (index < cols))
```

普通 `tl.store` 没有冲突仲裁；相同 `row,index` 会有多个写入者。

包装函数不创建新 tensor，而是返回被修改的 out。kernel 连续读取 src 和 indices，再按索引分散写入。mask 只控制逻辑元素是否执行 store，不能建立写入的先后顺序。

### CUDA 实现

[cuda/049_scatter_rows.cu](cuda/049_scatter_rows.cu) 同样存在重复写数据竞争。launcher 不检查 `row*cols+index` 或别名。

```cuda
const int row = offset / count;
const int64_t index = indices[offset];
if (index >= 0 && index < cols)
  out[row * cols + index] = src[offset];
```

条件只拒绝非法列，不同步写相同地址的 CUDA thread。

### GPU 硬件映射

连续逻辑元素或 CUDA thread 可合并读取 src 和 indices，但目标地址由索引决定，写入可能分散。不同逻辑元素之间没有同步；GPU 调度也不保证哪个 Triton program instance、CUDA thread block（CTA）或 warp 最后写入。

### 教程

普通 store 只适合每个目的位置一个写入者。需要累加时使用 050；需要确定覆盖规则时，应拒绝重复索引，或先排序再明确选择要保留的值。练习：生成重复列并重复运行，说明即使某台 GPU 上结果暂时稳定，也不能据此建立契约。

### 验证与限制

测试和 benchmark 都没有把重复索引传给 049，也没有覆盖别名。CUDA 未运行。

## 050 — Scatter-Add Rows

### 概念与用途

Scatter-Add（分散相加）把多个更新累加到索引指定的位置。它适合梯度聚合、直方图式更新，以及任何允许目的位置重复的累加。

### 输入与输出

输入 `src[R,I]` 提供更新值，`indices[R,I]` 提供目的列号，`out[R,C]` 是被原地修改的目标。每个 `src` 值累加到对应 `out` 位置。非法列被忽略，重复目的位置合法。

### 算法

对每个源值计算目的地址，并执行原子加法。原子加保证更新不丢失，但浮点加法顺序仍可能变化。

### Triton 实现

[python/050_scatter_add_rows.py](python/050_scatter_add_rows.py) 使用 `tl.atomic_add`，只接受 FP32 out/src。它不检查别名。

```python
value = tl.load(src_ptr + offset, mask=mask, other=0.0)
tl.atomic_add(out_ptr + row * cols + index, value,
              mask=mask & (index >= 0) & (index < cols))
```

`tl.atomic_add` 把目的地址的读、加、写作为原子更新，但不解决源张量别名。

地址生成与 049 相同，唯一关键差异是 store 变为原子“读—修改—写”。包装函数限制 FP32，是因为教学路径只承诺该原子 dtype。

### CUDA 实现

[cuda/050_scatter_add_rows.cu](cuda/050_scatter_add_rows.cu) 使用 `atomicAdd`。`rows*count` 检查仍未覆盖 `row*cols+index` 的地址范围。

```cuda
const int64_t index = indices[offset];
if (index >= 0 && index < cols)
  atomicAdd(out + row * cols + index, src[offset]);
```

`atomicAdd` 保留重复更新；同地址更新由硬件原子路径串行提交。

### GPU 硬件映射

原子单元为同一地址串行提交更新；不同地址仍可并行。目的地址越集中，串行化越严重；目的地址越分散，可并行提交的原子更新越多。

### 教程

给两个更新相同索引：049 的覆盖值未指定，050 保留两次加法。热点越集中，原子串行化越严重。练习：固定更新总数，逐步减少唯一目的位置数，测量冲突率与耗时。

### 验证与限制

测试覆盖重复位置和非法索引，不覆盖别名。CUDA 未运行。

## 051 — Index Select Rows

### 概念与用途

Index Select Rows（按索引选择行）从矩阵中复制若干完整行。它常用于重排批次、选择样本或根据路由结果收集特征。

### 输入与输出

输入 `x[R,C]` 是 R 行 C 列的矩阵，`indices[I]` 包含 I 个源行号。输出 `out[I,C]`。重复源行合法，非法行号写零。

### 算法

对每个输出行读取一个源行号。源行号位于 `[0, R)` 时复制全部 C 列；否则把该输出行写成 0。不同输出行互不重叠。

### Triton 实现

[python/051_index_select_rows.py](python/051_index_select_rows.py) 每个 Triton program instance 复制一行；`row_launch_meta(C)` 限制 C 为 1 到 65536。

```python
source_row = tl.load(indices_ptr + selected_row).to(tl.int64)
valid_row = (source_row >= 0) & (source_row < rows)
value = tl.load(x_ptr + source_row * cols + col,
                mask=valid_row & (col < cols), other=0.0)
```

`selected_row` 是 Triton program instance 拥有的输出行；`source_row` 是间接读取的源行。

包装函数分配 I x C 输出。`tl.program_id` 返回的 Triton program instance ID 选择 indices 中的一个条目，Triton block tensor 中的每个逻辑元素复制该源行的一列。

### CUDA 实现

[cuda/051_index_select_rows.cu](cuda/051_index_select_rows.cu) 中 `source_row` 是 `int64_t`，所以源偏移 `source_row * cols` 按 64 位计算。输出偏移 `blockIdx.x * cols` 仍按 32 位计算；launcher 未验证这个乘积或实际分配边界。

```cuda
const int64_t source_row = indices[blockIdx.x];
for (int col = threadIdx.x; col < cols; col += blockDim.x)
  out[blockIdx.x * cols + col] = source_row >= 0 && source_row < rows
      ? x[source_row * cols + col] : 0.0f;
```

一个 CUDA thread block（CTA）拥有一个输出行，其中的 CUDA thread 协作复制连续列。

### GPU 硬件映射

一个 Triton program instance 或 CUDA thread block（CTA）负责一个输出行。单行内部读写连续；重复行只产生重复读取，不产生输出竞争，因为每个实例拥有独立输出行。

### 教程

047 与 051 使用相同映射；区别只是源行代表词表项还是一般矩阵行。练习：把 047 的 weight/x 和 vocab/rows 改名映射到 051，列出不变的地址公式。

### 验证与限制

测试覆盖重复、合法和非法行号。CUDA 未运行。

## 052 — One-Hot

### 概念与用途

One-Hot（独热编码）把一个类别 ID 表示为只有对应类别位置为 1 的向量。它适合教学、生成小类别空间的显式标签，以及某些损失函数的输入准备。

### 输入与输出

输入 `indices[count]` 包含 count 个 ID，`classes` 是类别总数。输出 `out[count,classes]` 是 FP32 矩阵；匹配列为 1，其余为 0，非法 ID 产生零行。

### 算法

把输出展平成一维。对每个线性位置，用商得到输入项，用余数得到类别列；类别列等于该项 ID 时写 1，否则写 0。

### Triton 实现

[python/052_one_hot.py](python/052_one_hot.py) 写全部 `count*classes` 元素，但包装函数没有显式限制 kernel 的 32 位线性索引范围。

```python
item = offset // classes
cls = offset % classes
index = tl.load(indices_ptr + item, mask=mask, other=-1)
tl.store(out_ptr + offset, tl.where(cls == index, 1.0, 0.0), mask=mask)
```

商选择输入项，余数选择类别列；`tl.where` 为每个元素生成确定的 0 或 1。

kernel 从 `offset` 计算 `item` 和 `cls`，然后比较 `indices[item]` 与 `cls`。它不需要预先清零，因为每个输出元素都会被一个逻辑元素写成 0 或 1。

### CUDA 实现

[cuda/052_one_hot.cu](cuda/052_one_hot.cu) 检查 `count*classes <= INT_MAX`。每个 CUDA thread 写唯一元素，无需原子操作。

```cuda
const int item = offset / classes;
const int cls = offset % classes;
out[offset] = indices[item] == cls ? 1.0f : 0.0f;
```

每个 CUDA thread 拥有唯一 `offset`，所以没有重复写。

### GPU 硬件映射

连续逻辑元素或 CUDA thread 写连续输出地址，因此写入可以合并访存。为同一个输入项生成类别行时，多个逻辑元素读取同一个 ID，可利用缓存；输出流量随 classes 增长。

### 教程

One-hot 存储随类别数线性增长。若结果只用于查表，直接 gather 可避免物化。练习：计算 count=1M、classes=50K 时 FP32 输出所需字节数，并判断是否应物化。

### 验证与限制

测试覆盖合法和非法 ID；benchmark 只用合法 ID。CUDA 未运行。

## 053 — Row Top-K

### 概念与用途

Top-K 从一组值中选出最大的 k 项。逐行 Top-K 常用于分类候选、束搜索、稀疏注意力和推荐系统召回。

### 输入与输出

输入 `x[R,C]` 是 R 行 C 列的矩阵，k 是每行需要选择的元素数。输出两个 `[R,k]` tensor，分别保存值和 int64 列号。相等值选较小列号；输入必须是有限数，即不含 NaN 或正负无穷，且 `1 <= k <= min(C,64)`。

### 算法

每一轮扫描当前行，找出最大值及其列号，保存结果，并把已选位置替换为负无穷。重复 k 轮即可得到降序结果。

### Triton 实现

[python/053_row_topk.py](python/053_row_topk.py) 每轮用 `tl.argmax`/`tl.max` 扫描整行。工作量为 $O(kC)$，归约树每轮还有 $O(\log C)$ 并行深度。有限数检查中的 `item()` 会等待 GPU 结果并让 CPU 与 GPU 同步。

```python
for rank in tl.static_range(0, K):
    index = tl.argmax(working, axis=0, tie_break_left=True)
    value = tl.max(working, axis=0)
    tl.store(indices_ptr + row * K + rank, index)
    working = tl.where(col == index, -float("inf"), working)
```

`tl.static_range` 展开固定 K 轮；相等值选择较小列号的规则与屏蔽已选列共同定义输出顺序。

Triton 先把尾列填成负无穷，再归约最大值及索引，并把选中的逻辑元素改为负无穷。

### CUDA 实现

[cuda/053_row_topk.cu](cuda/053_row_topk.cu) 每个候选还检查此前选中的列表。所有 CUDA thread 合计的总工作量为 $O(k^2C)$；若一个 CUDA thread block（CTA）含 $T$ 个 CUDA thread，则每个 CUDA thread 的扫描工作约为 $O(kC/T+k^2C/T)$。这与 Triton 的 $O(kC)$ 总工作量不同。`selected[64]` 决定 k 上限；CUDA 不检查有限值。

```cuda
for (int rank = 0; rank < k; ++rank) {
  Op053ValueIndex local{-CUDART_INF_F, INT_MAX};
  candidates[threadIdx.x] = local;
  __syncthreads();
  if (threadIdx.x == 0) selected[rank] = candidates[0].index;
  __syncthreads();
}
```

`candidates` 和 `selected` 位于 shared memory；两次 barrier 分别保护归约输入和下一轮已选索引。

CUDA 中，每个 CUDA thread 先求局部候选，把 value/index 对写入 shared memory，再用树形归约合并。每轮 barrier 保证所有相关 CUDA thread 都能看到已写入 shared memory 的数据。

### GPU 硬件映射

一个 Triton program instance 或 CUDA thread block（CTA）处理一行。行内归约利用并行执行资源，但 k 轮之间存在数据依赖，必须串行执行；因此 k 增大时，并行度不随之增加。

### 教程

跟踪 k=2 的两轮选择。生产实现通常用排序或专用 selection 减少重复扫描。练习：分别推导 Triton 与 CUDA 在 C=1024、T=256、k=8 时的候选比较数量级。

### 验证与限制

测试覆盖随机值和相等值。benchmark 通用比较器对整数索引也用浮点容差，可能漏掉相近但错误的大索引。CUDA 未运行。

## 054 — CSR SpMV

### 概念与用途

CSR 只存储非零项，适合大量元素为零的矩阵。SpMV（Sparse Matrix-Vector Multiplication，稀疏矩阵向量乘法）常用于图算法、科学计算和稀疏线性模型。

### 输入与输出

CSR 用 `row_ptr[R+1]` 标记每行在 `col_idx[nnz]` 和 `values[nnz]` 中的半开区间。输入还包含长度为 C 的稠密 `vector`，输出是长度为 R 的 `y`。非法列被忽略，空行输出 0。

### 算法

每行计算：

$$
y_r=\sum_{p=\mathrm{row\_ptr}_r}^{\mathrm{row\_ptr}_{r+1}-1} values_p x_{col_p}.
$$

一个 Triton program instance 或 CUDA thread block（CTA）遍历一行存储的非零项，把 `values[p] * vector[col_idx[p]]` 归约为一个输出值。

### Triton 实现

[python/054_csr_spmv.py](python/054_csr_spmv.py) 把 int32 row_ptr 复制到 CPU，验证首项为 0、末项等于 nnz、单调不减且最大行长不超过 max_nnz_per_row。kernel 仍以 int32 计算 `position=start+lane`，其中源码变量 `lane` 是逻辑偏移。结构元数据接近 `INT_MAX` 时可能回绕；`position<end` 不能在回绕后保证安全。

```python
start = tl.load(row_ptr + row)
end = tl.load(row_ptr + row + 1)
position = start + lane
active = position < end
col = tl.load(col_idx + position, mask=active, other=0)
value = tl.load(values + position, mask=active, other=0.0)
```

`start/end` 来自结构元数据。`active` 只比较终点；它没有检查 `position >= 0` 或 `position < nnz`，也无法识别 int32 回绕。

包装函数的 CPU 校验发生在 launch 前，会引入同步。kernel 读取 `start/end`，让 Triton block tensor 中的逻辑元素按行内位置读取 col_idx、values 和 vector，最后归约。

### CUDA 实现

[cuda/054_csr_spmv.cu](cuda/054_csr_spmv.cu) 不接收 nnz，也不验证 row_ptr。负起点、倒序区间或超尾终点可越界读取。当前接口无法仅靠参数证明 CSR 安全。

```cuda
for (int position = row_ptr[row] + threadIdx.x;
     position < row_ptr[row + 1]; position += blockDim.x) {
  const int col = col_idx[position];
  if (col >= 0 && col < vector_size)
    partial += values[position] * vector[col];
}
```

循环完全信任 `row_ptr`。列范围检查只保护 `vector[col]`，不能保护读取 `col_idx[position]` 和 `values[position]`。

### GPU 硬件映射

values 和 col_idx 在一行内连续，因此这些读取较容易合并；vector 是间接读取，局部性取决于列号。行长差异让不同 Triton program instance 或 CUDA thread block（CTA）的工作量不同，长行与空行并存时会产生负载不均。

### 教程

row_ptr=[0,2,2,5] 表示三行，第二行为空。末项改为 6 时 Python 会拒绝；CUDA launcher 不知道真实 nnz。练习：补出安全 CUDA API 所需的 nnz 参数与四项结构检查。

### 验证与限制

测试只覆盖正常结构、空行、非法列和一例非单调 row_ptr。未覆盖错误首尾、超长行或 INT_MAX 边界。benchmark 包含 GPU 到 CPU 同步，不是纯 kernel 时间。CUDA 未运行。

## 055 — COO Scatter-Add

### 概念与用途

COO 用坐标描述每个非零项或更新。这里的一维 COO Scatter-Add 把 `(index, value)` 更新流累加为稠密向量，适合直方图、稀疏梯度和图上的聚合。

### 输入与输出

输入 `indices[nnz]` 和 `values[nnz]` 表示 nnz 个更新，`size` 指定稠密输出长度。Python 输出新的 `out[size]`，先清零再原子累加；非法索引被忽略。

### 算法

先把输出初始化为 0。然后读取每一对 index/value；index 位于 `[0, size)` 时，把 value 原子累加到 `out[index]`。

### Triton 实现

[python/055_coo_scatter_add.py](python/055_coo_scatter_add.py) 固定 int32 索引和 FP32 值，分配零输出并使用 `tl.atomic_add`。

```python
index = tl.load(indices_ptr + offset, mask=mask, other=0)
value = tl.load(values_ptr + offset, mask=mask, other=0.0)
tl.atomic_add(out_ptr + index, value,
              mask=mask & (index >= 0) & (index < size))
```

线性 COO 位置读取一对 index/value；目的范围 mask 保护原子地址。

包装函数先用 `torch.zeros` 建立输出初值为 0 的语义。kernel 连续读取 COO 流，再按 index 原子写稠密输出。

### CUDA 实现

[cuda/055_coo_scatter_add.cu](cuda/055_coo_scatter_add.cu) 只增量更新调用者提供的 out，不负责清零。因此两种公开层次的初始状态语义不同。

```cuda
const int position = blockIdx.x * blockDim.x + threadIdx.x;
if (position >= nnz) return;
const int index = indices[position];
if (index >= 0 && index < size)
  atomicAdd(out + index, values[position]);
```

CUDA kernel 没有初始化阶段；调用前 `out` 中的值会保留并参与累加。

### GPU 硬件映射

连续逻辑元素或 CUDA thread 可以合并读取 COO 流。索引均匀时原子操作可分散；大量相同索引会集中到一个缓存行和原子地址，导致串行化。

### 教程

索引 `[2,2]` 的两项都累加到 `out[2]`。调用 CUDA 前必须明确输出初值。练习：令 CUDA out 初值为 10，写出它与 Python 从 0 开始累加所得输出的关系。

### 验证与限制

测试覆盖重复、非法索引和空输出。热点降低吞吐。CUDA 未运行。

## 056 — Embedding Bag Sum

### 概念与用途

Embedding Bag 把数量不同的 ID 分组，并把每组对应的 embedding 合并成一个向量。求和版本常用于推荐系统中的变长类别特征。

### 输入与输出

输入 `weight[V,D]` 是 embedding 表，`indices[N]` 是 N 个 ID，`offsets[B+1]` 把 ID 流切成 B 个半开区间。输出 `out[B,D]` 为 FP32。每个 bag 直接求 embedding 行之和，不物化中间行；非法 ID 被忽略。

### 算法

对 bag `b`，读取 `start=offsets[b]` 和 `end=offsets[b+1]`。遍历 `[start,end)` 中的 ID，把每个合法 ID 对应的 embedding 行逐列累加到 `out[b]`。

### Triton 实现

[python/056_embedding_bag_sum.py](python/056_embedding_bag_sum.py) 验证 offsets 首项为 0、末项等于 ID 数、单调不减且 bag 长度不超过 max_bag_size。`MAX_BAG=4096` 只限制单 bag 循环；`row_launch_meta(D)` 另行限制列宽。kernel 的 int32 `position=start+step` 接近 `INT_MAX` 时可能回绕。

```python
start = tl.load(offsets_ptr + bag)
end = tl.load(offsets_ptr + bag + 1)
for step in range(0, MAX_BAG_SIZE):
    position = start + step
    active = position < end
    index = tl.load(indices_ptr + position, mask=active, other=0)
```

`MAX_BAG_SIZE` 是编译期循环界。`active` 表示 bag 的半开区间，但没有独立的 ID 数量参数来保护回绕后的 `position`。

包装函数先把 offsets 复制到 CPU 验证，再为每个 bag 启动一个 Triton program instance。Triton block tensor 中的一个逻辑元素固定负责一个 embedding 列，并循环读取该 bag 的 ID 与权重。

### CUDA 实现

[cuda/056_embedding_bag_sum.cu](cuda/056_embedding_bag_sum.cu) 不接收 ID 数，也不验证 offsets。负 offset、倒序区间或超尾末项可能越界。CUDA launcher 没有 max_bag_size 上限。

```cuda
for (int position = offsets[bag]; position < offsets[bag + 1]; ++position) {
  const int index = indices[position];
  if (index >= 0 && index < vocab)
    sum += weight[index * width + col];
}
```

循环边界完全来自 offsets。ID 范围判断保护 weight 行，却不能保护 `indices[position]` 本身。

### GPU 硬件映射

相同 step 的逻辑元素读取同一 ID，随后访问连续的权重列，适合合并访存。不同 bag 的长度差异导致 Triton program instance 或 CUDA thread block（CTA）的运行时间不同，因而产生负载不均。

### 教程

offsets=[0,3,4,6] 形成 [0,3)、[3,4)、[4,6)。相邻 offset 相等表示空 bag。结构元数据决定所有间接读取范围，必须在 launch 前验证。练习：构造空 bag、单元素 bag 和最长 bag，逐项写出 start/end/active。

### 验证与限制

测试覆盖正常 bag、空 bag 和非法 ID，不覆盖损坏 offsets、超长 bag 或 int32 边界。benchmark 包含同步，不是纯 kernel 时间。CUDA 未运行。

## 运行验证

```shell
python3 -m pytest -q tests/test_04_indexing_sparse.py
python3 benchmarks/run.py --op 054 --size medium --check
```

这些命令只运行 Triton/Python 路径。静态测试只检查文件布局、符号和直接 kernel launch，不编译 CUDA。当前项目记录的环境没有 nvcc 或可用 GPU，因此 CUDA 编译、Triton JIT、GPU 数值和性能均未验证。
