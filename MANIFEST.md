# 001–100 完整算子清单

## 001–011：Triton 官方教程

|编号|算子|编号|算子|
|---:|---|---:|---|
|001|Vector Add|002|Fused Softmax|
|003|Matrix Multiplication|004|Low-Memory Dropout|
|005|LayerNorm Forward/Backward|006|Fused Attention Forward/Backward|
|007|External Libdevice Asin|008|Grouped GEMM|
|009|Persistent/TMA Matmul|010|Block-Scaled Matmul|
|011|Programmatic Dependent Launch|||

## 012–033：逐元素、激活与广播

|编号|算子|编号|算子|编号|算子|
|---:|---|---:|---|---:|---|
|012|Subtract|013|Multiply|014|Divide|
|015|Scalar Add|016|AXPBY|017|ReLU|
|018|LeakyReLU|019|Sigmoid|020|Tanh|
|021|GELU|022|SiLU|023|Softplus|
|024|ELU|025|HardSigmoid|026|HardSwish|
|027|Square|028|Square Root|029|Exponential|
|030|Logarithm|031|Clamp|032|Where|
|033|Row Bias Add|||||

## 034–046：归约与归一化

|编号|算子|编号|算子|编号|算子|
|---:|---|---:|---|---:|---|
|034|Row Sum|035|Row Mean|036|Row Maximum|
|037|Row Minimum|038|Row L1 Norm|039|Row L2 Norm|
|040|Row Variance|041|Row Argmax|042|Row LogSoftmax|
|043|RMSNorm|044|BatchNorm Inference|045|GroupNorm|
|046|Inclusive Cumsum|||||

## 047–056：索引与稀疏

|编号|算子|编号|算子|编号|算子|
|---:|---|---:|---|---:|---|
|047|Embedding Lookup|048|Gather Rows|049|Scatter Rows|
|050|Scatter-Add Rows|051|Index Select Rows|052|One-Hot|
|053|Row Top-K|054|CSR SpMV|055|COO Scatter-Add|
|056|Embedding Bag Sum|||||

## 057–060：线性代数

|编号|算子|编号|算子|编号|算子|编号|算子|
|---:|---|---:|---|---:|---|---:|---|
|057|Batched Matmul|058|GEMV|059|Outer Product|060|Linear + Bias|

普通 Matmul 已是官方 003，不重复占号。

## 061–068：视觉

|编号|算子|编号|算子|编号|算子|编号|算子|
|---:|---|---:|---|---:|---|---:|---|
|061|Conv1D|062|Conv2D|063|Depthwise Conv2D|064|Pointwise Conv2D|
|065|MaxPool2D|066|AvgPool2D|067|Nearest Resize|068|Bilinear Resize|

## 069–081：Transformer 推理

|编号|算子|编号|算子|编号|算子|
|---:|---|---:|---|---:|---|
|069|Causal Attention|070|RoPE|071|ALiBi Bias|
|072|KV Cache Append|073|Paged Attention|074|Residual LayerNorm|
|075|Residual RMSNorm|076|SwiGLU|077|GEGLU|
|078|MoE Expert Count|079|Temperature Scaling|080|Greedy Decode|
|081|Repetition Penalty|||||

## 082–088：量化

|编号|算子|编号|算子|编号|算子|
|---:|---|---:|---|---:|---|
|082|Per-Tensor Quantize|083|Per-Tensor Dequantize|084|Per-Row Quantize|
|085|Per-Channel Quantize|086|INT8 Matmul|087|Fake Quantize|
|088|Blockwise Quantize|||||

## 089–100：损失、相似度与优化器

|编号|算子|编号|算子|编号|算子|
|---:|---|---:|---|---:|---|
|089|Row MSE|090|BCE With Logits|091|Cross Entropy|
|092|Cosine Similarity|093|KL Divergence|094|SGD|
|095|Momentum SGD|096|Adam|097|AdamW|
|098|Adagrad|099|Apply Gradient Norm Clip Scale|100|Non-Finite Check|

`Row LogSoftmax=042`、`Cross Entropy=091`、`KL Divergence=093` 与常用 reduction 全部保留。099 接收已经计算好的 norm，只执行原地缩放；它不包含 norm reduction。编号变化来自按领域重新排序。
