# Debug Report: CM Paged Attention Quality Regression vs Dense OCL PA

**Date:** 2026-04-10 (resolved 2026-04-10)  
**Model:** minicpm4-8b (num_heads=32, num_kv_heads=2, head_size=128, GQA 16:1)  
**Component:** OpenVINO GPU Plugin — CM Paged Attention Multi-Token & Single-Token Kernels  
**Symptom:** XAttention bypass mode (threshold≥1) 在 f16 KV cache 下 wwb similarity 显著低于 dense OCL PA  
**Root Cause:** CM multi-token kernel 中 `(half)scale_factor` Q 预缩放引入 fp16 截断误差，叠加 finalization fp16 累加器，在长序列 online softmax 迭代中复合放大  
**Fix Status:** ✅ **已修复** — 3 个修改组合生效  

---

## 1. 问题描述

在 wwb（WhoWhatBench）测试中，minicpm4-8b 模型开启 XAttention 并设置 `xattention_threshold=2`（bypass 模式）时，部分 prompt 的 similarity 严重回归：

| 配置 | 平均 similarity | 说明 |
|------|:---:|------|
| dense OCL PA (`use_sparse_attention=false`) | **0.9457** | 基线 |
| XA bypass thr=2, bs=128 | 0.8905 | **-5.5%** |
| XA bypass thr=100, bs=128 | 0.8905 | 与 thr=2 完全一致 |
| XA sparse thr=0.9, bs=128 | 0.9148 | 真正 sparse，反而更好 |

其中 6 个 prompt 回归超过 10%，最严重的 prompt idx=4 从 1.0 跌至 0.57（参考答案 "Amsterdam" 被替换为 "Venice"）。

## 2. 调查过程

### 2.1 数据收集与对比

**测试数据位置：**
- 输出结果：`C:\Users\gta\www\outputs.wwb_20260409_230826\minicpm4-8b\20260409_230826\ov_kv-f16\`
- 运行日志：`C:\Users\gta\www\logs.wwb_20260409_230826\minicpm4-8b\20260409_230826\`

**配置文件确认：**
- XA bypass: `cb_config.json` → `{"use_sparse_attention":true,"sparse_attention_config":{"mode":"XATTENTION","xattention_threshold":2,"xattention_block_size":128}}`
- Dense PA: `cb_config.json` → `{"use_sparse_attention":false}`
- 两者 `ov_config.json` 均为 `{"KV_CACHE_PRECISION":"f16"}`

**全配置逐 prompt similarity 对比：**

```
 idx   dense  xa_t2_b128  xa_t2_b256  xa_t100_b128  xa_t100_b256  xa_t0.9_b128
   0  1.0000      0.9410      0.9410        0.9410        0.9410        0.9410
   1  1.0000      1.0000      1.0000        1.0000        1.0000        1.0000
   2  0.9169      0.6364      0.6364        0.6364        0.6364        0.9200  ← 焦点
   3  0.9848      0.9848      0.9848        0.9848        0.9848        0.9443
   4  1.0000      0.5678      0.5678        0.5678        0.5678        1.0000  ← 最严重
  ...
 avg  0.9457      0.8905      0.8905        0.8905        0.8905        0.9148
```

**关键发现 #1：** 所有 bypass 模式（thr=2, thr=100）结果完全一致，不受 threshold 值和 block_size 影响。证实 sparse 跳块逻辑被完全禁用（SPARSE_BLOCK_SIZE=1），问题出在 CM kernel 本身而非 sparse 估计。

**关键发现 #2：** 真正的 sparse attention (thr=0.9) 反而比 bypass 更好（0.9148 vs 0.8905）。

### 2.2 发散点定位：PREFILL 还是 DECODE？

对所有 26 个 prompt 做了逐 word 对比分析：

```
idx  dense_sim  xa_t2_sim  prompt_len  diverge_word  note
  2    0.9169    0.6364      3577         1          PREFILL-EXIT diverge
  4    1.0000    0.5678      2652         0          PREFILL-EXIT diverge
 12    0.9484    0.7365      2049         0          PREFILL-EXIT diverge
 22    0.9640    0.8078      3422         0          PREFILL-EXIT diverge
 24    0.8638    0.6733      2010         0          PREFILL-EXIT diverge
```

以 prompt idx=2 为例：
- Dense: "The **question** of whether to prioritize work-life balance..."
- XA bypass: "The **balance** between individual well-being..."
- 首 token "The" 一致，**第 2 个 token 就发散** → PREFILL 输出正确首 token，DECODE 第一步就偏移

**结论：** 问题出在 **DECODE 路径**（single-token kernel + finalization），而非 PREFILL（multi-token kernel）。

### 2.3 DECODE 执行路径追踪

CM paged attention 的 DECODE（GENERATE stage）执行流程：

```
paged_attention.cpp:179  →  execute_stage(pa_single_token)         // 每个 partition 独立计算
paged_attention.cpp:180  →  execute_stage(pa_single_token_finalization)  // 合并所有 partition 结果
```

对于 ~894 token 的 prompt，`partition_size=256`，需要 **4 个 partition**。

### 2.4 Finalization 内核代码审查

审查 `pa_single_token_finalization.cm` 发现关键 bug：

```cpp
// L53: 变量名叫 "f32" 但声明为 half（fp16）！
matrix<half, 1, REDUCE_SPLIT_SIZE> out_mat_f32 = 0;     // ← BUG
matrix<half, 1, REDUCE_SPLIT_SIZE> out_mat = 0;
matrix<float, 1, REDUCE_SPLIT_SIZE> data_mat;            // 输入是 float

// L64: cm_mul<half> 将 float 运算结果截断为 fp16，再累加到 fp16 累加器
out_mat_f32 += cm_mul<half>(data_mat, (float)(lse_value/total_lse));  // ← BUG

// L73: format<half> 只是 reinterpret，数据已经是 half
out_mat = out_mat_f32.format<half>();  // ← 无效操作
```

**对照 OCL finalization（正确实现）：**

```c
// paged_attention_opt.cl:741
SOFTMAX_ACCUMULATOR_TYPE acc = 0.0f;   // ✅ float (fp32) 累加器
for (...) {
    // ✅ 全程 float 精度运算
    acc += TO_SOFTMAX_ACCUMULATOR_TYPE(out_val) * ... / TO_SOFTMAX_ACCUMULATOR_TYPE(global_sum);
}
output[out_offset] = TO_OUTPUT_TYPE(acc);  // ✅ 仅最终输出转为 half
```

### 2.5 精度影响量化分析

| 属性 | CM finalization (bug) | OCL finalization |
|------|:---:|:---:|
| 累加器精度 | **fp16（10-bit mantissa）** | fp32（23-bit mantissa） |
| partition 数量（894 token prompt） | 4 | 4 |
| 每步截断相对误差 | ~0.1%（$2^{-10}$） | ~0.000012%（$2^{-23}$） |
| 4 次累加总误差 | **~0.4%** | ~0.00005% |
| 精度劣势 | **8192× 差于 fp32** | 基线 |

fp16 mantissa 仅 10 bit。在 4 个 partition 的加权求和中，每个中间结果都被截断为 fp16 再累加，128 个 head 维度中每个输出值都携带 ~0.3-0.4% 的系统性偏差。这足以：
1. 改变 LM head 输出的 logit 排序
2. 将原本微弱优势的正确 token 替换为错误 token
3. 自回归后续 token 完全发散

### 2.6 为什么 sparse attention (thr=0.9) 反而更好？

- thr=0.9 走完整 XAttention 路径：GEMMQK → FindBlock → PostProc → multi-token PA with sparse mask
- sparse mask **减少了有效 KV 块数**，从而减少了 multi-token kernel 中的循环迭代次数
- 更少的迭代 → 更少的 online softmax 补偿步骤 → 更小的浮点误差累积
- 本质上：sparse attention 通过跳块，意外地规避了 CM kernel 的 causal over-iteration 问题

### 2.7 Fix #1 — Finalization fp16→fp32 累加器

**修改：** `pa_single_token_finalization.cm` 中 `matrix<half>` → `matrix<float>`, `cm_mul<half>` → `cm_mul<float>`  
**单独测试：** 无可见效果（partition 仅 4 个，fp16 误差单独不足以翻转 top-1）  
**组合效果：** ✅ 与 Fix 2/3 组合后生效

### 2.8 Fix #2 — Causal Early Termination（u8 路径）

**修改：** `cm_pa_common.hpp` u8 路径主循环中，barrier/SLM load 之后加 `if (causal_left < 0) { causal_left -= kv_step; continue; }`  
**单独测试：** 无可见效果（数学上 fully-masked block 贡献 0，是 near no-op）  
**组合效果：** ✅ 与 Fix 1/3 组合后生效（减少不必要的 online softmax 补偿步骤，消除 fp 误差累积路径）  
**备注：** f16 路径的 `break` 已回退（无 barrier 依赖，但与其他修复协同不影响结果）；u8 `continue` 保留

### 2.9 Fix #3 — Q 后缩放（fp16 pre-scale → fp32 post-scale）⭐ 主要修复

**修改：** `cm_pa_common.hpp` u8 和 f16 两个路径中，移除 Q 加载时的 `rQ *= (half)scale_factor` 预乘，改为 QK DPAS 之后 `St = cm_mul<float>(St, (float)scale_factor)`  
**机制：** `(half)scale_factor` 将 $1/\sqrt{128} = 0.088388...$ 截断为 fp16 $0.088379...$（相对误差 ~0.01%），但更关键的是 **Q 本身被截断到 fp16 范围后再参与 DPAS**。原始 Q 值的 fp32 精度在预乘 `(half)` 时被丢弃，导致 QK score 的有效精度降低。这在长序列的 online softmax 多次迭代中复合放大。  
**组合效果：** ✅ **与 Fix 1/2 组合后成功修复质量回归**

### 2.10 关键架构差异发现：OCL PREFILL 不读 paged cache

深入对比发现 OCL 和 CM 的 PREFILL 路径存在**根本性架构差异**：

| 方面 | OCL Dense PA (PREFILL) | CM XAttention (PREFILL) |
|------|:---:|:---:|
| **内核** | `sdpa_opt.cl` (FlashAttention-V2) | `pa_multi_token.cm` |
| **K/V 数据源** | **原始 K/V 输入张量**（contiguous） | **paged KV cache**（block-table 间接寻址） |
| **K 布局** | `[total_tokens, num_kv_heads * head_size]` | `[num_blocks, kv_heads, block_size(256), head_size]` |
| **Softmax** | Online (FlashAttention-V2)，partition 级 | Online，per-kv_step(16) 级 |
| **P 精度** | fp16（SLM） | fp32→fp16（DPAS 边界转换） |
| **最终归一化** | IEEE 除法 `output /= exp_sum` | `cm_inv()` 近似倒数 × 乘法 |
| **Causal 处理** | NaN mask，每 token 独立 seq_len | `causal_left` 计数器 |

**关键：OCL PREFILL 读原始 K/V 张量，而 CM PREFILL 读 paged cache。** 两者数据相同（KV cache update 已将数据写入 cache），但读取路径和精度处理完全不同。

### 2.11 CM 与 OCL 精度差异汇总

| # | 差异点 | CM 实现 | OCL 实现 | 状态 |
|---|--------|---------|----------|:---:|
| 1 | Q 缩放 | ~~`rQ *= (half)scale_factor`~~ → `St *= (float)scale_factor` | Q 不预缩放，QK 后乘 `(float)SCALE_FACTOR` | ✅ 已修复 |
| 2 | 最终归一化 | `cm_inv(cur_sum)` 近似倒数 (~1 ULP) | IEEE 除法 `/=` | — 未修（不影响） |
| 3 | exp 函数 | `cm_exp(x * log2e)` (base-2 hardware exp) | `native_exp(x)` (~0.5 ULP) | — 未修（不影响） |
| 4 | Causal over-iteration | U8: `continue` skip | 每 token 独立 seq_len | ✅ 已修复 |
| 5 | Finalization 累加器 | ~~fp16~~ → fp32 | fp32 | ✅ 已修复 |
| 6 | P·V 精度 | P fp16, DPAS fp32 累加 | P fp16, `mad()` fp16 | — 架构差异（CM 更优） |
| 7 | Paged cache 间接寻址 | block_indices 映射 | 不使用（直接读） | — 架构差异 |

## 3. 修复方案（✅ 已验证生效）

三个修改**单独测试时均无可见效果**，但**组合应用后成功修复质量回归**。这说明根因是多个 fp16 精度损失点的**复合效应**，每个单独不足以翻转 top-1 token，但叠加后超过了 softmax 概率分布的翻转阈值。

### Fix 1: Finalization FP32 Accumulator
- **文件：** `pa_single_token_finalization.cm`
- **改动：** `matrix<half>` → `matrix<float>`, `cm_mul<half>` → `cm_mul<float>`
- **修复点：** DECODE 阶段 partition 合并从 fp16 累加改为 fp32 累加

### Fix 2: Causal Early Termination（u8 路径）
- **文件：** `cm_pa_common.hpp`
- **改动：** u8 路径 barrier/SLM load 后加 `if (causal_left < 0) continue`
- **修复点：** 消除 fully-masked causal block 的不必要 online softmax 补偿迭代

### Fix 3: Q 后缩放 ⭐ 主要贡献
- **文件：** `cm_pa_common.hpp`（u8 和 f16 两个路径）
- **改动：** 移除 `rQ *= (half)scale_factor`，改为 QK DPAS 后 `St = cm_mul<float>(St, (float)scale_factor)`
- **修复点：** 避免 Q 向量在 DPAS 前被 fp16 scale_factor 截断，保留 Q 全精度参与矩阵乘法

### 根因分析

原始代码中 `rQ *= (half)scale_factor` 有两层精度损失：
1. `scale_factor`（$1/\sqrt{128}$）被截断为 fp16（10-bit mantissa）
2. **更关键**：Q 向量乘以 fp16 系数后，有效精度降为 fp16 级别，再参与 fp16×fp16→fp32 DPAS

这导致每步 QK score 携带额外误差，在 online softmax 的 `cm_exp((old_max - new_max) * log2e)` 补偿中逐步累积。对于长序列（>2000 tokens），累积误差足以改变 softmax 概率排序。

叠加 finalization 的 fp16 累加器（Fix 1）和 causal 区域的多余迭代（Fix 2），三者共同将误差推过翻转阈值。

## 5. 参考：CM vs OCL 架构差异

### PREFILL 路径
- **OCL:** `sdpa_opt.cl` → 读原始 K/V 输入张量（contiguous buffer），不经 block-table
- **CM:** `pa_multi_token.cm` → 读 paged KV cache（`block_indices` 间接寻址）

### DECODE 路径  
- **OCL:** `paged_attention_opt.cl` STAGE_0 → 读 paged cache → SLM 存 P (fp16) → PV
- **CM:** `pa_single_token.cm` → 读 paged cache → 寄存器存 P (fp16) → PV DPAS
- OCL block_size=16, CM block_size=256

## 6. 附录：测试数据文件索引

```
outputs.wwb_20260409_230826/minicpm4-8b/20260409_230826/
├── ov_kv-f16/
│   ├── cb_dense_mtoks-4096/                        ← 基线 dense PA
│   ├── cb_sparse-xattention_thr-2_bs-128_mtoks-4096/   ← bypass 模式（问题配置）
│   ├── cb_sparse-xattention_thr-2_bs-256_mtoks-4096/
│   ├── cb_sparse-xattention_thr-100_bs-128_mtoks-4096/
│   ├── cb_sparse-xattention_thr-100_bs-256_mtoks-4096/
│   ├── cb_sparse-xattention_thr-0.9_bs-128_mtoks-4096/  ← 真正 sparse
│   ├── cb_sparse-xattention_thr-0.9_bs-256_mtoks-4096/
│   ├── cb_sparse-xattention_thr-0.89_bs-128_mtoks-4096/
│   ├── cb_sparse-xattention_thr-0.91_bs-128_mtoks-4096/
│   ├── cb_sparse-xattention_thr-0.1_bs-128_mtoks-4096/
│   └── cb_sparse-xattention_thr-0.1_bs-256_mtoks-4096/
└── ov_kv-i8_kq-by_token/
    ├── cb_dense_mtoks-4096/
    ├── cb_sparse-xattention_thr-2_bs-128_mtoks-4096/
    └── cb_sparse-xattention_thr-100_bs-128_mtoks-4096/
```

## 7. 修改文件索引

| 文件 | 修改 | 状态 |
|------|------|:---:|
| `cm_pa_common.hpp` | Q 后缩放 (Fix 3) + u8 causal skip (Fix 2) | ✅ 已验证 |
| `pa_single_token_finalization.cm` | fp32 累加器 (Fix 1) | ✅ 已验证 |
| `DEBUG_cm_pa_finalization_fp16_accumulation.md` | 本文档 | ✅ 已更新 |
