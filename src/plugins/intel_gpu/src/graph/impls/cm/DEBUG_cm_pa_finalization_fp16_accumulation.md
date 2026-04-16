# Debug Report: CM Paged Attention Quality Regression vs Dense OCL PA

**Date:** 2026-04-10 (resolved 2026-04-10, rebased 2026-04-13)  
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

首 token "The" 一致，**第 2 个 token 就发散** → 问题同时影响 PREFILL 和 DECODE 路径。

### 2.3 Finalization 内核代码审查

审查 `pa_single_token_finalization.cm` 发现关键 bug：

```cpp
// 变量名叫 "f32" 但声明为 half（fp16）！
matrix<half, 1, REDUCE_SPLIT_SIZE> out_mat_f32 = 0;     // ← BUG
// cm_mul<half> 将 float 运算结果截断为 fp16，再累加到 fp16 累加器
out_mat_f32 += cm_mul<half>(data_mat, (float)(lse_value/total_lse));  // ← BUG
```

### 2.4 Q 预缩放精度分析

Multi-token kernel 中 Q 加载时做了 fp16 精度的预缩放：

```cpp
rQ[ri].format<half>() = cm_mul<half>(rQ[ri].format<half>(), (half)scale_factor);
```

`(half)scale_factor` 将 $1/\sqrt{128} = 0.088388...$ 截断为 fp16，更关键的是 **Q 向量乘以 fp16 系数后，有效精度降为 fp16 级别**，再参与 fp16×fp16→fp32 DPAS。这导致每步 QK score 携带额外误差，在 online softmax 的多次迭代中复合放大。

## 3. 修复方案（✅ 已验证生效）

三个修改**单独测试时均无可见效果**，但**组合应用后成功修复质量回归**。这说明根因是多个 fp16 精度损失点的**复合效应**，每个单独不足以翻转 top-1 token，但叠加后超过了 softmax 概率分布的翻转阈值。

### Fix 1: Finalization FP32 Accumulator
- **文件：** `pa_single_token_finalization.cm`
- **改动：** `matrix<half>` → `matrix<float>`, `cm_mul<half>` → `cm_mul<float>`
- **修复点：** DECODE 阶段 partition 合并从 fp16 累加改为 fp32 累加
- **备注：** 此修复已在上游 master HEAD 中（代码重构时已修正）

### Fix 2: Causal Early Termination（u8 路径）
- **文件：** `cm_pa_xe1.hpp`, `cm_pa_xe2.hpp`（原 `cm_pa_common.hpp`，已重构拆分）
- **改动：** u8 路径 barrier/SLM load 后加 `if (causal_left < 0) { causal_left -= kv_step; continue; }`
- **修复点：** 消除 fully-masked causal block 的不必要 online softmax 补偿迭代

### Fix 3: Q 后缩放 ⭐ 主要贡献
- **文件：** `cm_pa_xe1.hpp`, `cm_pa_xe2.hpp`（u8 和 f16 全部路径）
- **改动：** 移除 `rQ *= (half)scale_factor`，改为 QK DPAS 后 `St = cm_mul<float>(St, (float)scale_factor)`
- **修复点：** 避免 Q 向量在 DPAS 前被 fp16 scale_factor 截断，保留全精度参与矩阵乘法

### 根因分析

原始代码中 `rQ *= (half)scale_factor` 有两层精度损失：
1. `scale_factor`（$1/\sqrt{128}$）被截断为 fp16（10-bit mantissa）
2. **更关键**：Q 向量乘以 fp16 系数后，有效精度降为 fp16 级别，再参与 fp16×fp16→fp32 DPAS

这导致每步 QK score 携带额外误差，在 online softmax 的 `cm_exp((old_max - new_max) * log2e)` 补偿中逐步累积。对于长序列（>2000 tokens），累积误差足以改变 softmax 概率排序。

叠加 finalization 的 fp16 累加器（Fix 1）和 causal 区域的多余迭代（Fix 2），三者共同将误差推过翻转阈值。

## 4. 参考：CM vs OCL 架构差异

### PREFILL 路径
- **OCL:** `sdpa_opt.cl` → 读原始 K/V 输入张量（contiguous buffer），不经 block-table
- **CM:** `pa_multi_token.cm` → 读 paged KV cache（`block_indices` 间接寻址）

### DECODE 路径
- **OCL:** `paged_attention_opt.cl` STAGE_0 → 读 paged cache → SLM 存 P (fp16) → PV
- **CM:** `pa_single_token.cm` → 读 paged cache → 寄存器存 P (fp16) → PV DPAS
- OCL block_size=16, CM block_size=256

### CM 与 OCL 精度差异汇总

| # | 差异点 | CM 实现 | OCL 实现 | 状态 |
|---|--------|---------|----------|:---:|
| 1 | Q 缩放 | ~~`rQ *= (half)scale_factor`~~ → `St *= (float)scale_factor` | Q 不预缩放，QK 后乘 `(float)SCALE_FACTOR` | ✅ 已修复 |
| 2 | 最终归一化 | `cm_inv(cur_sum)` 近似倒数 (~1 ULP) | IEEE 除法 `/=` | — 未修（不影响） |
| 3 | exp 函数 | `cm_exp(x * log2e)` (base-2 hardware exp) | `native_exp(x)` (~0.5 ULP) | — 未修（不影响） |
| 4 | Causal over-iteration | U8: `continue` skip | 每 token 独立 seq_len | ✅ 已修复 |
| 5 | Finalization 累加器 | ~~fp16~~ → fp32 | fp32 | ✅ 已修复 |
| 6 | P·V 精度 | P fp16, DPAS fp32 累加 | P fp16, `mad()` fp16 | — 架构差异（CM 更优） |

## 5. 代码重构适配（2026-04-13）

### 5.1 背景

在 rebase 到最新 master（commit `9ac988fde1`）时发现，原 `cm_pa_common.hpp` 已被上游重构为两个文件：

| 旧文件 | 新文件 | 说明 |
|--------|--------|------|
| `cm_pa_common.hpp` | `cm_pa_xe1.hpp` | Xe1 架构（ARL-H 等），使用 gather load + `cm_svm_block_read` |
| `cm_pa_common.hpp` | `cm_pa_xe2.hpp` | Xe2 架构（BMG/LNL 等），使用 LSC 2D block load + prefetch |

两个文件各自包含 u8 路径（`pa_lsc_u8`）和 f16 路径（`pa_kernel_lsc_prefetch_f16`），且 Xe2 的 f16 路径额外有 `OPTIMIZED_SPARSE_PIPELINE` 分支。

### 5.2 Fix 1（Finalization fp32）已在上游

`pa_single_token_finalization.cm` 在当前 master HEAD 已经使用 `matrix<float>` 和 `cm_mul<float>`，无需额外修改。

### 5.3 修改对照表

下表列出了在新文件结构中应用 Fix 2 和 Fix 3 的所有改动点：

#### cm_pa_xe2.hpp（Xe2 架构）

| 改动 | 路径 | 位置 | 说明 |
|------|------|------|------|
| 移除 Q pre-scale | u8 `pa_lsc_u8` | Q load 循环 | `-rQ[ri].format<half>() = cm_mul<half>(..., (half)scale_factor)` |
| 添加 QK post-scale | u8 optimized sparse pipeline | `St = ugemm_KQ(...)` 之后 | `+St = cm_mul<float>(St, (float)scale_factor)` |
| 添加 causal early termination | u8 non-optimized fallback | `#endif` (sparse skip) 之后 | `+if (causal_left < 0) { causal_left -= kv_step; continue; }` |
| 添加 QK post-scale | u8 non-optimized fallback | `St = ugemm_KQ(...)` 之后 | `+St = cm_mul<float>(St, (float)scale_factor)` |
| 移除 Q pre-scale | f16 `pa_kernel_lsc_prefetch_f16` | Q load 循环 | `-rQ[ri].format<half>() = cm_mul<half>(..., (half)scale_factor)` |
| 添加 QK post-scale | f16 optimized sparse pipeline | DPAS block `}` 之后 | `+St = cm_mul<float>(St, (float)scale_factor)` |
| 添加 QK post-scale | f16 non-optimized fallback | DPAS block `}` 之后 | `+St = cm_mul<float>(St, (float)scale_factor)` |

#### cm_pa_xe1.hpp（Xe1 架构）

| 改动 | 路径 | 位置 | 说明 |
|------|------|------|------|
| 移除 Q pre-scale | u8 `pa_lsc_u8` | Q gather load 循环 | `-rQ[ri].format<half>() = cm_mul<half>(..., (half)scale_factor)` |
| 添加 causal early termination | u8 | `#endif` (sparse skip) 之后 | `+if (causal_left < 0) { causal_left -= kv_step; continue; }` |
| 添加 QK post-scale | u8 | `St = ugemm_KQ(...)` 之后 | `+St = cm_mul<float>(St, (float)scale_factor)` |
| 移除 Q pre-scale | f16 `pa_kernel_lsc_prefetch_f16` | Q gather load 循环 | `-rQ[ri].format<half>() = cm_mul<half>(..., (half)scale_factor)` |
| 添加 QK post-scale | f16 | DPAS block `}` 之后 | `+St = cm_mul<float>(St, (float)scale_factor)` |

### 5.4 Git Diff 摘要

```
 cm_pa_xe1.hpp | 13 +++++++++++--
 cm_pa_xe2.hpp | 17 +++++++++++++++--
 2 files changed, 26 insertions(+), 4 deletions(-)
```

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
| `cm_pa_xe2.hpp` | Q 后缩放 (Fix 3) × 4处 + u8 causal skip (Fix 2) | ✅ 已验证 |
| `cm_pa_xe1.hpp` | Q 后缩放 (Fix 3) × 2处 + u8 causal skip (Fix 2) | ✅ 已验证 |
| `pa_single_token_finalization.cm` | fp32 累加器 (Fix 1) — 上游已修正 | ✅ 无需改动 |
