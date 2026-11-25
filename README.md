---
title: 6in1
emoji: 🧬
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: true
license: apache-2.0
short_description: frame1 skeleton + five small models, exact logit fusion
models:
  - IFM/K2-Horizon-0.9B
  - paradigma-inc/limite-1b-violetto
  - Qwen/Qwen3.5-0.8B
  - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  - LiquidAI/LFM2.5-Embedding-350M
---

# 🧬 6in1 — frame1 的骨骼，五個模型的器官

**六合一**：以 [frame1](https://github.com/vv0101v01/frame1)（MoonBit，六輪 28 module / 113 測試）為骨骼，
把五個架構、維度、詞表完全不同的模型合成一個系統：

| 器官 | 模型 | 角色 |
|---|---|---|
| 規劃 / 代碼 / STEM | **K2-Horizon-0.9B** | `K2HorizonForCausalLM`，64 256 詞表，`reasoning_effort=low` |
| 競賽數學 | **Limite-1B-Violetto** | `LimiteForCausalLM`，固定 `\boxed{}` system prompt，MUDD / value-embedding / sigmoid softcap |
| 長鏈推理 | **DeepSeek-R1-Distill-Qwen-1.5B** | `Qwen2ForCausalLM`，`<think>` 預填 |
| 快速多語 | **Qwen3.5-0.8B** | `Qwen3_5ForConditionalGeneration`（VLM），non-thinking |
| 路由 / 判斷 / 度量 | **LFM2.5-Embedding-350M** | 1024-d 雙向 encoder，CLS pooling |
| **骨骼** | **frame1** | 狀態空間、耦合圖、共識、可判定憑證 |

## L2：精確 logit 融合（不是近似）

四個生成器詞表兩兩不同（64 256 / 151 680 / 248 320 / 151 936），**權重直接合併不可行**。
但佢哋嘅 byte-level BPE 表面字母表交集有 **42 207 個 byte-identical token**（`audit_vocab.py` 可重現；
Space 啟動時亦會即時計算並顯示在 Diagnostics），覆蓋率隨語言而變（見下表）。

| 文字樣本 | K2 | Limite | Qwen3.5 | R1 |
|---|---|---|---|---|
| 中英混合證明題 | 100.0 % | 90.3 % | 95.1 % | 90.3 % |
| 粵語問句 | 100.0 % | 82.1 % | 89.7 % | 82.1 % |
| Python 代碼 | 91.6 % | 84.4 % | 91.7 % | 84.4 % |

兩兩交集（實測）：K2∩Limite **42 991** · K2∩Qwen3.5 **50 939** · K2∩R1 **42 990** ·
Limite∩Qwen3.5 **131 610** · Limite∩R1 **151 660** · Qwen3.5∩R1 **131 603**。
（早前版本寫 42 967 / 42 184 等數字，來自一個已廢棄嘅原型 `ModelVocab.build`——佢把 BPE 的 marker 字元當 UTF-8 編碼，令計數偏差；現已全部改為即時計算。）

逐步解碼時：

1. 每個模型對同一位置輸出 logits → 只取交集上嘅 id → **限制 + 重新歸一化**；
2. 合成 `p ∝ ∏ p_m^{w_m}`（product / log-linear）或 `Σ w_m p_m`（mixture）；
3. 揀一個 token → **四個模型各自用自己的切法**接收同一個表面符號，繼續。

限制到交集只係乘上一個「每步常數」；product 規則下該常數在融合 softmax 中**完全抵銷**，
所以融合係精確嘅，冇任何近似。三個 pseudo-token（END、`</think>`、answer pragma）處理停用符號與 budget forcing。
實作見 `spine.py`（self-test 用可手算嘅假模型驗證 product = 幾何平均、mixture = 線性混合）、`fuse.py`。

## L1：系統合成（frame1 骨骼）

- **LFM 路由**：query 向量對 frame1 128 張 agent 卡做 top-16 投票 → 四個生成器權重；
- **耦合圖**：k-NN + 跨家族橋 + Kruskal 修復，λ₂ 由 ring 0.0024 升到 **0.540**（1275 邊）；
- **共識診斷**：逐步 top-1 一致率、Jensen–Shannon 散度、邊緣度（誰主導）；
- **budget forcing**：思考到預算 75 % 強制收 `</think>`，留答案空間。

## 五個模型嘅官方實測（方案文件）

`docs/HEXAD_6IN1_PLAN.md` 記錄 K2 64 256 詞表 / Limite 151 680（AIME 2026 94.01 %）/ Qwen3.5 248 320 詞表（201 語言）/ R1 151 936 詞表，
以及 frame1 四條管線在 128 × 640 下全部收斂（wasm 單核 13.4 秒）。

## 相容性修補（transformers 5.15）

- **Limite** 讀 `config.use_cache`，5.15 已從 `PretrainedConfig` 移除 → 載入時補回；
- **LFM2.5-Embedding** 嘅 remote code 對 4.56 寫，5.15 會多傳 `seq_idx=` 且刪咗 `slow_forward` → 加 tolerant 包裝；
- **Qwen3.5** 係 VLM（`Qwen3_5ForConditionalGeneration`）→ 用 `AutoModelForImageTextToText` 載入；
- **K2** 自帶 `auto_map` → `trust_remote_code=True`。

所有修補都係無害 no-op（舊/新版本唔需要時唔會觸發）。
