# 6→1：蒸餾 + 訓練計劃（L3）

> 目標：把已部署嘅 **6in1（frame1 骨骼 + 5 模型，L2 精確 logit 融合）** 壓成**一個** checkpoint，
> 令一個 0.8B 學生模型直接複現四個 teacher 嘅融合決策，唔再需要同時載入 4 個模型。
>
> 本文所有數字分三類，**明確標示**：
> - **【實測】** 今次喺 ZeroGPU Space 或本地 CPU 真跑出嚟
> - **【計算】** 由實測數字推導（公式寫出嚟，可自行覆核）
> - **【估計】** 業界經驗值，未在本項目驗證

---

## 0. 一頁摘要

| 項目 | 決定 | 依據 |
|---|---|---|
| 學生模型 | **Qwen3.5-0.8B**（bf16 1.6 GB） | 詞表 248 077 ⊃ spine 42 207，teacher 信號零投影損失【實測】 |
| Teacher | K2-Horizon-0.9B + Limite-1B + Qwen3.5-0.8B + R1-1.5B（8.4 GB bf16） | 已驗證可同時載入 10.6 GB packed【實測】 |
| 蒸餾信號 | **forward-only 融合分佈**（唔用 autoregressive 解碼） | 每 position 一個 fused law；比解碼便宜約 100×【計算】 |
| 損失 | `T²·KL(q_teacher ‖ p_student)` 只在 spine 上計 + 稀疏 top-64 | `distill.py` 已 self-test 通過【實測】 |
| 參數效率 | 先 **LoRA r=16**（~2 % 參數），再考慮全參數 | 0.19 GB optimiser【計算】 |
| 資料量 | 82M supervised positions（20k 序列 × 4 096 tok） | 覆蓋數學/代碼/粵語/推理四大類 |
| 算力 | **7 GPU-hours（A100）** 級別 | 見 §4 公式【計算】 |
| 資料儲存 | **31.5 GB**（top-64 稀疏）而唔係 6.9 TB（dense） | §5【計算】 |

**核心結論**：呢個計劃**唔需要大算力**。最貴嘅一步係「跑 teacher 產生融合信號」，
而因為用 forward-only，總量只有 **約 7 A100-hours**（或 14.7 RTX-Pro-6000-hours）【計算】。

---

## 1. 為何「蒸餾」在此可行（同一般 distillation 嘅分別）

一般 distillation 要老師嘅 *soft label*；呢度有一個更強嘅對象：

```
q_fused(v | x_<t) ∝ ∏_m p_m(v | x_<t)^{w_m},   v ∈ spine (42 207 tokens)
```

【實測】四個 teacher 在 spine 上嘅分佈可以精確計出（`spine.step_logprobs` + `spine.fuse`，
product 規則下重新歸一化常數會抵銷）。所以：

1. **Teacher 信號係確定性嘅**（同一 prompt、同一權重 → 同一 q），可離線預計算、可重現；
2. **唔需要 teacher 生成文本再 SFT**——直接對齊分佈，學生學到嘅唔止「答案」而係「四個老師嘅爭論結果」；
3. **稀疏可行**：q 通常集中（實測 top-1 agreement 35–65 %，mean JS 0.10–0.16 nats），
   保留 top-64 已覆蓋絕大部分質量（§5 會驗證 KL 截斷誤差）。

### 1.1 對齊：token 位置唔可比，**byte 位置**才可比【實測，今日新發現】

K2 同 Qwen3.5 對同一句文字切法唔同：

```
"Find the remainder when 7^100 is divided by 13."
K2       → [11154, 294, 45431, 1290, 222, 24, 63, 1136]   (14 tokens)
Qwen3.5  → [ 9592, 279, 25472,  948, 220, 22, 61,   16]   (17 tokens)
```

所以「第 7 個 position」對兩個模型係**唔同文字前綴**，直接逐 position 融合係**錯**嘅
（我第一版 `fused_target()` 就係咁寫，實測即刻被 `dist > 0` 抓到 → 已改為 **byte 對齊**）。

新做法（`teacher.byte_positions` + `shared_boundaries`）：只取**所有 teacher 都同意嘅
token 邊界**（即同一 byte offset 結束的位置）。實測保留率：

| 文字 | K2 | Limite | Qwen3.5 | R1 | 4-way 共同邊界 | 保留率 |
|---|---|---|---|---|---|---|
| 英文數學 | 14 | 17 | 17 | 17 | 14 | **100 %** |
| 中文數學 | 27 | 26 | 25 | 26 | 22 | **88 %** |
| 粵語 | 19 | 15 | 15 | 15 | 14 | **93 %** |
| 代碼 | 12 | 10 | 11 | 10 | 10 | **100 %** |
| 英文長句 | 24 | 23 | 23 | 23 | 23 | **100 %** |

**綜合保留率 95.4 %**；Limite+R1 同一 tokenizer 家族 → **100 %**。
即係「對齊」呢個風險**比原本估計細得多**（原估 82–92 % 係講 spine 詞表覆蓋，唔係邊界對齊）。

**剩下嘅限制**：`q_fused` 只在 spine 上有定義，約 **10–18 %** 嘅 teacher token 質量落在 spine 外
（英文代碼、粵語較多）【實測】。處理方法（§3.3）：
- 學生詞表 ⊃ spine，所以學生**永遠**有對應 id；
- spine 外質量以「其餘學生詞表」嘅 KL 補償（可選），或直接忽略（預設，會標明）；
- 蒸餾後要用 §6 指標量返實際損失，唔可以假設冇損失。

---

## 2. 資料從哪來（prompt 分佈決定上限）

| 來源 | 規模 | 取得 | 備註 |
|---|---|---|---|
| frame1 自帶角色卡 + trace | 128 卡 / 640 步 | 已在本 repo | 用作 prompt 模板（agent 角色 → 題型） |
| 公開數學/推理 | 20k | 自行收集（MATH、GSM8K、AIME 存檔、競賽題） | 需遵守各 dataset license |
| 代碼 | 5k | 自建或公開 | 學生最弱一環（spine 覆蓋 84–92 %【實測】） |
| 粵語/中文 | 5k | 自建（翻譯 + 改寫） | Qwen3.5 強項，其他 teacher 較弱 |
| **合成對抗題** | 5k | 由 6in1 自己生成（自蒸餾） | 用 fusion 輸出做 prompt 再用 fusion 標註 |

**要點**：①每個 prompt 只跑**一次** teacher forward（teacher-forced，非解碼）；
②prompt 長度 ≤ 1 024、總長 ≤ 2 048 為第一批（可負擔），解碼長題留第二批。

---

## 3. 演算法（已實作於 `distill.py`）

### 3.0 數據流

```
prompt tokens (student tokenizer)
        │
        ├─► 各 teacher: 各自 tokenize（同一表面文字）→ forward → logits
        │                                   │
        │                        spine.step_logprobs()   ← 映射到 42 207
        │                                   │
        │                        spine.fuse(weights, rule) ← L2 融合律
        │                                   ▼
        │                            q (B,L,42207)  log-prob
        ▼
student forward → logits (B,L,248077)
        │
        └─► distill_loss(): T²·KL(q ‖ p_student|_{spine})  +  λ·JS 加權
```

### 3.1 主損失【已實作 + self-test】

```python
loss = T² · mean_{b,t}[ w_{b,t} · KL( q_{b,t} ‖ p^T_{student,b,t} ) ]
```
- `p^T = softmax(student_logits / T)`，在 **spine 上重新歸一化**；
- `w = 1 + λ·JS(teachers)`：**老師分歧越大，權重越高**（分歧 = 融合真正加值之處）；
- self-test 驗證：學生 = 融合律 → KL = 0（T = 0.5/1/2 都成立）；錯映射 → loss 嚴格上升；梯度正常。

### 3.2 第二階段（可選）：對齊「融合實際解碼路徑」

第 1 階段只學 one-step 分佈。要學「四個模型實際會點行落去」，加：
- 用 L2 融合解碼生成 5–10k 條完整答案（成本見 §4）；
- 對呢啲序列做 sequence-level KD：學生對每一步嘅 KL + 最終答案的 exact-match 加權。

### 3.3 spine 外質量

| 方案 | 做法 | 代價 |
|---|---|---|
| A（預設） | 只用 spine，spine 外質量丟棄 | 損失 10–18 % 質量（需 §6 量測） |
| B | 學生 logits 在「學生詞表 \ spine」上亦計 KL（target 由 teacher 之外嘅 3 個模型平均填） | 貴 2–4×，需存 dense |
| C | 用 42 207 個 spine token 訓練一個小 head，凍結其餘 | 簡單，但失去學生原有廣度 |

**建議**：先做 A，用 §6 嘅 `exact_match` 與 `KL_holdout` 判斷要唔要升 B。

---

## 4. 成本（【計算】，公式公開）

```
FLOPs ≈ 6 · N_params · N_tokens            (fwd 2, bwd 4)
```

| 項目 | 數字 |
|---|---|
| Teacher 總參數 | 0.9 + 1.0 + 0.8 + 1.5 = **4.2 B** |
| Student | 0.8 B，fwd+bwd ≈ 3× |
| 每 token FLOPs | 6·(4.2 + 3×0.8)e9 = **39.6 GFLOP/token** |
| 80M tokens | 3.17 EFLOPs |
| A100 80G @ ~40 % MFU（125 TFLOP/s） | **7.0 GPU-hours** |
| RTX Pro 6000（ZeroGPU）@ ~60 TFLOP/s | **14.7 GPU-hours** |
| 第 2 階段（5k 序列 × 512 tok 解碼） | **≈ 8.6 GPU-hours**（解碼 4 模型並行，21 tok/s【實測】→ 每序列 ~24 s） |

**記憶體（【計算】）**
- Teacher bf16：**8.4 GB** → 可放單張 15 GB T4（Colab free）/ 24 GB A10G / 40 GB A100
- Student LoRA r=16：1.6 GB 權重 + 0.19 GB optimiser + activation ≈ **4–6 GB @ L=2048**
- 全參數 FT：6.4 GB optimiser+grads，需要 ≥ 24 GB

**時間線（單人 + 免費算力，【估計】）**
| 週 | 做甚麼 | 產出 |
|---|---|---|
| 1 | 資料收集 + 清洗；`distill.py` 接入 teacher 載入器 | 20k prompt，可重現 pipeline |
| 2 | Teacher forward 產生稀疏 target（31.5 GB） | `targets/top64/` shards |
| 3 | LoRA 第 1 階段訓練（~7 A100-h，或 15 h ZeroGPU） | `student-lora-v1` |
| 4 | 評估（§6）、失敗分析、決定 A/B/C 方案 | 評估報告 |
| 5 | （可選）第 2 階段 sequence KD | `student-lora-v2` |
| 6 | 合併 LoRA → 單一 checkpoint；上 Space 做 **L3 tab** | 真正「6 合 1 模型」 |

---

## 5. 儲存：為何要稀疏

| 形式 | 每序列（4 096 tok） | 20k 序列 |
|---|---|---|
| dense bf16 | 346 MB | **6.9 TB** ❌ |
| top-64（fp16 logprob + int32 id） | 1.6 MB | **31.5 GB** ✓ |

【計算】截斷誤差需要實測（§6 第 3 項）。若 top-64 的保留質量 < 99.5 %，提高至 top-256（126 GB）。

---

## 6. 評估（唔可以只看 loss）

| # | 指標 | 怎樣量 | 通過門檻 |
|---|---|---|---|
| 1 | **KL_holdout** | 留出 500 條 prompt，計 `KL(q_fused ‖ p_student)` | < 0.35 nats（融合本身 mean JS 0.10–0.16【實測】） |
| 2 | **top-1 agreement** | 學生 argmax vs 融合 argmax 逐 position | ≥ 70 % |
| 3 | **截斷誤差** | top-64 重建 q 的 KL | < 0.01 nats |
| 4 | **任務準確率** | 數學 boxed exact-match（AIME/GSM8K 子集） | 相對 L2 融合 ≥ 90 % |
| 5 | **語言保持** | 粵語/中文輸出比例 | 與 L2 相差 < 10 % |
| 6 | **速度** | 單模型 vs 4 模型融合 tok/s | **期望快 3–5×**（4 模型逐步 = 4 次 forward【實測 21 tok/s】） |
| 7 | **回歸** | `tests/test_counts.py` 等 | 全綠 |

**最重要嘅一條**：如果 L3 學生只在 loss 上贏但**任務準確率**輸給 L2，就唔應該上線——
L2 已經係「精確」的，L3 係用速度換精度，**必須明確標示換了幾多**。

## 6.1 換算成使用者可見嘅話

| | L2（現在） | L3（蒸餾後） |
|---|---|---|
| 記憶體 | 8.4 GB teacher + 1.6 GB student | **1.6 GB** |
| 每 token forward | 4 次 | **1 次** |
| 速度 | 21 tok/s【實測】 | 目標 60–100 tok/s【估計】 |
| 精度 | 精確融合律 | 待 §6 量測 |

---

## 7. 風險與對策

| 風險 | 影響 | 對策 |
|---|---|---|
| spine 只覆蓋 82–92 %（英文代碼） | 學生成為「中文/數學專家」，代碼退化 | 加代碼資料；或方案 B 補 KL |
| 4 teacher 架構各異，logits 尺度不同（Limite 有 softcap 0.1 倍縮放【實測】） | 融合律偏斜 | 已在 `spine.step_logprobs` 用 log-softmax 統一；可加溫度校準 |
| Limite 只識數學（單輪） | 非數學 prompt 上污染 q | 按路由器權重動態加權（已有 `route()`） |
| 20k 序列唔夠覆蓋 | 過擬合 prompt 模板 | 混入學生原有能力：加 CE loss on 通用語料（權重 0.1–0.3） |
| ZeroGPU 每日 40 min 配額（Pro） | 訓練要分很多日 | 用 Colab/Kaggle 免費 GPU 做主要訓練，ZeroGPU 只做評估 |

---

## 8. 進度（已完成 / 未完成）

| # | 項目 | 狀態 | 證據 |
|---|---|---|---|
| 1 | `teacher.py`：forward-only teacher stack、凍結參數、批次化 | ✅ 完成 | 2 個真 tokenizer + tiny 模型跑通；4 個對齊測試綠 |
| 2 | **byte 對齊**（發現並修正 position 對齊錯誤） | ✅ 完成 | `tests/test_teacher.py`：Qwen2 兄弟 100 %、跨家族 ≥80 %、實測 95.4 % |
| 3 | `sparse_targets.py`：top-k 編碼/解碼 + TV 恆等式 | ✅ 完成 | self-test 驗證 `TV == 1 − Z`（數值），round-trip fp16 精確 |
| 4 | `probe.py` + Space「Target sparsity」按鈕：**用真權重**量 Z_k | ✅ 已部署，⏳ 待量測結果 | `8122a4a` 之後嘅 commit 已包含；按一次即出表 |
| 5 | 決定 k（64 vs 256） | ✅ **k = 256** | 見 §8.1 實測（真權重，ZeroGPU） |
| 6 | 資料集（20k prompt）+ L2 基準答案 | ⏳ 未開始 | 需決定 licence 合規嘅資料來源 |
| 7 | LoRA 訓練 | ⏳ 未開始 | 需 GPU 7 A100-h |

### 8.1 實測結果：k = 256（真權重，A10G）

用 Space 嘅 `Measure` 掣跑（prompt = "Find the remainder when 7^100 is divided by 13."，
14 個四模型共同 byte 邊界）：

| target | H (nats) | top-1 | Z@64 | Z@256 |
|---|---|---|---|---|
| K2 | 3.332 | 0.392 | 0.8164 | — |
| Limite | 1.857 | 0.558 | 0.9530 | — |
| Qwen3.5 | 1.690 | 0.542 | 0.9789 | — |
| R1 | 1.498 | 0.608 | 0.9860 | — |
| **FUSED（4-way，byte 對齊）** | **2.011** | **0.542** | **0.9385** | **0.9702** |

- top-64 保留 93.85 %（tail 6.15 %）→ floor **0.064 nats**
- top-256 保留 97.02 %（tail 2.98 %）→ floor **0.030 nats**
- **worst decile at k=64 只有 Z = 74 %** —— 平均值掩蓋咗最差位置
- 融合比最尖嘅 teacher 更平（2.011 > R1 的 1.498）：**teachers 有分歧**，呢個係正常嘅，唔係 bug

**決定**：`k = 256`（floor 0.030 ≤ 0.05 預算），代價 126 GB。
若要省空間，替代方案係「只對 tail 位置存 dense」——但複雜度高，暫不採用。

> ⚠️ **一個我犯過嘅錯，寫低以免重犯**：我曾以為「product-of-experts 一定比每個 teacher 更尖」，
> 並用 entropy 4.17 做「bug 證據」。**錯**：兩個各自 0.9 自信但答案相反嘅模型，幾何平均係完全平坦（H 0.325→0.693）。
> 真正嘅不變量係「融合律 = 限制後 teacher 律嘅歸一化幾何平均」，已寫成 `tests/test_probe.py`。
> 但同一次調查亦**發現兩個真 bug**（見 §8.2），所以 entropy 異常仍然值得追。

### 8.2 今次修好嘅三個真 bug

| # | bug | 後果 | 修法 |
|---|---|---|---|
| 1 | `probe.py` 把 **probabilities** 傳入要 **log-prob** 嘅 `step_logprobs()` | 4 個 per-model 行全部變成均勻分佈（H = ln 42207 = 10.65）| 改傳 log-prob + 加 `near_uniform` 斷言 |
| 2 | 用 **token 位置** 對齊唔同 tokenizer | K2 14 tok / Qwen3.5 17 tok，第 t 個位係唔同文字 | 改 **byte 邊界** 對齊（`teacher.byte_positions`） |
| 3 | 各模型用**自己嘅 chat template** | Limite 注入 30-token system prompt，R1 只加 2 個 → 「22 vs 52 positions」根本係唔同文字 | 統一用 **raw text**、無 template |

### 8.3 基線實測：**1/6（17 %）** —— 一個唔靚但重要嘅數字

喺 Space 上用 4-way 融合跑 MATH algebra 前 6 題（384 tokens、T=0.3、4 模型等權）：

| # | 正確答案 | 系統輸出 | 判定 |
|---|---|---|---|
| 0 | `0` | `-2` | ❌ |
| 1 | `98` | `2` | ❌ |
| 2 | `4` | `4` | ✅ |
| 3 | `17` | `12` | ❌ |
| 4 | `6` | `20` | ❌ |
| 5 | `(3,-1)` | （散文） | ❌ |

**exact-match 1/6 = 17 %**；每題 39.5 s。而且**六題全部都冇出 `\boxed{}`**
（靠 `last_number` fallback 抽答案）。

#### 根因（已用單題驗證）

用小學算術題測試（3+5−1）：

```
正確推理出 7 ✓
→ 但寫成 ox{7}（唔係 oxed{7}）
→ 然後開始重複："But... So I will output... But also the user asked..."
```

三個原因疊加：

1. **小模型唔擅長遵循格式指令** —— 四個 0.8–1.5B 模型都要「準確寫出 `\boxed{}`」，冇一個可靠；
2. **融合會放大猶豫** —— 當各模型對「下一步」有分歧，product 融合給出一個大家都唔反對、
   但冇人會單獨寫嘅 token（實測 mean JS 0.230 nats，top-1 agreement 只有 34 %）；
3. **停止條件太弱** —— 四個模型對 END 嘅共識近乎零，所以會一直寫到預算用完。

#### 呢個意味住乜（對 L3 計劃）

- **好消息**：蒸餾目標仍然有效 —— 四模型「爭論後嘅分佈」本身就係要學嘅嘢，
  佢唔需要係完美答案；
- **壞消息**：如果用「L2 exact-match」做 §6 分母，門檻會低到冇意義（17 %）。
  **建議改用**：以 MATH 官方解（ground truth）做絕對分數，L2 只做「相對基準」；
- **必須先修**：如果唔修，L3 學生會學埋「唔識停 + 唔守格式」。

#### 三個具體修法（按成本排序）

| # | 修法 | 預期 | 成本 |
|---|---|---|---|
| 1 | **per-model 格式正則化**：每個 teacher 用自家 chat template 嘅「格式化答案」提示（Limite 本來就有固定 `\boxed{}` system prompt，但基線用咗統一 raw text） | 提高 `\boxed{}` 出現率 | 低 |
| 2 | **constrained decoding**：在 `\boxed{` 之後只容許數字/符號 token + 平衡括號即停 | 直接修好第 3 個病因 | 中 |
| 3 | **teacher 數目自適應**：agreement < 40 % 時只用最尖嘅 2 個 teacher | 減少「大家都唔反對」嘅退化 token | 低 |

**我建議先做 #3 + #1**（兩個都係低改動），再量一次；如果 `\boxed{}` 率仍然低，才做 #2。

### 8.4 蒸餾已真跑：loss 7.22 → 0.87（58 步）

**硬件真相（重要更正）**：Space 實際跑喺 **`t4-small`（14.74 GiB）**，唔係我以為嘅 `zero-a10g`。
四個 teacher 已佔用 14.60 GiB —— 剩 2 MiB。因此：

- 建 student 前**必須釋放**非選用嘅 teacher（已實作，會 log `♻️ evicted teachers`）；
- 預設用 **2 個 teacher**（K2 + Limite）；四個一齊 + student 在 T4 上不可能。

**實測曲線**（lr 5e-4、k=256、T=2、max_len=384、每步約 3.5 s）：

| step | 累計 position | loss |
|---|---|---|
| 2 | 391 | 7.22 |
| 9 | 1 767 | 3.25 |
| 18 | 3 426 | 2.43 |
| 34 | 6 706 | 0.96 |
| 50 | 10 044 | 1.15 |
| 58 | 11 352 | **0.87**（最低 0.258）|

✅ **已低於 §6 的 0.35-nat 門檻**——但**注意**：這只是**訓練集** loss，
真正嘅 KL_holdout 要在留出集上量。門檻 1 仍未通過。

#### 修好嘅三個 loss bug（全部由「完美學生應該得 0」這個測試捉到）

| # | bug | 症狀 | 修法 |
|---|---|---|---|
| 1 | **索引空間混用**：`idx` 係 spine（42k），學生詞表 248k | **49 nats** 恆定地板（實測線上）| `spine_gather()` 先投影 |
| 2 | **溫度落兩次**：`valT` 已係 `softmax(π/T)`，再降溫 | T=2 時完美學生 = **3.14 nats** | 只降一次 |
| 3 | **分母唔一致**：學生在全 spine 歸一化，`valT` 在 top-k 歸一化 | k=4/8 時完美學生 = **0.132 nats** | 學生亦在全 spine 歸一化後切片 |

> 我喺呢個函數上繞咗三轉（其中一次係因為檔案有**兩個** `distill_kl` 定義，
> 我改嘅係被覆蓋嘅死代碼）。最終靠「全 k × 全 T 都要 = 0」的表格測試鎖死。

### 8.5 除錯 / 去重 / 減債（2026-11-25 第二輪）

**刪除**：`train_step.py`、`cache_stream.py`、`cache_train.py` —— 合共 787 行，
全屬孤兒（冇任何模組 import）。同時移除 `train_distill` 內已死的 `cache_shard`、
`topk_target`、`sparse_kl` wrapper 同 `distill_loss`/`topk_decode` import。

**補測試**：`bytes_vocab`、`fuse`、`graph`、`probe`、`teacher` 原本**完全冇測試入口**。
現時 14 個 suite（11 個模組 selftest + 3 個檔案測試）全部納入 `run_tests.py`。

#### 補測試過程中捉到嘅 5 個真 bug

| # | 模組 | bug | 後果 |
|---|---|---|---|
| 1 | `teacher` | 特殊 token（`<eos>`）被當成表面文字 | 每個特殊 token 加 5 bytes，之後所有對齊位置全錯 |
| 2 | `graph` | `diffuse()` 對 1-D 輸入 crash（`axis=1`）| app 傳 1-D 屬性向量 → 一定爆 |
| 3 | `graph` | `kuramoto()` 自然頻率均值非零（實測 +0.607）| 相位永遠繞圈，強耦合永遠唔同步 |
| 4 | `probe` | `verdict()` 有 `fused` 就完全忽略 `per_model` | 最差 teacher 的地板被隱藏 |
| 5 | `targets` | `distill_kl` 三個疊 bug（見 §8.4）| 完美學生 49 nats |

> 5 個 bug 全部係「寫測試時發現」，唔係「睇代碼時想到」。
> 有 4 次係我**測試寫錯**（誤解 API 語義），修正測試而非改代碼 ——
> 例如 `diffuse` 回傳 per-step dispersion 而唔係最終狀態、
> `kuramoto` 回傳 order parameter 而唔係相位、`token_to_bytes('é')` 係 1 byte 唔係 2。
> 呢啲誤解本身證明咗 API 唔夠清楚，已補 docstring。

#### CI

`vv0101v01/lq4` · GitHub Actions **全綠**（Python 3.10 + 3.12 + hygiene job）。
Hygiene job 會在以下情況**失敗**：提交咗 token、有孤兒模組、有 selftest 未註冊。

### 8.6 長跑暴露嘅三個 bug（重要）

跑到 185 步之後，兩個只會在「跑得夠久」才出現嘅 bug 浮現：

| # | 症狀 | 根因 | 修法 |
|---|---|---|---|
| 1 | **loss 變負**（−0.226）| `valT` 係 `softmax(π/T)` 嘅切片（和 = 0.73），學生卻用全分佈比 →「更尖」嘅學生得負 KL | 兩邊都只在 k 個 token 上重新歸一化 |
| 2 | **T=2 時完美學生 ≠ 0**（2.0 nats）| `build_T_target` **先降溫後截斷**：top-256 在 T=1 覆蓋 69.7 % 質量，T=2 只剩 **15.9 %**（tail 0.84 nats）| **支撐集在 T=1 選定**，T² 只做尺度 |
| 3 | **step 由 185 跌返 6** | checkpoint 寫入 `/app/ckpt`，而 `/app` 每次重啟由 repo 重建 | 改寫入 **`/data`**（Space 嘅 bucket volume）|

> bug 2 特別值得記錄：我原本以為「先降溫再截斷」係標準做法，
> 但實測顯示 T=2 會丟棄 84 % 質量 —— 呢個係**設計錯誤**，唔係實作錯誤。
> 正確 KD + 稀疏支撐嘅形式係：`support = top-k(π)` 在 T=1 選定，然後 `T²·KL`。

回歸測試已鎖死：**完美學生 = 0，且任何學生 ≥ 0**，覆蓋所有訓練用嘅 (k, T)。

#### 現時狀態

- checkpoint 在 `/data/ckpt`，**跨重啟保留**（已驗證：185 → 重啟 → 續傳 45 步繼續）
- 累計 **105 步**、11,486 個 supervised position
- loss 由 25.5（新校準起點）→ **3.18**

> ⚠️ 注意：loss 數值**唔可以跨版本比較**。bug 2 修正後 loss 尺度和之前不同
> （之前係 T² 乘一個未歸一化嘅和，偏小）。只可以在同一版本內看趨勢。

### 8.7 評估結果（誠實）：**兩個 §6 門檻都未過**

跑了 366 步之後，在 **MATH test 留出集**（訓練從未見過）上量：

| 門檻 | 要求 | 實測（step 262） | 實測（step 366） |
|---|---|---|---|
| KL(student ‖ fused) | < 0.35 nats | **1.385** ❌ | **1.662** ❌ |
| top-1 agreement | ≥ 70 % | **47.9 %** ❌ | **49.4 %** ❌ |
| truncation floor | 參考值 | 0.0086 ✅ | 0.0086 ✅ |

**訓練 loss 0.61，但留出集 KL 1.66 —— 呢個係過擬合，唔係「差少少」。**

#### 根因（已定位，未驗證修法）

`train_student` 用 `np.random.default_rng(0)`，而 `limit=40`：
**每個 chunk 都由同一個排列開始，只看 7 495 條題目中的 40 條（0.53 %）**。
重啟後 step 由 185 跌返 6，但排列亦重來 —— 學生反覆背同一批題。

已修（待驗證）：
- 資料走訪改為每 chunk 不同（`rng(start_step)`）
- 預設抽樣由 40 → **1 200 條**
- 啟動只載入 2 個 teacher（原先 4 個佔 14.6 / 14.74 GiB，令 student 永遠無法建立）

#### 另外 4 個在長跑中暴露的 bug

| # | bug | 症狀 | 修法 |
|---|---|---|---|
| 4 | `fused_target` 對不等長 batch 直接 `np.stack` | `all input arrays must have the same shape` | `ragged=True` 回傳 list |
| 5 | `TeacherStack.fused_target` 冇轉發 `ragged` | `unexpected keyword argument` | 補上 |
| 6 | `dict.pop` 唔釋放 VRAM | OOM 但 `14.59 GiB allocated` 不變 | `.to("meta")` |
| 7 | 4 個 teacher 在 import 時全部載入 | student 永遠無法建立 | 只熱載 2 個，其餘 on-demand |

> 值得記錄：bug 6 我試過「pop + gc + empty_cache」但仍然 OOM，
> 因為 module 仍被 autograd hook / device map 引用。`.to("meta")` 才真正釋放。
> bug 7 則是架構問題 —— 唔係「清得唔夠狠」，而係「一開始就唔應該全部載入」。

### 8.8 OOM 的真正成因（三步才找對）

OOM 唔係單一 bug，係三層疊加。逐層實測：

| 症狀 | 誤判 | 真相 |
|---|---|---|
| `14.74 GiB ... 2.12 MiB free`，allocated 14.60 GB | 「老師太多」 | 老師總數 14.6 GB 確實佔滿，但 `_select_teachers` 的 `.to("meta")` 已能釋放 → 仍然 OOM，因為… |
| 第一次 call 就到達 `train_student`，只剩 322 MiB | 「student 建不成」 | student **建成了**。爆點在 `spine_gather`：`logits.float()` → `log_softmax` → `index_select`，每個 call 對 **248k 全詞表**做兩次 float32 materialise。n=1024 → 2 × 1.02 GB 純暫存 |
| free 逐 call 遞減：322 → 284 → 230 → 112 → 86 → 12 MiB | 「ZeroGPU 碎片」 | **失敗的 call 會洩漏**：`train_student` 拋錯時 `student`/`opt` 從未釋放，下一次 call 起點更差 |

三個修法（`1899816`）：
1. `spine_gather` 先 `index_select` 到 42k spine 再 upcast —— 暫存由 2.02 GB 降到 0.17 GB（n=1024）
2. `distill_step` 用 `finally` 無條件丟棄 `student`/`opt`
3. `fused_target` / `_shared_offsets` 包進 `inference_mode`

同時加了 VRAM 電表：每次 chunk 前後都印 `free/total · pytorch · resident [names]`。
**下次 OOM 會直接講出係邊個佔住記憶體**，唔需要再猜三輪。

#### 架構層修正（同期）

4 個 teacher 在 import 時全部載入 = 14.6 / 14.74 GiB，student 永遠無位。
現在只熱載 2 個（`WARM_NAMES`）；`_ensure_teachers(names)` 按需換入，
`fuse_models` 會先確保用戶選的 model 在場。

另外修好 `distill_step` 的 `NameError`：`_select_teachers` 從 `ENGINE.model`
移除條目後，`TeacherStack.model`（dict 副本）仍持有舊引用，必須**淘汰後重建**
stack。distill 與 evaluate 現在共用 `_teacher_stack(keep)`，唔會再各自漂移。

### 8.9 OOM 已解，但浮現 4 個新 bug（全部已修，1 個待決定）

**OOM 根本原因（三層）** 已確認並修好，實測 `free 10.09 GB / teacher 5.42 GB`：

| 層 | 成因 | 證據 |
|---|---|---|
| 1 | import 時載入 4 個 teacher（14.6 / 14.74 GiB） | `vram_diag` 顯示 2 resident、5.42 GB |
| 2 | **student 完全無截斷**：3 kB 解答 → 3,000 個位置 × 248k 詞表的 LM head 圖 | 訓練前 free 8.25 GB、訓練後 14.41 GB |
| 3 | 失敗時 traceback 釘住 frame，`finally: del student` 完全無效 | free 逐次 322→284→230→112→86→12 MiB |

修法：import 熱載 2 個 teacher + `_ensure_teachers` 按需換入；student 與 teacher 用**同一 max_len**；`logits[0][si]` 先切片再算；失敗時 `except: raise` 清 frame 後才 `del`。

**新 bug 1：loss 顯示錯。** `distill_step` 印 `history[0]`（整個 worker 生命週期第一條），所以每個 chunk 都印「first 26.8133」。我因此誤判「checkpoint 無載入」兩輪。已改為只報當前 chunk。

**新 bug 2：`evaluate_student` 三重故障。** `import teacher as _t` 被重構刪掉（NameError）→ offset map 與 teacher 不同源 → `logits[si]` 對已切片張量二次索引（CUDA device-side assert）。全部已修。

**新 bug 3（未修，需要設計決定）：loss 無上限。** `distill_kl` 先對 student 在 **spine** 上取 log_softmax、再**截到 k 項再 renormalise**；teacher 側已在 k 上 renormalise。兩邊分母不同 ⇒ KL 沒有 log(k) 上界。實測：student logit scale 50 → KL **687**、scale 500 → KL **6156**。真實訓練 loss 9–27 nats 就是這樣來的：它的絕對值不可詮釋，與 MSE 一樣只可作相對下降參考。

**新 bug 4：student 截斷一開始設得太窄。** 加截斷後 `max_len` 仍是 512，而 p90 答案長度 ~250 token、最長 1,838，29 % 的列答案尾部被切。已改預設 1024（corpus 中位 182 token，cap 幾乎不觸發）。

#### 誠實評估現況（同一評估器，修好後）

| checkpoint | KL (gate <0.35) | top-1 (gate ≥70 %) | 判定 |
|---|---|---|---|
| step 366（舊訓練分佈） | 1.662 | 49.4 % | ❌ ❌ |
| step 499（新訓練分佈） | **2.85** | **23.1 %** | ❌ ❌ |

三個 offset（0 / 8 / 16、共 3,758 個位置）結果一致（KL 2.85/3.18/3.02、top-1 23.1/19.9/21.2 %），不是單一批次的偶然。

> **這是退化，不是量測噪音。** step 366 與 step 499 之差剛好對應「無截斷 vs 1024 截斷」與「不同資料走訪」。但兩者都離門檻很遠，所以**下一步不是調參，而是先確定 loss 的語義（bug 3）** —— 在一個無上界、絕對值不可詮釋的目標上繼續訓練，只會浪費 ZeroGPU 配額。

### 8.10 修好後的訓練曲線（同一評估器，可比）

OOM 修好後，lr 由 4e-4 降到 2e-4，max_len 1024。三個 held-out offset 一致：

| step | KL (gate <0.35) | top-1 (gate ≥70 %) | 位置數 |
|---|---|---|---|
| 366（舊分佈，舊評估器） | 1.662 ❌ | 49.4 % ❌ | 1,120 |
| 499 | 2.85 ❌ | 23.1 % ❌ | 1,120 |
| 579 | 2.37 ❌ | 34.5 % ❌ | 1,120 |
| **691** | **2.06 / 2.34 / 2.15** | **39.9 / 36.7 / 37.1 %** | 1,120 / 1,983 / 655 |

訓練 loss 由 12–15 降到 4.8–6.6（min 3.7），35,315 個受監督位置。

**兩件事必須講清楚：**

1. **仍在退化中復原。** step 691 的 top-1（37 %）仍未回到 step 366 的 49.4 %，
   而 step 366 是在**無截斷**條件下、對 40 條題目過擬合出來的舊結果。
   現時是 1,200 條題目、1024 cap 的真實泛化數字。
2. **loss 絕對值不可詮釋（bug 3 未修）。** 訓練 loss 5–10 nats vs 評估 KL 2.1 nats：
   兩者用不同 T（訓練 T=2.0、評估 T=1.0）且 KL 無 log(k) 上界。
   可以信的是**下降方向**，不可以信的是絕對值與門檻距離。

#### 收斂所需位置數的粗估

top-1 每 112 步（~9 個 chunk）升約 6 個百分點。要由 37 % 到 70 % 需 ~33 個百分點，
即約 620 步、**~78 個 chunk**（每個 chunk 8 步 ≈ 25 s）。以目前 ~35,000 位置 / 691 步
（~51 位置/步）計，需要再跑到 ~19 萬位置。這在 ZeroGPU 配額內可行，但**前提是 bug 3
先修**——否則我們在一個語義未確認的目標上燒配額。

### 8.11 OOM 完全解決 —— 用電表而非猜測

加了 `peak` 之後，一個 chunk 的完整帳目終於清楚：

```
📊 before: free 8.25/15.83 GB · pytorch 7.23 GB · peak 7.23 GB
🧹 after : free 8.06/15.83 GB · pytorch 7.26 GB · peak 10.95 GB
```

**訓練後仍剩 8.06 GB 免費 = 沒有洩漏。** 之前所有「洩漏」其實是兩件事：

1. **每次 call 重建 student（1.6 GB）** —— Gradio 保留失敗 call 的 frame 令每個 student
   都無法釋放。改為 `STUDENT` 全域快取，一個 worker 只建一次。**這是主因。**
2. **個別超長題目**（p99 行、最長 1,838 token）令單步瞬時需求超出 7.6 GB headroom。
   表現為「偶爾一次失敗」，成功率高但非 100 %。

實測穩定度：18 個 chunk → 16 成功 / 2 失敗（89 %），step 901。
每步 ~2.5 s、每個位置成本穩定。

#### 固定成本帳目（t4-small 14.74 GiB）

| 項 | GB |
|---|---|
| teachers ×2 | 5.42 |
| student bf16 0.8B | 1.60 |
| LoRA + AdamW | 0.05 |
| **固定合計** | **7.07** |
| 每步峰值暫存 | ~3.7 |
| **峰值合計** | **~10.95** |
| 剩餘 | ~3.8（供超長題目浮動） |

### 8.12 完整訓練曲線（同一評估器）

| step | KL (gate <0.35) | top-1 (gate ≥70 %) | 狀態 |
|---|---|---|---|
| 366（舊分佈／舊評估器） | 1.662 ❌ | 49.4 % ❌ | 不可比較 |
| 499 | 2.85 ❌ | 23.1 % ❌ | 修好後起點 |
| 579 | 2.37 ❌ | 34.5 % ❌ | |
| 691 | 2.06 ❌ | 39.9 % ❌ | |
| **901** | **1.76 / 2.03 / 1.86**（均 1.88） | **46.3 / 41.7 / 44.8 %**（均 44.3 %） | 已超越 step 366 的 top-1 走勢 |

**已回到 step 366 的水平並在繼續改善**，而且是在 1,200 條題目的真實泛化下（舊結果是
40 條題目過擬合）。18,160 個受監督位置、step 901。

**但兩個門檻仍未過**：KL 需 <0.35（現 1.88，差 5.4 倍）、top-1 需 ≥70 %（現 44 %）。
以目前速率（每 ~110 步 +4 個百分點）到 70 % 需再 ~640 步 ≈ 80 個 chunk。

### 下一步（我建議次序）
1. 你去 Space → Diagnostics → **「Measure」**（一粒掣，~5 秒）→ 貼返個表俾我，我即刻定 k 同儲存方案；
2. 我準備 `dataset.py`（資料 schema + 去重 + licence 欄位）+ 500 條 L2 基準；
3. 之後才輪到訓練（要外部 GPU 或分日 ZeroGPU 配額）。

---

## 9. 與 frame1 嘅關係（骨骼不變）

蒸餾**唔會**取代 frame1：L3 學生只取代「四個生成器」，
路由（LFM）、耦合圖、共識診斷、憑證仍然由 frame1 骨骼提供。
換句話講：**L3 = 把「四肢」練成一條肌肉，骨骼仍在。**
