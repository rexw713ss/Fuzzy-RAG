# FuzzyRoute-RAG

[English](README.md) | [繁體中文](README.zh-TW.md)

以可解釋模糊控制器進行不確定性感知的檢索增強生成路由。

> **目前狀態：研究原型，核心模組已實作。** 已完成檢索、不確定性訊號、模糊推論、寬度校準、cluster 切分、控制器基準方法與評估函式。改寫問題建置、完整 A0/A1/A2 執行流程及答案生成尚未整合。下方數值為實作檢查與合成控制器診斷，不是資料集上的回答品質結果。

## 研究概述

檢索增強生成（retrieval-augmented generation，RAG）以檢索片段作為答案生成的上下文。FuzzyRoute-RAG 探討如何依檢索端的不確定性分配額外處理成本，透過 Type-1 Mamdani 模糊推論系統整合四個訊號，產生升級分數。

預定回答的研究問題為：

> 模糊路由能否改善回答品質與成本的取捨，並降低語意相同、措辭不同的問題之間的有害路由切換？

這是研究目標，尚非已獲證實的成果。控制器回傳隸屬度與觸發規則作為決策紀錄；能檢視決策過程，不代表選出的 action 已證實有效。

## 研究方法

下圖為預定的完整流程。共用生成器與完整 action 執行器仍在開發中。

~~~mermaid
flowchart TD
    Q["輸入問題"] --> R["BM25 與 Contriever"]
    R --> S["分數融合與不確定性訊號"]
    S --> F["模糊升級分數"]
    F --> A0["A0：直接使用 top-5"]
    F --> A1["A1：重新排序 top-20"]
    F --> A2["A2：改寫與再次檢索"]
    A0 --> G["共用 top-5 上下文與生成器"]
    A1 --> G
    A2 --> G
~~~

### 檢索與分數融合

- **BM25：** 使用 Lucene/Pyserini 檢索 Dense Passage Retrieval（DPR）的 Wikipedia corpus；$k_1=0.9$、$b=0.4$。
- **稠密檢索：** 使用 `facebook/contriever-msmarco`，依 attention mask 進行平均池化，保留未正規化向量，以內積計分。
- **Checkpoint revision：** `abe8c1493371369031bcb1e02acb754cf4e162fa`。
- 兩個 retriever 各回傳 top-50。每個問題分別做 min-max 正規化，再取等權平均；未出現在某個清單的 passage，在該 retriever 計為零分。融合清單保留 top-50。
- 完全相同的分數目前全部正規化為 1。融合與 A2 合併遇到同分時，依 passage ID 的字串表示排序。
- 稠密檢索目前對 21,015,324 個 passages 進行完整搜尋，運算與記憶體成本較高，正式搜尋後端尚待決定。
- **HotpotQA：** 由於原始 corpus 主機無法連線，改用 BEIR 版本的 HotpotQA Wikipedia abstracts（5,233,329 篇），搭配 Pyserini 預建的 BM25 與 `contriever-msmarco` indexes。稠密 index 直接以 NumPy 讀取其 FAISS 檔案，不需要安裝 FAISS 函式庫。

### 不確定性訊號

訊號的設計方向為「數值越高，越傾向升級處理」。這項假設仍需實際資料驗證，尤其 multi-hop 問題可能需要不同來源的證據。

令融合分數為 $s_1\geq s_2\geq\cdots\geq s_k$，其中 $k=10$。

| 訊號 | 意義 | 函式 |
|---|---|---|
| S1 | 最高分與次高分的差距較小 | `s1_margin` |
| S2 | BM25 與 dense ranking 的不一致 | `s2_disagreement` |
| S3 | top-10 passage embeddings 的分散程度 | `s3_dispersion` |
| S4 | top-10 融合分數的熵 | `s4_entropy` |

$$
S_1=1-\frac{s_1-s_2}{s_1-s_k+\epsilon},\qquad \epsilon=10^{-9}.
$$

S2 使用包含 residual term 的 extrapolated Rank-Biased Overlap（RBO）：

$$
\mathrm{RBO}_{EXT}@k=(1-p)\sum_{d=1}^{k}p^{d-1}A_d+p^kA_k,\qquad
S_2=1-\mathrm{RBO}_{EXT}@k.
$$

$A_d$ 為深度 $d$ 的重疊比例，$p=0.9$。相同 ranking 的 S2 約為零；完全不重疊時為 1。

$$
S_3=1-\frac{2}{k(k-1)}\sum_{i<j}\cos(\mathbf e_i,\mathbf e_j).
$$

S3 僅在計算 cosine 時正規化 embeddings；稠密檢索仍使用未正規化向量的內積。

$$
p_i=\frac{\exp(s_i/\tau)}{\sum_{j=1}^{k}\exp(s_j/\tau)},\qquad
S_4=-\frac{\sum_{i=1}^{k}p_i\log p_i}{\log k},\qquad \tau=0.1.
$$

`fit_scaler` 針對每個訊號與 dataset，使用 training clusters 估計第 5 與第 95 百分位數（P5/P95）。`apply_scaler` 將數值縮放並截斷至 $[0,1]$。呼叫端必須確保只用 training 資料擬合，因為函式本身不檢查 split 標籤。擬合前也應檢查近乎常數的訊號。

### 模糊推論與寬度校準

主要控制器使用 minimum AND、截斷輸出集合、maximum aggregation，以及 1,001 個輸出取樣點的 centroid defuzzification。

每個輸入包含 Low、Medium、High 三種標籤，交叉中心為 0.35 與 0.65。完整 $3^4=81$ 條規則由 Low = 0、Medium = 1、High = 2 的 ordinal levels 產生。令總和為 $t$：

| Ordinal sum | Consequent | 規則數 |
|---|---|---:|
| $t\leq3$ | Low | 31 |
| $t=4$ | Medium | 19 |
| $t\geq5$ | High | 31 |

固定寬度版本使用 $w=0.20$。校準版本蒐集完整 training cluster 中三組措辭配對的訊號絕對差：

$$
w_s=\operatorname{clip}\!\left(P_{75}(|S_{i,s}-S_{j,s}|),\,0.20,\,0.25\right).
$$

程式也回傳 median-based sensitivity widths、未截斷 jitter 超過 0.25 的標記，以及被跳過的不完整 clusters 數量。如果全部寬度仍為 0.20，校準未改變控制器，不能將改善歸功於獨立的 calibration 貢獻。

已實作的 single-stage operator ablations 分別使用 product t-norm、probabilistic-sum aggregation 或 zero-order Sugeno output。

### Actions 與控制器基準方法

| Action | 預定處理方式 | 目前支援 |
|---|---|---|
| A0 | 融合 top-5 直接交給生成器 | 完整執行器尚未完成 |
| A1 | rerank top-20，保留 top-5，再生成答案 | 完整執行器尚未完成 |
| A2 | 改寫、再次 hybrid retrieval、合併與去重、rerank top-30、保留 top-5 | 已完成合併與 rewrite 可用性檢查；完整執行器尚未完成 |

A2 合併時，重複 ID 保留最高融合分數，再縮放聯集並保留 30 個候選。失敗、空字串或只改變大小寫與空白的 rewrite，視為不可用。

三個 actions 預定共用 generator、context size、prompt 與 decoding settings。未來實作 fallback 時，成本必須計入所有已嘗試的工作，包括失敗 rewrite calls。

| Baseline | 分數與參數選擇 |
|---|---|
| B1：加權硬門檻 | 訊號加權和；0.25 lattice 上的 35 組 weights；validation threshold search |
| B2：Crisp ordinal | 以 0.35/0.65 切分輸入；總和 0–8；45 組整數 cut points |
| B3：固定模糊寬度 | 所有 widths 為 0.20 的 Mamdani score |
| B4：校準模糊寬度 | 使用 training-calibrated per-signal widths 的 Mamdani score |

升級分數 $e<\theta_1$ 選 A0；$\theta_1\leq e<\theta_2$ 選 A1；其餘選 A2。目前連續分數的搜尋範圍為 $\theta_1\in[0.20,0.50]$、$\theta_2\in[0.50,0.80]$、step 0.01，共 960 組有效配對。

參數選擇在平均成本限制下最大化 cached F1。預設 budget 為 always-A1 latency，但受限的 threshold grid 可能無法滿足此 budget，程式會回報錯誤。`pareto_frontier` 目前回傳各 budget 的最佳品質 sweep，可能重複同一 policy，並非已去重的嚴格 nondominated frontier。

## 已驗證結果與證據範圍

以下檢查於 **2026-10-08**，針對 source commit [`65d5cc4`](https://github.com/rexw713ss/Fuzzy-RAG/commit/65d5cc49eb0a287720cd393c252ecc6fda06d54b) 重新執行。

### Unit tests

驗證環境：Linux、Python 3.12.14、NumPy 2.3.5、pytest 9.1.1、regex 2026.9.29、PyYAML 6.0.3。這是輕量模組驗證，不是重新安裝原本記錄的 Windows GPU 環境。

| 結果 | 數量 | 說明 |
|---|---:|---|
| Passed | 251 | 已實作模組，以及合成或手動建立的測試資料 |
| Failed | 1 | `test_data_paths_match_the_scripts`：Windows 反斜線與正斜線在 POSIX `Path` 的比較結果不同 |
| Skipped | 1 | `test_rule_base_matches_csv`：缺少外部 `Fuzzy_RAG_personal/etc/rule_base_81.csv` |

內部產生的 81-rule count 與 31/19/31 分布另有測試，均通過。以上不代表大型外部 indexes 或 generated-answer quality 已獲驗證。

### 控制器單調性診斷

設定：主要 Mamdani controller、所有 widths 為 0.20、centroid output grid 為 1,001 點、violation tolerance 為 $10^{-9}$。每組比較只將一個訊號增加一個網格步長。比例的分母是相鄰比較次數，不是實際 queries 數量。

| 輸入間距 | 網格點數 | 比較次數 | 分數下降 violations | 比例 | 最大下降 | 跨越任一候選 threshold 的 violations |
|---|---:|---:|---:|---:|---:|---:|
| 0.20 | 1,296 | 4,320 | 64 | 1.48% | 0.006720 | 0 |
| 0.10 | 14,641 | 53,240 | 1,584 | 2.98% | 0.006720 | 0 |
| 0.05 | 194,481 | 740,880 | 44,800 | 6.05% | 0.037344 | 7,072 |

間距 0.05 時，每個訊號各有 11,200 個 violations；平均下降 0.014994，P95 為 0.029791。61 個候選 threshold 值中，有 47 個在此網格上不會造成降級。示範配對 $(0.30,0.60)$ 會造成 484 次 action 降級。**此配對不是使用真實 validation 資料選出的設定。**

例如，固定其餘訊號為 $(0.40,0.70,0.75)$，只將 S1 從 0.30 提高至 0.35，score 會由 0.612179 降至 0.574834；在 $\theta_2=0.60$ 時，action 從 A2 降為 A1。

Ordinal consequents 本身具單調性，但 defuzzified surface 並非全域單調。粗網格的零 action flips 不能保證細網格或真實 paraphrases 的穩定性。校準後的 widths 與 validation-selected thresholds 都需要重新診斷。

### 尚未量測的結果

目前沒有已提交的 benchmark outputs 能證明 Natural Questions 或 HotpotQA 上的 EM/F1 提升、latency 節省、calibrated fuzzy 優勢或 paraphrase robustness。已有 retrieval smoke scripts，但尚未公開資料集結果檔。Baseline 與 evaluation tests 使用合成 cached outcomes。

## 安裝

先 clone repository，以下指令均從專案根目錄執行：

~~~bash
git clone https://github.com/rexw713ss/Fuzzy-RAG.git
cd Fuzzy-RAG
~~~

### 輕量模組檢查

不需要 PyTorch、Java、模型下載或大型 passage indexes：

~~~bash
python -m venv .venv-core
source .venv-core/bin/activate
python -m pip install numpy==2.3.5 pytest==9.1.1 regex==2026.9.29 PyYAML==6.0.3
python -m pytest -q -rs
~~~

對齊上述驗證環境請使用 Python 3.12。Windows PowerShell 的 activation 指令為 `.venv-core\Scripts\Activate.ps1`。前述平台路徑失敗與缺少 CSV 仍是已知限制。

### 已記錄的檢索開發環境

Tracked requirement files 是開發環境快照：

| 環境 | 已記錄 Python | Dependency file |
|---|---|---|
| Main/dense | 3.14.4 | `requirements.txt` |
| BM25 | 3.13.0 | `requirements-bm25.txt` |

Windows 安裝範例：

~~~powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

py -3.13 -m venv .venv-bm25
.\.venv-bm25\Scripts\python.exe -m pip install -r requirements-bm25.txt
~~~

Main snapshot 指定從 CUDA 12.6 package index 安裝 `torch==2.14.1+cu126`；BM25 snapshot 包含 Windows-specific packages。這次未獨立重裝兩個完整環境，不應視為可跨平台使用的 lock files。檢索前請確認 Python、PyTorch/CUDA 與 Java 相容性；正式實驗中更換版本時，必須記錄。

Pyserini 需要相容的 JDK 與 `JAVA_HOME`。既有開發紀錄使用 Microsoft OpenJDK 25.0.4，實驗時應記錄實際 Java 版本。Windows console 若有編碼問題，可設定 `$env:PYTHONUTF8 = "1"`。

## 資料與路徑設定

大型資料與 indexes 由 `.gitignore` 排除。既有 repository 記錄：

| 資源 | 來源 | 已記錄範圍 |
|---|---|---|
| DPR passages | [psgs_w100.tsv.gz](https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz) | 21,015,324 passages |
| Contriever-MSMARCO embeddings | [wikipedia_embeddings.tar](https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar) | 16 個 pickle shards |
| BM25 index | [Lucene DPR index](https://rgw.cs.uwaterloo.ca/pyserini/indexes/lucene/lucene-inverted.wikipedia-dpr-100w.20260508.deb4c7b.tar) | 相同 passage-ID space |
| NQ-open dev | [NQ-open.dev.jsonl](https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.dev.jsonl) | 3,610 題；僅供 smoke checks |
| HotpotQA 問題 | Hugging Face `hotpotqa/hotpot_qa`，轉回原始 JSON 格式 | train 90,447 題；dev distractor 與 dev fullwiki 各 7,405 題 |
| HotpotQA BM25 index | [Lucene BEIR HotpotQA index](https://huggingface.co/datasets/castorini/prebuilt-indexes-beir/resolve/main/lucene-inverted/flat/lucene-inverted.beir-v1.0.0-hotpotqa.flat.20221116.505594.tar.gz) | 5,233,329 篇 abstracts |
| HotpotQA 稠密 index | [FAISS flat，contriever-msmarco](https://rgw.cs.uwaterloo.ca/pyserini/indexes/faiss/faiss-flat.beir-v1.0.0-hotpotqa.contriever-msmarco.20230124.tar.gz) | 相同 abstracts；IDs 記錄於其 `docid` 檔 |

<details>
<summary>既有檔案大小與 checksums 紀錄</summary>

保留自既有 README；這次未重新下載或重新計算雜湊。

| 資源 | 已記錄 bytes | 已記錄 checksum |
|---|---:|---|
| DPR passages | 4,694,541,059 | SHA-256：`c39b020c855a2b5c25ffef3abe4a3b6f9b829ad7dbc14ec3d163d34d7c53ea8d` |
| Embedding archive | 32,499,691,520 | 未記錄官方 checksum；可執行 embedding verification |
| BM25 archive | 11,078,635,520 | MD5：`1ef94a97f2ac418577d1e6a9ecf44806` |
| NQ-open dev | 未記錄 | SHA-256：`f15567f38099f3615f5b8a685c0aef449c11ad90d3da3735e8d1b98115b40616` |
| HotpotQA BM25 archive | 2,019,088,696 | MD5：`3f41d640a8ebbcad4f598140750c24f8` |
| HotpotQA 稠密 archive | 14,889,518,959 | MD5：`38c37708f9927501ca2f7563aa43f407`；已用 `verify_embeddings --dataset hotpotqa` 驗證 |

</details>

Loader 使用可能執行程式碼的 `pickle`，僅應讀取可信任的 release。`verify_embeddings` 檢查抽樣 embeddings 的方向一致性並回報 norm ratios，不是密碼學完整性驗證。請使用對應的 `contriever-msmarco` archive，不要混用 unsupervised `contriever` archive。

將 archives 解壓至資料目錄：

~~~bash
mkdir -p bm25 contriever-msmarco hotpotqa
tar -xf lucene-inverted.wikipedia-dpr-100w.20260508.deb4c7b.tar -C bm25
tar -xf wikipedia_embeddings.tar -C contriever-msmarco
tar -xzf lucene-inverted.beir-v1.0.0-hotpotqa.flat.20221116.505594.tar.gz -C hotpotqa
tar -xzf faiss-flat.beir-v1.0.0-hotpotqa.contriever-msmarco.20230124.tar.gz -C hotpotqa
~~~

HotpotQA 的 abstract 文字只存在 BM25 index 內。請在 BM25 環境中執行一次匯出，轉成主要環境可讀取的 DPR 格式 TSV：

~~~powershell
.\.venv-bm25\Scripts\python.exe -m scripts.export_corpus "data/hotpotqa/lucene-inverted.beir-v1.0.0-hotpotqa.flat.20221116.505594" "data/hotpotqa/corpus.tsv.gz"
~~~

所有 script 路徑集中在 `scripts/datasets.py`（每個資料集的 corpus、稠密 index 與 BM25 index），`tests/test_config.py` 會檢查它與 `config.yaml` 的 `data` 區段是否一致。資料搬移時兩者都要更新；只改 YAML 不會重新導向 scripts 的檔案讀取路徑。Corpus/index IDs 必須一致；請準備約 32 GB embeddings 所需 RAM 及額外搜尋 workspace。

## 執行目前提供的 scripts

以下使用 PowerShell 與各自環境的 interpreter，請將輸入路徑改為本機資料位置。

### BM25 檢索

輸入為 NQ-open 的 JSON Lines 檔或 HotpotQA 的 `.json` 檔。HotpotQA 請加上 `--dataset hotpotqa`（預設為 `nq`）；smoke test 與 embedding 檢查也使用相同選項。輸出保持順序，記錄 `qid`、`question` 與 hits。

~~~powershell
.\.venv-bm25\Scripts\python.exe -m scripts.bm25_search "path/to/NQ-open.dev.jsonl" "results/smoke/bm25.jsonl" --n 100 --k 50
~~~

### Hybrid retrieval smoke test

使用相同 question file，需包含 `question` 與 `answer`（NQ：gold-answer 字串列表；HotpotQA：單一字串）。HotpotQA 的 yes/no 答案無法以字串比對在 passage 中找到，因此這些問題不列入 Recall@k，另行計數。

~~~powershell
.\.venv\Scripts\python.exe -m scripts.smoke_retrieval "results/smoke/bm25.jsonl" "path/to/NQ-open.dev.jsonl" "results/smoke"
~~~

輸出：

- `retrieval.jsonl`：sparse、dense、fused rankings、原始 S1–S4，以及 answer-hit flags。
- `report.json`：包含答案的 passage 之 Recall@k、訊號摘要、correlations 與 timing。

Recall 使用 DPR-style token-sequence answer matching，不是生成答案正確率或 multi-hop supporting-fact recall。單一措辭的 smoke checks 無法證明 paraphrase robustness。

### Embedding 檢查

~~~powershell
.\.venv\Scripts\python.exe -m scripts.verify_embeddings --n 1000 --seed 0 --out "results/checks/verify_embeddings.json"
.\.venv\Scripts\python.exe -m scripts.verify_embeddings --dataset hotpotqa --n 1000 --seed 0
~~~

### 單調性報告

不需要檢索資料或語言模型，但需先設定輸出路徑。

~~~powershell
.\.venv-core\Scripts\python.exe -m scripts.monotonicity_report --steps 5 10 20 --theta1 0.30 --theta2 0.60
~~~

報告寫入 `scripts/datasets.py` 中 `DATA` 目錄下的 `checks/monotonicity.json`。上述 thresholds 僅為示範，不是擬合結果。

## 資料切分、評估與可重現性

- `splits.py` 對每個 dataset 分別以 seed 42 切分 clusters，比例約 60/20/20。同一問題的三種措辭必須位於相同 split，並保存實際 split IDs。
- P5/P95 scalers 與 jitter widths 僅使用 training；weights、cuts、thresholds 僅使用 validation。Test 資料保持鎖定。
- `OutcomeTable` 要求每個 `(cluster_id, variant_id)` 都有 A0/A1/A2 cached F1 與 cost。合併 datasets 時，cluster ID 應包含 dataset，避免碰撞。
- 目前 oracle 依固定順序 A0 < A1 < A2，選擇 F1 距離該 instance 最佳值不超過 $\delta=0.02$ 的第一個 action；它**不是**依 measured latency 找最便宜者。
- Routing flip rate（RFR）比較每個完整 cluster 的三組配對。Harmful routing flip rate（HRFR）使用相同的全部配對分母與 counterfactual F1 條件，因此 HRFR 不高於 RFR。不完整 clusters 會被跳過並計數。
- Within-cluster 指標包含 escalation-score SD、routed-F1 SD 與 best-minus-worst F1。
- Confidence intervals 與 paired comparisons 使用 cluster bootstrap：1,000 次 resamples、percentile 95% intervals、seed 42。
- `config.yaml` 記錄已實作 constants，測試檢查其一致性。`fitted` 欄位目前仍為 null；未來 pipeline 須保存回傳的 scalers、widths、thresholds、configuration 與 split IDs。
- 記錄 commit、model revisions、prompts、seeds、hardware、環境版本及各階段成本。成本需包含線上路由 overhead 與所有嘗試過的 action 工作。
- Exact batched-search timing 不等於逐題線上 latency。正式 protocol 固定前，smoke timings 僅視為診斷。

## Repository 導覽

| 位置 | 用途 |
|---|---|
| `main_logic/signals.py`、`dense.py`、`bm25.py`、`corpus.py`、`answers.py` | 檢索、訊號、corpus 存取、答案比對 |
| `main_logic/fuzzy.py`、`calibration.py`、`baselines.py`、`routing.py` | 推論、校準、baseline 擬合、policy selection |
| `main_logic/actions.py` | A2 合併與 rewrite 可用性規則 |
| `main_logic/splits.py`、`evaluation.py`、`monotonicity.py` | 切分、cached-outcome 指標、診斷 |
| `main_logic/config.py`、`config.yaml` | Configuration 載入與參數紀錄 |
| `scripts/` | Retrieval smoke、embedding、單調性檢查；`datasets.py`（所有資料路徑）；`export_corpus.py`（Lucene passages 轉 TSV） |
| `tests/` | Unit tests、合成評估、config 一致性 |
| `requirements*.txt` | 已記錄的開發 dependencies |

部分 source comments 引用的 specification v0.3 與外部 rule/provenance files 尚未包含於 repository。

## 後續工作

- [x] Sparse/dense retrieval 與 S1–S4 基礎函式
- [x] NQ 與 HotpotQA indexes（HotpotQA 使用預建 BEIR indexes，embeddings 已驗證）
- [x] 完整 fuzzy rule base 與 operator ablations
- [x] Cluster split 與 width-calibration 基礎函式
- [x] B1–B4 擬合與 cached-policy evaluation
- [x] Oracle、RFR/HRFR、within-cluster 與 bootstrap utilities
- [x] 單調性診斷與 unit tests
- [ ] 可跨平台、完全 config-driven 的路徑與環境
- [ ] 納入 repository 的 rule CSV、specification 與 provenance
- [ ] 已驗證的 paraphrase clusters 與 pilot signal analysis
- [ ] Cross-encoder、rewrite model、prompts 與 shared generator
- [ ] 完整 A0/A1/A2 action-outcome collection
- [ ] EM/token-F1、實際成本量測與 held-out comparisons
- [ ] 公開 outputs、confidence intervals 與論文表格

擴大 generation 前，先以 pilot 確認 A1/A2 對部分問題優於 A0。不能從 unit tests 或合成網格推論路由有效性。

## 引用與授權

目前尚未提供對應論文引用或軟體授權。可重現性報告請註明 repository URL 與精確 commit。重新散布或整合程式前，請先聯絡 owner；外部資料與模型權重仍依各自條款使用。
