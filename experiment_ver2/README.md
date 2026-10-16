# FuzzyRoute-RAG experiment_ver2

這是獨立的第二版研究程式。開發分支為 `experiment_ver2`，所有新增檔案均在本目錄；`main` 與第一版程式保留。

**目前：實作與合成流程驗證完成；真實 NQ／HotpotQA 重評待第一輪 cache。** 本機沒有第一輪逐題資料。任何合成輸出都不是 benchmark 結果，也不能據此宣稱研究成功。舊 test 已被檢視，所以未來以原 cache 跑出的 ver2 結果仍屬探索性；確認性結果需要新的原始問題 clusters。

## 為何改方法

第一輪在 always-A1 時間上限下幾乎全部選 A1，因 rerank 相對 generation 幾乎免費；硬性要求多用 A0／A2 不會解決這件事。第二版把主問題改為：**哪些問題值得額外執行 A2？**

- 以 A1 為預設，估計 `F1(A2) - F1(A1)`，允許 rewrite 造成負效益。
- 使用 81 條 product fuzzy rules 學習 action 品質；資料少的 rule 向 training global mean 收縮。
- 比較固定／校準 fuzzy、同訊號 ridge、global、原 B3/B4 uncertainty 排序及相同實際 A2 次數的隨機參考。
- 固定最大 A2 配額 `[0, .10, .25, .50, 1]`，主配額 `.25`；不強迫花完配額。這是離線批次分配，不是線上 latency 保證。
- TRAIN 擬合 scaler、width 與模型；VAL 選超參數及是否回退 always-A1；TEST 只評分。不得挑 test 最好的 budget 或 baseline 當作事前設定。

完整問題定義、統計限制與下一輪收集程序見 [研究程序](RESEARCH_PROTOCOL.md)。

## 安裝與驗證

從 repository 根目錄執行，Python 3.12 已驗證。只需 CPU，不需模型、GPU、Java 或 Wikipedia indexes。

```powershell
py -3.12 -m venv .venv-core
.\.venv-core\Scripts\python.exe -m pip install -r experiment_ver2/requirements.txt
.\.venv-core\Scripts\python.exe -m pytest experiment_ver2/tests -q
.\.venv-core\Scripts\python.exe -m experiment_ver2.run --demo --out experiment_ver2/runs/synthetic-check
```

完整舊版測試另需 `regex`、`PyYAML`：

```powershell
.\.venv-core\Scripts\python.exe -m pip install regex PyYAML
.\.venv-core\Scripts\python.exe -m pytest -q -rs
```

Linux／macOS 可用 `python3 -m venv .venv-core` 與 `.venv-core/bin/python`。舊版 `tests/test_config.py` 的 Windows 路徑測試仍可能在 POSIX 失敗；此分支未修改第一版程式。

## 取得真實 cache 後執行

最小目錄結構（`signals.jsonl` 必須是原始 S1–S4，不能拿 scaled 檔替代）：

```text
full/
  splits.json
  nq/
    queries.jsonl
    signals.jsonl
    outcomes.jsonl
  hotpotqa/
    queries.jsonl
    signals.jsonl
    outcomes.jsonl
```

`splits.json` 是 `{dataset: {train: [cluster_id], val: [...], test: [...]}}`。每個 query 以 `(dataset, cluster_id, variant_id)` 對齊，outcomes 要有 A0/A1/A2 三列，包含 `F1`、`EM`、`latency_total`。loader 接受原版的 1–3 variants，不完整 cluster 保留於品質評估、從 paraphrase pair 指標排除並計數。

```powershell
# 請換成下載／解壓後的 full 路徑。out 必須是全新或空目錄。
.\.venv-core\Scripts\python.exe -m experiment_ver2.run --root "D:/cache/full" --out experiment_ver2/runs/input-audit --check-only
.\.venv-core\Scripts\python.exe -m experiment_ver2.run --root "D:/cache/full" --out experiment_ver2/runs/replay-001
```

可加 `--datasets nq` 先跑單一資料集。輸入只讀；輸出不能位於原 cache root 內；已存在的 run 不覆寫。缺檔／重複 key／跨 split／非法數值會明確失敗並輸出 `failure.json`，不會用合成資料代替。

每次執行保存：

| 產物 | 內容 |
|---|---|
| `manifest.json` | commit、source SHA256、環境、固定候選與配額 |
| `data_audit.json` | 輸入 hashes、split 數量、train scaler／width、缺 paraphrase 計數 |
| `<dataset>_models.json` | global／ridge／fuzzy 模型與 rule support |
| `<dataset>_decisions.jsonl` | TEST 每筆 frozen action、score；主配額 fuzzy 啟動規則 |
| `<dataset>_report.json` | 全配額 F1／EM／cost、RFR／HRFR、validation fits、paired cluster CI |
| `REPORT.md` | 主配額比較摘要與證據限制 |

隨機參考使用相同 A2 次數的期望表現，沒有用 test 品質優化混合比例。CI 固定已擬合模型、批次 assignment 與隨機參考比例，反映 cluster 抽樣不確定性；不是重新訓練或重排未來 batches 的總不確定性。latency 是舊 cache 的分量總和，沒有重新驗證線上速度；配額相同也不代表 latency 相同。

## 請原實驗人提供資料

可將 [資料交接說明](DATA_HANDOFF.md) 轉給原實驗人；`package_cache.py` 只打包小型逐題資料，不收模型或索引。不需要重新執行 A0/A1/A2 generation。

## 原始檔案

| 檔案 | 責任 |
|---|---|
| `data.py` | 嚴格讀取、train calibration、來源雜湊 |
| `models.py` | 可序列化 fuzzy/global/ridge action outcome 模型 |
| `policies.py` | 正效益排序、批次配額、validation 選擇 |
| `metrics.py` | 固定政策評估與 paired cluster bootstrap |
| `run.py` | 預先固定研究流程與結果保存 |
| `demo.py` | 明確標示的合成機制檢查 |
| `package_cache.py` | 最小 cache 交接工具 |
| `tests/` | 邊界、資料洩漏、配額與完整流程測試 |

真實 cache 未到齊、人工 paraphrase 稽核與 fresh test 未完成前，不宣稱 fuzzy 優於 baseline。若重新分析仍無收益，也會是應保存的有效負結果。
