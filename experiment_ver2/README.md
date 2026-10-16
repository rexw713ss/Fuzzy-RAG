# FuzzyRoute-RAG experiment_ver2

這是獨立的第二版研究程式。開發分支為 `experiment_ver2`，所有新增檔案均在本目錄；`main` 與第一版程式保留。

**目前：已整合 main 的 `667639a`，完成真實 NQ／HotpotQA cache 重評。** 原 v0.4 的 6,494 個報告數值完全重現，並完成 v0.3 舊測試、v0.4 舊測試、v0.4 fresh 四訊號主分析及三訊號消融。結果與限制見 [實驗報告](EXPERIMENT_REPORT.zh-TW.md)，完整輸出見 [evidence](evidence/)。所有分析均為探索性；名為 fresh 的資料已有公開結果，不能再視為本方法未見過的確認性測試。

**新增：已完成穩健性稽核與另一輪真實cache實驗。** 兩資料集各10,000次cluster bootstrap、100次TRAIN／VAL重新抽樣；另測12基底additive fuzzy。HotpotQA F1由原fuzzy的0.47497增至0.48458，但對ridge差值區間仍含0，且改寫HRFR較高；NQ未改善。請見 [完整穩健性稽核](robustness/REPORT.zh-TW.md)、[新實驗結果](ADDITIVE_EXPERIMENT_REPORT.zh-TW.md) 與 [預先寫定的新實驗程序](ADDITIVE_EXPERIMENT_PROTOCOL.md)。沒有新增模型API或GPU生成呼叫。

## 為何改方法

第一輪在 always-A1 時間上限下幾乎全部選 A1，因 rerank 相對 generation 幾乎免費；硬性要求多用 A0／A2 不會解決這件事。第二版把主問題改為：**哪些問題值得額外執行 A2？**

- 以 A1 為預設，估計 `F1(A2) - F1(A1)`，允許 rewrite 造成負效益。
- 使用四訊號 81 條 product fuzzy rules 學習 action 品質；資料少的 rule 向 training global mean 收縮。另固定評估 S2–S4 三訊號、27 規則消融。
- 比較固定／校準 fuzzy、同訊號 ridge、global、原 B3/B4 uncertainty 排序及相同實際 A2 次數的隨機參考。
- 固定最大 A2 配額 `[0, .10, .25, .50, 1]`，主配額 `.25`；不強迫花完配額。這是離線批次分配，不是線上 latency 保證。
- 原始資料模式以 TRAIN 擬合 scaler、width 與模型；封存模式沿用有 provenance 的 TRAIN scaler，從 TRAIN scaled signals 重算並核對 width。VAL 選超參數及是否回退 always-A1；TEST 只評分。

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

## 直接重評 repo 已提供的真實資料

以下四個設定都已執行，沒有新的 LLM 呼叫。`--archive` 是明確啟用的封存相容模式；因 v0.3 未上傳 raw training signals，不能宣稱從原始值獨立重驗過 scaler。

```powershell
# 四訊號主分析，v0.4 的 150 fresh clusters / dataset
.\.venv-core\Scripts\python.exe -m experiment_ver2.run --archive runs --version v0.4 --test-set fresh --out experiment_ver2/runs/replay-v04-fresh
# 三訊號消融，不能依 test 結果取代主方法
.\.venv-core\Scripts\python.exe -m experiment_ver2.run --archive runs --version v0.4 --test-set fresh --signals S2 S3 S4 --out experiment_ver2/runs/replay-v04-3sig
# 兩組歷史測試，各自保存
.\.venv-core\Scripts\python.exe -m experiment_ver2.run --archive runs --version v0.4 --test-set old_test --out experiment_ver2/runs/replay-v04-old
.\.venv-core\Scripts\python.exe -m experiment_ver2.run --archive runs --version v0.3 --test-set old_test --out experiment_ver2/runs/replay-v03-old
```

v0.4 矩陣只有 A1／A2，程式不補造 A0；v0.3 保留 A0 固定參考。所有模型按實際維度學習。

原 v0.4 數值與資料完整性驗證：

```powershell
.\.venv-core\Scripts\python.exe -m experiment_ver2.reproduction --full runs/v0.3 --v4 runs/v0.4 --out experiment_ver2/runs/reproduction-check.json
```

繪圖需額外安裝 report dependencies：

```powershell
.\.venv-core\Scripts\python.exe -m pip install -r experiment_ver2/requirements-report.txt
.\.venv-core\Scripts\python.exe -m experiment_ver2.plot_results --root experiment_ver2/runs/replay-v04-fresh
```

## 原始未縮放 cache 模式

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

目前 repo 的封存資料足以重評，不必再等交接。若要獨立驗證原 scaler 或新增實驗，可參考 [資料交接說明](DATA_HANDOFF.md)；`package_cache.py` 專用於原 v0.3 三動作、未縮放 cache，不收模型或索引。

## 原始檔案

| 檔案 | 責任 |
|---|---|
| `data.py` | 嚴格讀取、train calibration、來源雜湊 |
| `archive.py` | 明確啟用的 v0.3／v0.4 封存資料相容與 freshness 檢查 |
| `models.py` | 可序列化 fuzzy/global/ridge action outcome 模型 |
| `policies.py` | 正效益排序、批次配額、validation 選擇 |
| `metrics.py` | 固定政策評估與 paired cluster bootstrap |
| `run.py` | 預先固定研究流程與結果保存 |
| `demo.py` | 明確標示的合成機制檢查 |
| `package_cache.py` | 最小 cache 交接工具 |
| `tests/` | 邊界、資料洩漏、配額與完整流程測試 |
| `reproduction.py` | 原 v0.4 重現與逐題答案／成本／重複資料檢查 |
| `plot_results.py` | 主比較 CI 及所有固定配額圖表 |

HotpotQA 的效益路由有正向探索結果，但 fuzzy 沒有勝過 ridge；NQ 未明確勝過同 A2 次數隨機策略。人工 paraphrase 稽核與新的確認性 test 尚未完成，不宣稱 fuzzy 特有優勢。
