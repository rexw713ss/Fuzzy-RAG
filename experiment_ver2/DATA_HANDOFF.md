# 第一輪實驗資料交接

第二版需要重評第一輪的逐題快取。**不用重跑模型，也不用上傳 Wikipedia corpus、dense embeddings、Lucene／FAISS index 或 GGUF。**

必需檔案：

```text
full/splits.json
full/nq/queries.jsonl
full/nq/signals.jsonl
full/nq/outcomes.jsonl
full/hotpotqa/queries.jsonl
full/hotpotqa/signals.jsonl
full/hotpotqa/outcomes.jsonl
```

請保留原始 IDs、資料分割與數值，`signals.jsonl` 要原始 S1–S4。不要重新隨機分割，也不要只提供各方法平均值；paired bootstrap 與每題 action 效益需要逐題完整 A0/A1/A2 矩陣。

若方便，另外提供各資料集 `clusters.jsonl`、`results/{nq,hotpotqa,pooled}.json`、`evaluate.log`、`config.yaml`、原始 commit、模型 revisions、環境版本、人工 paraphrase 審核狀態及重跑紀錄。這些能核對來源與實驗條件。

可以先下載／切換本 repo 的 `experiment_ver2` 分支，再從 repo 根目錄執行：

```powershell
python -m pip install -r experiment_ver2/requirements.txt
python -m experiment_ver2.package_cache --root "D:/Han/rex_rag/data/full" --out "first_run_cache.zip"
```

工具先驗證必需檔案，將最小檔案、可選的 clusters 與各資料集報告打包，並附每個檔案的 SHA256；不修改原 cache。執行失敗會指出缺檔或資料不一致，請保留原資料並提供錯誤訊息。

請把 ZIP 放到 GitHub Release asset 或另開的資料分支，提供可下載連結。原 `main` 無須變更。本工具不會自動上傳或發送訊息。

收到後的第一步：

```powershell
python -m experiment_ver2.run --root "path/to/extracted/full" --out experiment_ver2/runs/cache-audit --check-only
```

ZIP 內頂層直接是 `splits.json` 和資料集資料夾，所以 `--root` 應指向解壓後包含 `splits.json` 的位置。
