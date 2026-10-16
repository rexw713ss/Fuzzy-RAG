# 後續研究執行工具

這裡承接「完成後續研究並推到experiment_ver2」的任務。[固定程序與完成條件](PROTOCOL.zh-TW.md)列出完整範圍；**目前尚無新action結果、未曝光確認集結果或真人語意審核完成證據**。工具通過測試不等於研究完成。

## 已準備的真實資料

- [prepared](prepared/)：兩資料集各34個完整TRAIN clusters、102個queries的action pilot；原fuzzy/ridge的frozen policy與scaler；每資料集450個已曝露原題的排除盤點。資料均來自repo真實cache。
- [human_review](human_review/)：同60個blind pairs的兩份空白獨立評分表。A／B只是兩個檔案的角色，不能由同一人或AI填成兩位真人。現在沒有任何人工分數。
- [evidence/readiness.json](evidence/readiness.json)：本機實際資產檢查。缺模型／索引等時明確顯示blocked；不會回退到合成資料或舊答案。
- [evidence/planning_final/report.json](evidence/planning_final/report.json)：已曝光資料的樣本數模擬，固定quota重配置及多重比較後，主規劃為NQ 1,800／HotpotQA 3,500個新原題。這是規劃估計，沒有新增模型生成結果。

## 人工審核

請把`rater_A.jsonl`和`rater_B.jsonl`分別交給兩位獨立真人。每列填入：

- `rater_id`：同一份表固定使用同一個姓名或研究代號。
- `status`：完成為`reviewed`，無法判定為`abstained`；未做保持`pending`。
- 三個判準填JSON布林值`true/false`；無法判定時保持`null`並寫原因。
- `notes`：任何不等義／新增刪除限制／不可作答判定必須說明。
- `attestations`：真人、獨立完成且未見outcomes時，三項才填`true`。

不要修改題目或加入gold答案、router名稱、F1。不要把審核對照key給審核者。匯入工具保留原始檔及hash；分歧或abstention需第三位不同真人裁決。

```powershell
.\.venv-core\Scripts\python.exe -m experiment_ver2.followup.review audit --tasks experiment_ver2/robustness/evidence/blind_paraphrase_review.jsonl --left path/to/completed_A.jsonl --right path/to/completed_B.jsonl --out experiment_ver2/runs/human-audit
```

若需裁決，再用新的output目錄加`--adjudications path/to/adjudications.jsonl`。每列必須含`review_id,rater_id,reason,human_review,outcome_blind`及相同三個布林判準。審核未完成時exit code為2，不會寫成成功。這60對沒有涵蓋全部variants，不可用來自動剔除未審查的題目。

## 新題抽樣

先取得原來源題庫。官方來源可從[EfficientQA資料說明](https://efficientqa.github.io/getting_started.html)與[HotpotQA官方資料說明](https://github.com/hotpotqa/hotpot)找到；下載後必須與已曝露450題逐一核對題序與原文。工具會保存下載來源URL及實際檔案SHA256；URL本身不等於原版本已驗證。

```powershell
.\.venv-core\Scripts\python.exe -m experiment_ver2.followup.sampling --archive runs --dataset nq --pool path/to/NQ-open.dev.jsonl --source-url https://raw.githubusercontent.com/efficientqa/nq-open/master/NQ-open.dev.jsonl --protocol experiment_ver2/followup/PROTOCOL.zh-TW.md --n 1800 --out experiment_ver2/runs/new-nq
.\.venv-core\Scripts\python.exe -m experiment_ver2.followup.sampling --archive runs --dataset hotpotqa --pool path/to/hotpot_dev_fullwiki_v1.json --source-url http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json --protocol experiment_ver2/followup/PROTOCOL.zh-TW.md --n 3500 --out experiment_ver2/runs/new-hotpotqa
```

這一步只封存新原題，未生成改寫、人工審核或outcomes。不能直接把它稱為完成的確認實驗。若另有未推上repo的試跑／pilot題目，需先補進排除紀錄。

## 新action執行

複製[runtime.example.json](runtime.example.json)到本機實際設定檔，填入原模型、llama-server、CPU模型cache、dense／BM25 index、BM25 Python／Java及hash。舊設定路徑只是查找線索，不代表檔案存在。需要torch/transformers的模型runtime、另一個可使用pyserini／Java的BM25 runtime；不要把歷史`requirements.txt`的版本無條件灌進已驗證的`.venv-core`。

`generator_identity`填入原實驗人提供的model SHA256及其來源證據、下載來源與固定revision；不能把自己計算的hash直接宣稱為歷史原模型。工具檢查Llama 8B Q4_K_M的GGUF基本metadata、declared llama.cpp b11505與本機檔案hash，但無法替你驗證來源聲明。HF模型使用原始碼中的固定revision，僅讀本機cache並記錄檔案hash。

每個index需要放在index目錄**外**的manifest，config `index_manifests`再記錄manifest路徑與SHA256。manifest包含`schema:1`、`dataset`、`kind:dense/bm25`、`format`、`source:{url,revision}`、`corpus_sha256`及`files:{相對檔名:SHA256}`。format分別為NQ `contriever-pickle-shards`、HotpotQA `faiss-flat-ip`、BM25 `lucene-bm25`；dense另需`encoder:{model_id,revision}`匹配固定Contriever。NQ另需可信來源的`trusted_pickle_source:true`，不能把未知pickle隨意標記可信。逐檔驗證且不允許遺失／額外檔或越出root；hash建立一致性，不會證明任意manifest真實。建立SQLite後填入它本身的`passage_store_sha256`，與corpus hash分開。

先把原corpus建立為SQLite passage lookup，避免每題掃描數百萬段落；資料建置不列入per-query latency，但來源hash與狀態要保留：

```powershell
.\.venv-core\Scripts\python.exe -m experiment_ver2.followup.passages --corpus path/to/psgs_w100.tsv.gz --sha256 VERIFIED_CORPUS_HASH --out experiment_ver2/runs/nq_passages.sqlite
python -m experiment_ver2.followup.runtime --config path/to/runtime.json --out experiment_ver2/runs/preflight-new.json
python -m experiment_ver2.followup.collect --queries experiment_ver2/followup/prepared/nq/pilot_queries.jsonl --config path/to/runtime.json --protocol experiment_ver2/followup/PROTOCOL.zh-TW.md --out experiment_ver2/runs/action-pilot-nq
```

`collect`中的`python`必須是已安裝torch/transformers與NumPy的模型環境，不是強制用`.venv-core`。HotpotQA使用自己的index/corpus/runtime設定及prepared query檔。CPU encoder／reranker與GPU generator同時常駐需要足夠RAM；NQ dense shards約32GB、Hotpot dense約16GB，詳見原`main_logic/dense.py`。這裡不會自動下載大型索引或模型。

每個action重新執行、保存prompt/context/raw答案、signals、counter及wall-clock分量。新動作名稱固定`A2-union`。相同run中斷可加`--resume`，但input/code/protocol/config任何fingerprint改變都必須建立新run。不要因低F1或慢而刪除失敗attempt；journal保留完整紀錄。若有lock，先確認PID是否仍在執行，不能只因等候逾時就刪除重開。

全部planned actions完成後才執行：

```powershell
.\.venv-core\Scripts\python.exe -m experiment_ver2.followup.evaluate --queries experiment_ver2/followup/prepared/nq/pilot_queries.jsonl --run experiment_ver2/runs/action-pilot-nq --out experiment_ver2/runs/action-pilot-nq-report
```

評估會重算答案EM/F1、檢查完整3-action矩陣與來源fingerprints，不接受歷史cache替代新action。pilot只有TRAIN題，不能拿pilot正結果直接宣稱確認性成功。原A1／A2的確認性policy評估與新A2-union的router訓練是兩條研究問題，不能交換outcomes。

## 新確認集的執行順序

以下以NQ示範，HotpotQA必須使用其自己的sample/runtime/policy。`paraphrases`只消費凍結sample，不會重新抽樣；保留全部原題與不足2個改寫的clusters，候選、NLI／cosine、排除理由及attempt journal均存檔。需要額外的固定revision DeBERTa與mpnet本機cache。

```powershell
python -m experiment_ver2.followup.paraphrases --config path/to/runtime.json --preflight-only
python -m experiment_ver2.followup.paraphrases --sample experiment_ver2/runs/new-nq --protocol experiment_ver2/followup/PROTOCOL.zh-TW.md --config path/to/runtime.json --out experiment_ver2/runs/new-nq-paraphrases
.\.venv-core\Scripts\python.exe -m experiment_ver2.followup.review export --tasks experiment_ver2/runs/new-nq-paraphrases/blind_review_tasks.jsonl --out experiment_ver2/runs/new-nq-blank-review
```

把這一批兩份空白表交兩位真人；既有60對審核不能代替新題審核。收回完整評分與必要裁決後執行`review audit`，以新的tasks檔及新的output保存原始評分。再執行：

```powershell
.\.venv-core\Scripts\python.exe -m experiment_ver2.followup.freeze --sample experiment_ver2/runs/new-nq --paraphrases experiment_ver2/runs/new-nq-paraphrases --lock experiment_ver2/followup/prepared/nq/policy_lock.json --protocol experiment_ver2/followup/PROTOCOL.zh-TW.md --human-report experiment_ver2/runs/new-nq-review/report.json --out experiment_ver2/runs/new-nq-frozen
python -m experiment_ver2.followup.collect --queries experiment_ver2/runs/new-nq-paraphrases/queries.jsonl --config path/to/runtime.json --protocol experiment_ver2/followup/PROTOCOL.zh-TW.md --confirmation-bundle experiment_ver2/runs/new-nq-frozen --out experiment_ver2/runs/new-nq-actions
.\.venv-core\Scripts\python.exe -m experiment_ver2.followup.confirmation --queries experiment_ver2/runs/new-nq-paraphrases/queries.jsonl --run experiment_ver2/runs/new-nq-actions --protocol experiment_ver2/followup/PROTOCOL.zh-TW.md --confirmation-bundle experiment_ver2/runs/new-nq-frozen --out experiment_ver2/runs/new-nq-confirmation
```

freeze會重新核對人審原始評分、題文、sample／exposure／policy／protocol hashes；collect在開始前把binding hash寫進run identity。完成後confirmation拒絕缺資料、錯誤seed、人審不完整、錯題prompt、空trace或矛盾成本。這些檢查無法證明真人身分、隱藏曝露或外部檔案來源真實性；研究紀錄仍需原實驗人確認。

主比較只在「全部原題＋人審有效改寫」配置固定q=.25，再做10,000次cluster bootstrap重配quota，兩資料集共2個主比較。無效改寫及A2-union仍保留於原始矩陣，全集另作描述性敏感度報告。不能把次要未校正區間或pilot換成主成功指標。

## 還需要原實驗人提供

1. 兩個原始題庫、retrieval corpus、dense與BM25 indices；若有未入repo的試跑題目，也要提供曝光清單。
2. 原generator GGUF、llama.cpp binary、HF pinned snapshots及來源revision／hash；可直接提供原機器的可用路徑或可存取的封存。
3. 兩位真人完成審核，遇分歧再由第三人裁決。這不是可由測試或AI評分代替的項目。

目前資產缺失阻止端到端重跑；本地準備及測試不等於上述三項已完成。完整待辦見[STATUS](STATUS.zh-TW.md)。
