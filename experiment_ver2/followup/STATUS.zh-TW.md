# 後續研究交付狀態（2026-10-16）

**研究尚未全部完成；本輪完成的是可執行流程、規劃模擬與證據檢查。** 新模型生成、人審及新資料確認結果仍缺，不能宣稱新的方法已證明優於ridge。

| 項目 | 實際證據與狀態 |
|---|---|
| 既有資料再實驗 | 前一輪additive與穩健性分析已完成；結果在上層報告／evidence，不把本輪測試算成另一批獨立資料 |
| 新A2設計 | A2-union把兩次候選合併、去重，再用原問題rerank；保留全部prompt/context/raw answer、額外CE與生成成本 |
| 新action pilot | 已固定每資料集34個TRAIN clusters、102題；共612個action outcomes **尚未生成** |
| 原政策鎖定 | fuzzy／ridge政策與前版完全一致，附TRAIN來源與protocol／scaler／policy hashes；測試題不擬合模型 |
| 新題抽樣 | 已排除每資料集450個已曝光原題及全部可見variants；實際來源題庫缺失，尚未抽取新樣本 |
| 樣本數 | 已跑真實cache的規劃模擬；新規劃NQ 1,800、HotpotQA 3,500個原題，不能把模擬當新生成答案 |
| 人工審核 | 既有60對的兩份表均未填；新資料需審全部保留pairs。沒有AI代簽，也沒有把空白補成有效 |
| 確認性分析 | 已實作完整指紋綁定與paired cluster推論，**沒有新確認結果** |
| 環境可用性 | 實際preflight失敗：原D:\Han\rex_rag資產、必要模型套件／snapshots、indices與passage store缺失 |
| Git交付 | 本輪工作限experiment_ver2分支與目錄；上一次已推送的commit為33984dc。本輪新增後續流程尚未推送，不能報成遠端已完成 |

整套測試 **702 passed、2 skipped**；兩項skip是原repo缺少的外部rule CSV。具體檢查見[readiness.json](evidence/readiness.json)、[測試原始輸出](evidence/tests.txt)、[固定程序](PROTOCOL.zh-TW.md)。初版prepared及規劃證據保留在`evidence/prior_preparation`、`planning`、`planning_v2`；目前使用`prepared`與`planning_final`，不要混用protocol hash。

## 這一輪改善了什麼

人工審核現在必須逐題核對左右原文、唯一review ID及hash，並從保存的兩份評分重算；不能把一個通過決策套到多個改寫。主分析使用「全部原題＋人審有效改寫」，在人審結果凍結之後才收集action outcomes。無效改寫的原gold可能不再正確，因此其F1只留作整條流程的敏感度分析。

收集端保留固定排程、失敗attempt和續跑原因，評估端核對整個三action矩陣與durable journal。空trace、錯題prompt、錯誤context、token／時間矛盾均拒絕；生成參數固定temperature0、seed42與原answer/followup token限制。模型、索引及passage store驗證在初始化執行，不能把初始化或JVM成本隱藏成加速收益。這些是紀錄一致性檢查，不是外部來源真實性的證明。

初版600／1,300題的規劃，在多重比較及重配quota後，對假設1個F1百分點差異的偵測率只有約0.68／0.77。提高至1,800／3,500後，在1.5倍SD情境約0.821／0.827。這些數字是曝光分布上的置中location-shift模擬；人審後有效群體、模型生成差異及資料移轉尚未觀察，不能保證實際power。

## 研究下一步的判準

目前最直接的改進是先確認A2能否保留有用證據且產生實質答案收益。A2-union pilot若無改善，應如實報告，不能靠router選擇把失敗action包裝成成功；若改善，才收集新action的完整TRAIN／VAL矩陣，重新擬合router並另鎖程序。原A2政策不能直接拿來代表新A2-union的確認結果。

81條rule相對180個TRAIN clusters仍然稀疏，既有研究顯示大量估計受global prior支配。additive較簡潔，但HotpotQA對ridge的差值區間仍含0、NQ未改善；其較高均值不足以證明fuzzy的必要性。論文應保留同訊號ridge、相同實際A2數量參考、完整負結果及改寫傷害率，再以新確認資料檢驗主張。

## 解除阻礙所需資料

請原實驗人提供可存取的原機器／封存路徑，包含原始NQ／HotpotQA題庫、corpus、dense／BM25 indices、generator GGUF、llama.cpp binary及固定revision的HF模型；另附可信來源與hash、未入repo的額外曝露題目。需要兩位真人完成盲審；分歧需第三人裁決。可直接使用[README](README.zh-TW.md)中的命令在具備資產的機器執行，所有失敗與不顯著結果都須保留。

先前自動核准審查因用量限制拒絕Git fetch／同步；本輪未改走其他連線繞過限制。環境資產、人審與同步限制解除前，不能把「全部完成並推送」標記為完成。
