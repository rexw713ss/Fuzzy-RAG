# 第三輪探索實驗：減少 fuzzy 交互作用

這一輪回應「再做一次實驗」，使用 v0.4 的真實 A1／A2 cached outcomes 重新訓練、選模與評分。這不是重跑 retriever／generator，也不是新收集的測試集。既有 fresh 結果已曝光，所有新方法結果皆為探索性。

本文件與新增模型在讀取本輪 test 結果前寫定。動機來自已完成的結構診斷：180 個 training clusters 支撐 81 個 product rules，許多 rule 資料量小，主模型選擇 prior=20。**本輪只檢查減少交互作用是否有改善；不保證結果變好，也不根據 test 再試其他超參數。**

## 固定設計

- 資料：`runs/v0.3` 的既定 TRAIN／VAL split、v0.4 action outcomes，以及 v0.4 fresh 的150個clusters／dataset。
- 維度：四訊號 S1–S4，兩種 action A1／A2；兩資料集 NQ、HotpotQA 均完整報告。
- 新方法：每訊號三個 Low／Med／High membership，共12個基底，僅相加，不組合成81個交互作用規則。以加權 ridge 學習兩個 action 的品質與成本；決策只依預測 `F1(A2)-F1(A1)`。
- 兩個方法族分別使用固定 width=.2 及原 TRAIN jitter 校準 width。它們都必須報告，不依 test 選贏家。calibrated 是本輪事先指定的新方法主列，fixed 是消融。
- 標準化：每個 membership 基底的均值／尺度只由 TRAIN 計算，每個原題 cluster 總權重1。沒有新參數可見 test label。截距不懲罰；alpha固定網格 `[.1,1,10]`。membership 存在線性相依，但正 ridge penalty 使解明確；不把12基底說成12個獨立自由度。
- VAL：與 ver2 相同，用 F1 選 alpha 及 always-A1 回退，平手順序為較少A2、較低cached cost、候選名稱。
- 配額：主結果固定最大 A2 fraction=.25，只有正預測改善才有資格。沒有新增 budget search。
- 比較：既有連續訊號 ridge、81-rule calibrated fuzzy、always-A1，以及相同實際A2次數的均勻 random expectation。兩個新族均以相同 pipeline 評估。
- 固定 seed=20261017，10,000次 cluster bootstrap。新方法−ridge、新方法−既有81-rule、新方法−random，共2族×2資料集×3比較=12個探索性比較。提供未校正95%與 Bonferroni percentile 敏感度區間；後者近似，不消除 test reuse 偏差。
- bootstrap 保持已選動作，用每次 resampled batch 的 A2 fraction 建立 random reference；不重訓，也不重新排序。本輪主列的區間與原ver2的固定random fraction區間定義不同，必須明列。
- 另報 original-only、complete-cluster-only 的 frozen-assignment 分數與選擇次數，這些是描述性敏感度，不新增宣稱成功的端點。
- 輸出所有 fitted candidates、VAL選擇表、逐題決策、source/input hashes。新檔案放 `experiment_ver2/additive_evidence/`，既有 evidence 不變。

## 如何解釋

若 additive 結果較差，保留負結果，將「較少交互作用」假設視為本資料不支持。若點估計較好但區間含0，寫證據不足。若有正探索性區間，仍需獨立未曝光題目確認。不能把模型族新增的搜尋自由度忽略，或只挑一個資料集報告。

此方法可輸出每訊號對預測改善量的分量；不宣稱它是因果解釋。相較原81-rule smoother，這同時改變了基底結構及估計方法，因此結果無法單獨歸因於規則數量；若要分離兩者，需事先規劃另一個估計器相同的 factorial 對照，而不是依本輪test追加。
