# 新一輪實驗結果：additive fuzzy

已依事先寫定的 [實驗程序](ADDITIVE_EXPERIMENT_PROTOCOL.md)，使用真實 v0.4 cached outcomes 完成兩資料集重新訓練、VAL選模及10,000次cluster bootstrap。**HotpotQA的F1點估計提高，但尚無足夠證據證明優於ridge或原81-rule fuzzy；NQ相對ridge與原fuzzy的F1點估計未提高。** 這是探索性離線實驗，未新增retrieval／generation outcomes，沒有新的未曝光test。

新模型使用12個單訊號Low／Med／High membership基底，以正則化線性模型學習每個action品質，不建81個交互作用規則。fixed與calibrated兩族都只在VAL選alpha，兩資料集均選0.1；沒有依test改網格或重新挑模型。基底的TRAIN標準化與樣本權重均按原題cluster處理。

| 資料集 | 方法 | F1 | A2次數 | cached成本／題 |
|---|---|---:|---:|---:|
| NQ | always-A1 | 0.44073 | 0 | 5.6611 |
| NQ | ridge | **0.44856** | 103 / 412 | 7.2413 |
| NQ | 原81-rule calibrated fuzzy | 0.44834 | 103 / 412 | 7.2252 |
| NQ | additive-fixed | 0.44396 | 103 / 412 | 7.2342 |
| NQ | additive-calibrated | 0.44696 | 103 / 412 | 7.2389 |
| HotpotQA | always-A1 | 0.45161 | 0 | 4.2906 |
| HotpotQA | ridge | 0.48023 | 110 / 443 | 5.4788 |
| HotpotQA | 原81-rule calibrated fuzzy | 0.47497 | 110 / 443 | 5.4515 |
| HotpotQA | additive-fixed | **0.48458** | 110 / 443 | 5.4692 |
| HotpotQA | additive-calibrated | **0.48458** | 110 / 443 | 5.4687 |

粗體僅標示本表點估計，沒有顯著勝出的意思。cached成本是原component秒數加總，非重新量測的端到端延遲；A1與A2的收集時段可能不同。

## 效果與不確定性

以下為事先指定新方法主列 additive-calibrated 的差值，單位為F1百分點：

| 資料集／對照 | 差值 | 95% cluster區間 | 12比較Bonferroni近似區間 |
|---|---:|---:|---:|
| NQ − ridge | −0.160 | [−1.423, +1.169] | [−2.014, +1.804] |
| NQ − 原81-rule fuzzy | −0.138 | [−1.450, +1.252] | [−1.981, +1.969] |
| NQ − 同A2次數random | −0.078 | [−1.173, +1.056] | [−1.639, +1.635] |
| HotpotQA − ridge | +0.434 | [−0.919, +1.803] | [−1.528, +2.530] |
| HotpotQA − 原81-rule fuzzy | +0.960 | [−0.241, +2.275] | [−0.672, +2.941] |
| HotpotQA − 同A2次數random | +2.290 | [+0.780, +3.903] | [+0.206, +4.642] |

12比較族包括兩個新模型族×兩資料集×三個對照；完整數字都在 [additive_evidence](additive_evidence/)。random reference是與政策同A2數量的均勻分配期望，且每次bootstrap按該次實際比例重算。區間固定模型與action，不重新排序batch、不重新訓練；其界線和 [穩健性稽核](robustness/REPORT.zh-TW.md) 的不同estimands不能混用。

HotpotQA相對random的訊號比原81-rule更強，但**並未證明 additive 優於ridge**。與原fuzzy的差值區間也含0，故不能把約0.96個百分點的點估計提升寫成已確證改進。NQ為負的差值也不能寫成已確證退步。即使校正區間為正，已曝光test仍只能提供後續假設，不能代替獨立確認。

## 品質改善伴隨的改寫穩定性代價

HotpotQA additive-calibrated 的EM為0.36569，與ridge完全相同；F1增加沒有轉化成更多exact matches。其legacy HRFR為0.15385，高於原81-rule fuzzy的0.10956與ridge的0.13054。這些是描述性點估計，未新增選擇性檢定，但提醒我們**平均F1較高並不等於改寫路由更穩定**。HRFR沿用舊action-order定義，仍應和實際品質及A2率一起解讀。

HotpotQA original-only F1為0.51617、complete-cluster-only為0.49022，分別高於原fuzzy的0.51041、0.48031。NQ original-only為0.45528，complete-cluster-only為0.44012；後者低於原fuzzy0.44309。這些subset保留full-batch動作，沒有在subset重新配額，不能依有利子群重命名主結果。

fixed與calibrated在HotpotQA有相同整體F1／EM，但不表示所有action相同，cached成本亦略有不同。沒有充分證據把width calibration列為本輪主要貢獻。

## 對研究的實際改善

這輪提供一個比「手動調更多規則」更具體的後續方向：以簡單效益模型和低交互作用fuzzy模型並列，研究在固定額外呼叫配額下，品質與改寫穩定性如何取捨。若論文仍主打fuzzy robustness，需要在新資料上同時預先指定品質與穩定性判定，避免只報這次較有利的F1。

目前不應再用同一fresh set持續搜尋alpha、width或新feature。下一輪應先在TRAIN／VAL完成模型決策、人工確認paraphrase語意，再鎖定模型／protocol，收集未曝光原題。A2的context合併、rerank或prompt改進可並行作下一個獨立action實驗，但必須生成新outcomes。

模型JSON、完整VAL候選表、逐題score/action、EM/F1/成本/改寫指標、12個比較與source/input hashes都已保存。既有ver2 evidence未被此結果覆寫。

結果與區間比較圖：[PNG](robustness/figures/robustness_and_additive.png)／[PDF](robustness/figures/robustness_and_additive.pdf)。
