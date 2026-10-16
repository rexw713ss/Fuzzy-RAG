# 文獻對照與下一階段實驗設計

日期：2026-10-16。本文依據已公開的 v0.3／v0.4 caches 與 `evidence/v04_fresh_4sig/` 的 ver2 輸出作探索性分析；沒有新增 model outcomes、改動 frozen 主分析、重選 test 超參數或宣稱確認性成功。以下干預是可檢驗的研究建議，不是已驗證的改善。

## 現有證據支持到哪裡

主配額仍是最多 25% query 使用 A2。HotpotQA 的 ridge F1 為 0.4802，fuzzy-fixed／calibrated 均為 0.4750；NQ 的 ridge 為 0.4486，fuzzy-calibrated 為 0.4483。這些是點估計排序，**不能寫成 ridge 已顯著擊敗 fuzzy**。HotpotQA 的 fuzzy−ridge paired ΔF1 為 −0.00526，95% CI 為 [−0.02397, 0.01161]。來源為 `evidence/v04_fresh_4sig/{nq,hotpotqa}_report.json` 的 `budgets.0.25.methods`。

HotpotQA 的 fuzzy 相對同 A2 使用數的 random expectation 有正的未校正 interval，但 NQ 沒有；ridge 的 HotpotQA 點估計更高。較合理的研究敘述是「學習 action 改善量在部分設定值得後續驗證」，目前無法歸因成 fuzzy 特有優勢，更不能宣稱跨資料集穩定勝出。所有模型與 budget 的完整結果應一併呈現。

不確定度與額外 action 的效益是不同量：S2 很高可能代表兩個 retriever 都找到互不相干資料；這不保證 A2 能救回答案。既有四訊號與單方向 escalation 規則適合作為對照，不能把它們視為實際改善量的正確排序。Adaptive-RAG 用模型實際 outcomes 建立路由 supervision，RouteLLM 則學習模型間的品質選擇，兩者提供「用觀察到的表現訓練決策」的相關研究依據；它們沒有證明本實驗的四訊號足夠，也沒有直接支持特定 fuzzy 架構。[Adaptive-RAG，NAACL 2024](https://aclanthology.org/2024.naacl-long.389/)、[RouteLLM，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5503a7c69d48a2f86fc00b3dc09de686-Abstract-Conference.html)。

## 優先改善順序

| 優先 | 要回答的問題 | 可先用現有 cache 做什麼 | 需要的新資料或 outcomes | 失敗時的解釋 |
|---|---|---|---|---|
| 1 | fuzzy 是否比簡單的效益模型多出可重現的價值？ | frozen fuzzy 對 ridge 的 paired 比較、有限 train/val grouped resampling、rule support 與 assignment 穩定度；不得依 test 改主模型 | 未曝光原始題目的確認性 test；新增訓練題可改善估計 | 若只能勝過 uncertainty baseline，支持的是效益建模，不是 fuzzy 本身 |
| 2 | 81-rule 模型是否主要依靠 global prior？ | 計算每個 rule 的 fractional support、收縮比例、test 啟動低支持 rule 的比例；在 train/val 比較較低交互作用模型 | 模型更動可先重用 cache；新模型效果必須由新 test 確認 | 若簡單模型相當或更好，保留簡單模型並縮小 fuzzy 貢獻主張 |
| 3 | A2 的改善上限和傷害從何而來？ | 分析 ΔF1 正／零／負、fallback、`n_new`、oracle headroom；人工核對典型成功與失敗案例 | 改 context 組合、follow-up prompt、generator 或檢索策略都需要重新產生相應 action outcomes | 若 A2 缺乏可用 headroom，再換 router 也無法解決 |
| 4 | 改寫是否保留相同問題及答案？ | original-only、完整 cluster、所有 query 的敏感度分析；準備盲審樣本 | 人工語意稽核；新 paraphrases 需要新 retrieval／generation outcomes | 低 flip rate 可能反映保守策略；高 flip rate 也可能來自實際語意改變 |
| 5 | 同 A2 配額是否等於同資源？ | 分開報呼叫數、token、CE passes、cached latency；禁止用 test latency 回調 policy | 隨機交錯執行的 latency 重測與線上 batch runner timing | 同配額只代表同 A2 呼叫上限，不能自動推論同秒數、金額或能耗 |

### 1. 下一次固定一個主比較

若研究重點仍是 fuzzy，建議在新 test outcomes 出現前指定 **一個 frozen fuzzy 方法 vs ridge utility** 為主比較，兩者使用相同可取得訊號、相同 A2 定義、同一 `q=0.25` 最大配額與正改善量 eligibility。fuzzy 的選擇應在 train/val 完成；fixed 與 calibrated 的其餘比較列次要。random expectation 與 always-A1 保留為操作價值對照，不能替代對 ridge 的比較。

若目標改為系統效益，則可選 ridge vs 同資源 random 為另一份未來 protocol 的主問題；這是研究問題的改變，不能用當前已見過的 HotpotQA 優勢重新命名本次主結果。兩資料集若各自提出確認性主張，事先處理兩個 primary tests 的 multiplicity；亦可先指定一個資料集為主、另一個為外部驗證，不能在讀新 test 後決定。

反覆看 holdout 後修改方法，即使 fit 函式沒有讀取 test labels，也會讓測試參與研究選擇。Dwork 等人的研究明確討論 adaptive holdout reuse 的泛化風險及特殊保護機制；本 repository 並未實作該機制，所以當前已曝光的 `fresh` 不能再被稱為本次重設計的未曝光確認集。[Generalization in Adaptive Data Analysis and Holdout Reuse，NeurIPS 2015](https://proceedings.neurips.cc/paper/2015/hash/bad5f33780c42f2588878a9d07405083-Abstract.html)。

### 2. 先檢查稀疏規則，再增加模型複雜度

每資料集 training 有 180 個 original clusters。樣本權重令每 cluster 總重為 1，因此 81 個 rule 的 fractional support 總和為 180，平均約 2.22；506／531 個 query rows 不是同樣數量的獨立問題。主配額下兩資料集的 fuzzy validation 均選 `prior=20.0`。

| 資料集／模型 | support 中位數 | support < 1 的 rule | support < 5 的 rule | support = 0 的 rule |
|---|---:|---:|---:|---:|
| NQ fixed | 1.2962 | 34 / 81 | 75 / 81 | 4 / 81 |
| NQ calibrated | 1.2915 | 34 / 81 | 75 / 81 | 2 / 81 |
| HotpotQA fixed | 0.8216 | 43 / 81 | 72 / 81 | 11 / 81 |
| HotpotQA calibrated | 0.8318 | 43 / 81 | 72 / 81 | 11 / 81 |

以上直接由 `evidence/v04_fresh_4sig/{dataset}_models.json` 中主配額所選模型的 `rule_support` 計算。rule-specific empirical mean 在 consequent 中的權重為 `support / (support + prior_strength)`；中位 rule 因此主要受 global prior 影響。這是結構診斷，不是宣稱每個 rule 有 0.8 或 1.3 個獨立樣本；`rule_effective_rows` 也不是獨立 cluster 的有效樣本數。

可檢驗的下一步是低交互作用 fuzzy regression，例如只用單訊號 membership，加上預先限定的少量兩訊號交互作用，對照既有 ridge 與完整 81-rule 模型。只能在 training / validation 的 grouped folds 比較；各 fold 重做允許的 fitting，並維持相同超參數搜尋預算。不要因 test 上某訊號反向就手動反轉或刪除它。若缺 raw training signals，scaler 的 train-only 來源仍只能沿用封存聲明，應在分析限制中保留。

也可將訓練目標寫成直接預測 ΔF1，但在相同線性設計、相同懲罰且未 clipping 時，對兩個 action 各做 ridge 再相減，與對差值做 ridge 可代數等價。單純更換名稱不是新方法；需要明確的新 loss、特徵或可驗證的穩健性假設，並控制額外搜尋自由度。

### 3. 改善 A2 與改善 router 分開驗證

v0.4 pilot 的 `pilot/report.json` 已保存選擇規則：V2 相對 V1 pooled gain 超過 0.01 才選 V2，選定版本 gain 至少 0.04 才通過。實際 V1 gain 僅 0.0011945，`passes_bar=false`。後續 full run 是有用資料，但不能回頭把 pilot 寫成通過。需取得原 runlog 說明未達門檻後繼續的理由。

現有 cache 可先分析 A2 是否增加相關證據、是否犧牲原 A1 有用段落、是否因 follow-up 漂移而傷害，以及 oracle headroom 是否集中於少數 cluster。`n_new=2` 只代表新文件數，不代表新文件相關；答案字串命中也不等於多跳推理證據完整。缺 passage texts 或完整 retrieval cache 時，這些機制不能只憑 F1 判定。

建議一次只測一個 action 改動，例如將現在「A1 前 3 + 新 2」改成「合併候選、用原問題 rerank 後取 5」，保留 generator、answer prompt、context 長度上限與 decoding。這需要重新計算 selection 和 generation outcomes；舊 A2 F1 不可直接當成新 action 的結果。若換 generator 或增加 reasoning tokens，另外報告算力與 prompt 變化，不能把品質增益全部歸因於 routing。

可考慮 A1 CE score gap、證據覆蓋、retriever 一致的高相關文件數等決策前訊號，但先列明取得成本與資料可用性。只能用在選 A2 前已產生的資訊；A2 後的新文件品質不能免費倒流成 routing feature。若用「先生成 A1 答案再決定 A2」的訊號，A2 成本必須新增 A1 generation，與現有成本定義分開。

### 4. 先確認語意，再解釋 paraphrase robustness

fresh 的 NQ 有 22 / 150、HotpotQA 有 7 / 150 個 incomplete clusters。自動 NLI 和 embedding screening 不能代替人工確認「答案與問題條件仍相同」。ID、source index、正規化原題文字零 overlap，也不能排除不同 ID 的語意重複。

建議按 dataset 分層抽原題／改寫 pairs，審核者看不到 router 與 action F1，標記人物／實體、關係、時間、範圍、問句類型與答案是否保持。對爭議案例採第二位審核者或 adjudication；保留判定理由。不要只抽高 HRFR 或低 F1 的個案當作代表性品質估計；若另做錯誤分析，要明確標成目的抽樣。

目前可做完整 cluster 與 original-only 的敏感度分析，但兩者 estimand 改變：原本每 instance 平均會讓三個 variants 的 cluster 權重高於一個 variant 的 cluster；每 cluster 等權是另一個合理但不同的指標。報告它們是否改變結論，而不是挑最有利的一個作主值。RFR、HRFR、within-cluster minimum F1 必須與實際 A2 rate 及總品質一起看；固定 action 天然零 flip。

### 5. 成本主張需要實測部署語境

v0.4 action outcomes 的 latency 是 component sums，檢索部分含 batch time 分攤；部分 A1 rows 重用舊實驗，A2 是不同時段的新呼叫。分量加總一致只能證明記帳一致，無法消除熱狀態、GPU 佔用或執行時間差。原 v0.4 HotpotQA B4 fresh 平均成本 6.7583 秒亦高於 validation budget 6.4775 秒，顯示 validation 平均限制不是 test 保證。

現行 25% quota 的合理主張是「額外 A2 呼叫上限相同」。同 call count 並不保證相同 tokens、文件長度或 latency；同配額且用相同 action 定義是必要對照條件，仍需逐項報告實際消耗。與 always-A2 比較可描述少用多少 A2 calls，但不能只用不同時間收集的 cache 宣稱部署加速百分比。

確認 timing 時，先固定硬體、模型 revision、batch/concurrency、cache 狀態、warm-up 與故障排除規則；在同一時間區塊內隨機交錯 A1／A2，保留所有原始 logs。計入 signal computation、模型 scoring、排序以及必要的 retrieval／generation；另呈現吞吐量與尾端延遲，避免將 batch 每題平均誤稱單題線上 response time。

## 隨機基準與 bootstrap 應估計的量

令 `a_i` 為 frozen policy 的 action，`Y_i1, Y_i2` 為 cached F1，`r = mean(a_i=A2)`。目前原點的比較：

\[
\widehat D = \overline{Y_{i,a_i}} - \left[(1-r)\overline{Y_{i1}}+r\overline{Y_{i2}}\right]
\]

確實是與相同實際 A2 數量的均勻隨機分配之**期望品質**比較；這裡不需要隨機抽 assignment 才能求期望。它不包含「部署一次隨機 assignment」的抽籤變異。

`metrics.py:paired` 先固定 r、建立每筆 random expectation，再交給 cluster bootstrap。抽到不同 clusters 後，原 policy 選 A2 的比例 `r_b` 通常會改變；即使每 cluster 都有三個 variants，cluster 之間選 A2 的數量仍不同。因此目前 CI 應解讀為對**固定 r random mixture**的 conditional 比較，而不是每個重抽樣 batch 都同數量的比較。

建議 sensitivity 保留三種定義，各自命名：

1. **Frozen actions、fixed r**：重現目前估計，不更動原 evidence。
2. **Frozen actions、resampled r_b**：每次 cluster 重抽樣後，以重抽樣內 frozen A2 數除以總 instance 數重算 r_b，再計算 random expectation；不重新訓練或排序。
3. **Frozen model、reranked batch**：每次 cluster 重抽樣後，用 frozen scores 重新取 `floor(q*n_b)`，遵守正改善量條件及既定 tie-break，再用該次實際 r_b 作隨機基準。重複抽到的 cluster occurrences 需保留，不能被 ID dictionary 去重。

第二、三種在原始樣本的點估計應與原定義一致，但 CI 回答的 sampling question 不同。第三種更接近「未來整批 query 的分配」變異，仍依賴 empirical bootstrap 對門檻附近排名的近似，且不含模型重新 fitting 的變異。不能在三種 CI 中挑一種顯著便宣稱成功。

配對檢定應尊重相同題目、評估量及實驗單位；這也是 Dror 等人的 NLP significance 指南所強調的選擇原則。本研究的 cluster 為重抽樣單位是因 paraphrases 非獨立，並非該論文直接驗證了本方法的 CI。[The Hitchhiker’s Guide to Testing Statistical Significance in Natural Language Processing，ACL 2018](https://aclanthology.org/P18-1128/)。

## 確認性樣本數與判定

先決定最小有意義的 ΔF1、主要 comparator、alpha、power、採哪個 estimand，以及是否跨兩資料集做主張；再決定新 original clusters 的數量。不能用當前有利的 observed effect 計算「已達多少 observed power」，當成結果可信的另一項證據。

若主 estimand 是等權 cluster 的 paired mean，簡單近似為 `n ≈ ((z_(1−α/2)+z_power) * σ_cluster / δ_min)^2`。這只是規劃起點：目前指標為 instance 平均、cluster 大小不同，batch quota 排序有相依性，估計的 σ 本身也不穩定，須用保守變異範圍及模擬檢查。舊的 150 clusters 不保證足以區別約 0.005 的 fuzzy−ridge 差；把同題改寫當獨立樣本會誇大資訊量。Card 等人說明 NLP 中低檢定力會妨礙辨別噪音與實質改善，並提出規劃工具；不能直接套用其特定 benchmark 樣本數到本研究。[With Little Power Comes Great Responsibility，EMNLP 2020](https://aclanthology.org/2020.emnlp-main.745/)。

在新 test 之前保存資料排除名單、樣本數、模型／prompt／code hash、主比較及停止規則。CI 包含零代表證據不足，不代表等效；「F1 在一個百分點以內」的點估計也不是非劣性結論。若需主張非劣性，應先設定 margin 與相應單側檢定，並重新規劃 power。

目前最有價值的下一步是完成現有結果的敏感度與語意審核，確立一個可辯護的研究主張，再收集針對該主張的新 outcomes。增加更多 test 上的候選模型或挑選最好 budget，無法替代這些證據。
