# FuzzyRoute-RAG ver2：效益導向路由與可檢驗的研究程序

制定日期：2026-10-15。這份程序是在讀過 v1 結果後制定的修訂，**不是事前註冊，也不是實驗已成功的證明**。v1 程式與結果保留；ver2 的程式、設定與產物均置於 `experiment_ver2/`，在同名 Git 分支開發。

## 1. v1 的問題與證據界線

下列數值來自 repository 的 `README.md`「Uncertainty signals」與「Full-run results (2026-10-12)」，本文件未重新產生這些模型輸出。AUC 表位於前者；latency、oracle shares、correlation 與樣本數位於後者。

| 發現 | 對研究設計的影響 |
|---|---|
| NQ 的 A0 / A1 平均時間為 6.07 / 6.15 秒；HotpotQA 為 4.39 / 4.30 秒。rerank 本身約 0.06–0.08 秒。 | A0 幾乎不能為 A2 騰出額外時間；HotpotQA 上整體 generation 長度甚至使 A1 比 A0 快。`mean latency <= always-A1` 下退化成 always-A1 是可接受的最佳化結果，不能靠強迫多種 action 製造貢獻。 |
| A2 比 A1 約多 2 秒；oracle 判定需要 A2 的 instance 僅約 NQ 3.3%、HotpotQA 2.5%。 | 額外處理是否有效與如何辨識少數可改善的查詢，比「是否困難」更重要。oracle 使用真實答案，是診斷上界，不能部署。 |
| HotpotQA 的 S1 difficulty AUC 為 0.43，S4 約 0.49；S1 / S4 高度相關。 | 四訊號等權、同方向相加的假設缺乏支持。不要看到 test 方向後翻轉特徵；應用 training outcomes 學習條件效益，並保留簡單非 fuzzy 對照。 |
| Mamdani 輸出在細網格有局部不單調。 | ordinal consequents 單調不代表 centroid surface 單調；但真正的 action 效益也未必隨 uncertainty 單調上升。ver2 不把任意單調性當成效益證據。 |
| B2/B3/B4 等同 always-A1；固定策略 RFR 與 HRFR 天然為 0。 | 「零翻轉」不能單獨證明 paraphrase robustness。必須同時看品質、額外呼叫率、效益與傷害；也不能將 always-A1 的改善歸功於 fuzzy。 |
| 舊 test 只有每資料集 60 個原始題目 cluster；budget sweep 已在 README 中被檢視。 | ver2 在同一批 test 的所有結果都是探索性結果，即使 fit 函式完全沒有讀 test labels，也不能重建未見過的確認性測試。 |

另有兩項評估定義需修正：v1 `scripts/evaluate.py` 用最高 **test F1** 挑選 strongest baseline，也用 test 的固定策略品質與成本建立最佳混合線。前者應改為事前指定或 validation 選定；後者若保留只能標為 hindsight reference，不能當作已部署的公平對手。

## 2. 修訂後的研究問題

**主問題：當 reranking 幾乎免費時，能否辨識哪些查詢值得額外 rewrite + retrieve，讓有限 A2 配額帶來正的答案改善？**

v2 主分析以 A1 為預設，僅在 A1 / A2 之間分配額外處理。A0 保留為固定參考策略；這是根據已知成本結構提出的新研究設定，不能宣稱原本「always-A1 時間上限下三路 routing」的假說已成立。

定義每筆查詢的真實改善量：

\[
\Delta(x)=F1(x,A2)-F1(x,A1).
\]

它可為正、零或負。`A1 失敗的機率`、`檢索不確定度`、`預測的 ΔF1` 是不同對象：難題可能兩個 action 都錯，低不確定度查詢也可能因 rewrite 受害。ver2 直接估計 action 品質，使用預測的改善量排序；不把 uncertainty 解釋為成功機率。

預先指定的假說與否證條件如下：

1. **路由效益**：主配額下，效益路由相對 always-A1 有正的 F1 改善，且優於相同實際 A2 使用率的隨機混合。如果只增加成本或等同隨機分配，就沒有證據支持有效 routing。
2. **fuzzy 表示的增益**：fuzzy utility 必須與同樣讀取 S1–S4、同樣使用訓練 outcomes 的 ridge utility 比較。只勝過舊 uncertainty score，不能證明 fuzzy 比非 fuzzy 學習更好。
3. **寬度校準的增益**：calibrated widths 與 fixed widths 分開報告。寬度未改變或效果無法區分時，不宣稱校準有貢獻。
4. **改寫穩健性**：在品質與實際 A2 使用率一併可比較時，檢查 HRFR、cluster 內 F1 變異與最低表現。若較穩定只是因為幾乎不用 A2，要如實說明。

這種以實際 outcomes 訓練決策的方向與 Adaptive-RAG 的 outcome-derived supervision 有關；RouteLLM 則提供以品質與成本權衡評估 router 的相關研究背景。兩篇論文都不能直接證明本方法有效，也不是本程式已重現的 baseline。[Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389/)、[RouteLLM](https://arxiv.org/abs/2406.18665)。

## 3. 模型、特徵與可解釋性

輸入僅為作決策時已可取得的 S1–S4。P5/P95 scaling 及 paraphrase jitter widths 都只以 training clusters 估計；validation 與 test 使用 frozen 參數。不可輸入 gold answers、任何 action 的實際 F1、後續 A2 context、或事後量得的個別 action latency。

Fuzzy utility 沿用 Low / Medium / High membership，每個訊號三種 membership 的和為 1。四訊號 membership 的乘積形成 81 個 rule 權重；每個 rule 估計各 action 的條件平均 F1，稀疏 rule 向 training global action mean 收縮。最後由 rule 權重加總 action 品質估計，再計算 `predicted_F1_A2 - predicted_F1_A1`。

目前實作保存模型的 81 個 rule 品質與成本估計、`rule_support`、`rule_effective_rows`，並將每筆 test query 的 ID、quota、method、score 與所選 action 寫入 decisions JSONL。主配額的 fuzzy 政策若未回退 always-A1，另輸出啟動 rule ID、乘積權重、training support 與各 rule 的 A2−A1 品質估計；回退政策不使用模型，所以 score 為 null。rule 支持程度與效益都是模型估計，不能寫成已驗證的因果解釋。

訓練時採每個 cluster 總權重相同，避免有三個 paraphrase 的題目比不完整 cluster 更支配參數。收縮使用 fractional membership mass；`rule_effective_rows` 是 Kish row 診斷而非獨立 cluster 樣本數。它們含相關 paraphrases，不能當成相互獨立的樣本數或信賴區間。

有限模型選擇範圍：fuzzy pseudo-count strength 為 `[1, 5, 20]`，ridge alpha 為 `[0.1, 1, 10]`；均由 validation 選擇，tie-break 固定且記錄。fuzzy fixed-width 與 calibrated-width 作為不同方法報告，不事後挑最好的一個改名為唯一主方法。小資料與 81 條規則的稀疏性是限制，收縮不能保證泛化。

保留原始 B3 / B4 uncertainty 排序作基準；所有方法有同一最大配額，但 uncertainty score 不是改善量，因此其候選策略取 top-q，utility 則只選正的預測改善量。所有方法均可由 validation 選擇 always-A1。這個 eligibility 差異需明確報告，不能將 uncertainty 對照的差異全部歸因於 fuzzy 表示。global mean utility 是不使用查詢訊號的基準；ridge utility 是採同一 positive-gain 規則的必要非 fuzzy 學習基準。

## 4. 配額與成本：在看新結果前固定

本次 ver2 固定 A2 最大配額 `q = [0, 0.10, 0.25, 0.50, 1.00]`，**主配額為 `q = 0.25`**。這是針對 v2 的設計選擇，不是從舊 test 最佳點推得的最佳預算，也不保證最佳。

對一批 n 個查詢，最多分配 `floor(q * n)` 次 A2。utility 方法依預測改善量由大到小分配，僅 `predicted ΔF1 > 0` 可啟用 A2；零門檻預先固定，不在 validation 上搜尋。同分以固定的查詢 hash 排序。不強迫花完配額，所以 `q=1` 不等同 always-A2。validation 選定 always-A1，或 test 沒有正的預測改善量，都可能使 A2 使用率為零。validation 的模型選擇與 tie-break 必須保存，test outcomes 不參與排序或調整。

這是**離線批次資源分配**：它使用整批待處理查詢的特徵來排序，但不讀取其答案或 outcomes。不能稱為逐筆到達、無未來查詢資訊的線上 router；若研究線上部署，需另在 validation 固定 threshold 並報告 test 使用率漂移。

公平基準包含 always-A0/A1/A2、原 B3/B4 uncertainty 排序、global utility、ridge utility，以及同實際 A2 使用率的隨機混合。若 router 實際 A2 比率為 r，隨機混合的期望 F1 為：

\[
(1-r)\overline{F1(A1)}+r\overline{F1(A2)}.
\]

r 是 policy 在讀取 test outcomes 前決定的使用率，不能以 test F1 最佳化混合權重。這個期望值不包含隨機 assignment 的 Monte Carlo 變異；其 cluster bootstrap 只描述題目抽樣不確定性。如要比較隨機策略的 RFR / HRFR，需另外固定多個 assignment seeds 並清楚區分兩種變異。

配額是 **A2 呼叫資源**，不是秒數、金額或人工設定的等價成本。每個 q 均須報告實際 A2 比率、總 F1、EM、實測 end-to-end latency 及其分量、retrieval / LLM / rerank 次數與 token 數。A2 fallback 仍耗用一個 A2 配額並計入實際 rewrite 成本。route 計算時間與新增特徵成本應分開量測；若尚未做整條線上的 timing，cached replay 不足以宣稱實際加速。cache 收集時把所有 action 都跑過的總離線成本，也不等於部署時選一個 action 的成本。

所有 budget points 都呈現，不只報最佳點；各點的 test latency 不一定依 q 單調，也不一定滿足某個秒數上限。不得用 test 成本反覆改 policy 直到看起來守住預算。

## 5. 分割、資料稽核與統計

同一原始題目的所有 paraphrases 必須屬於同一 split；dataset 與 cluster identity 都納入 key。檢查重複 key、跨 split cluster、缺 action、非有限數值、非法 F1 / 成本及 signal / outcome 對齊。缺少真實 cache 時停止真實資料分析；合成資料只可證明程式運作，不能填補真實 F1。

training 用於 scaler、width、utility 參數；validation 用於有限候選模型選擇；test 僅在政策 frozen 後計算指標。若任何候選、門檻、特徵或 budget 因 test 結果而改動，產物需另加 exploratory 標記。有限 validation 反覆選模型也會過擬合；保存候選 grid 並限制搜尋，而不是直到得到顯著結果才停止。[Cawley and Talbot, 2010](https://www.jmlr.org/papers/v11/cawley10a.html)。

主要品質採 instance 平均 F1，抽樣單位為原始題目 cluster；所有比較以相同 cluster 重抽樣，保留 paraphrase 相關性。報告 paired ΔF1 與 95% cluster-bootstrap interval，不能拿兩個各自 CI 是否重疊代替 paired 比較。目前 bootstrap 固定已選 action 與隨機基準的使用率 r，反映既有批次分配下的 cluster 抽樣變異；不包含模型選擇變異，也沒有在每次重抽樣重新做 top-q 分配，不能宣稱涵蓋未來批次組成改變的全部不確定性。舊資料上的 interval 仍屬探索性，無法消除先前看過 test 的偏差。60 個 test clusters 不等於約 180 個獨立測試樣本。

RFR / HRFR 只在完整、可配對的 cluster 上計算，清楚記錄 skipped clusters；品質指標的母體也要交代。報告缺 paraphrase 是否與原題難度有關。固定策略零翻轉是結構性事實；品質差的固定策略仍可能有零 HRFR。

保留 `δ=0.02` 作 oracle / harmful-flip 描述門檻，不把它誤稱為已經有統計檢定力的最小改善量。quality oracle 可用於估計 headroom，不能參與政策設定；配額受限 oracle 與 unconstrained oracle 必須分開標示。

NQ、HotpotQA 分開報告，不能只用 pooled 結果掩蓋方向相反。比較多個方法及 budget 的 95% CI 屬多重探索；不能挑其中一個排除零的點作唯一確認性結論。確認性研究需在開啟新 test outcomes 前選定主比較及 multiplicity 處理方式。

## 6. 尚未見過的確認性測試

舊 300 originals / dataset 及任何已用過的 pilot、paraphrase、測試分析都視為已暴露資料。新的 test 必須抽取**不同原始題目**；不能重新分割舊題目或為舊題目生成新的改寫，再稱為 fresh test。

建議的下一階段程序：

1. 先用舊資料完成 ver2 探索，固定模型族、資料處理、q=0.25、primary comparator、統計方法、最低實質改善量與停止規則；保存 commit、設定、資料 ID manifest 與檔案 hash。
2. 從同版本 NQ / HotpotQA 的尚未使用題目池抽樣，排除舊 question IDs、正規化文字重複與已知語意重複。先完成抽樣與封存名單，再產生 outcomes；查不到排除清單時不能聲稱 fresh。
3. 可繼續使用舊 training / validation fit frozen policy，再評分 fresh test，無須為了更新參數偷看新 test。若要新增 training / validation，也要在收集 outcomes 前依 original cluster 分割，與 fresh test 完全隔離。
4. 樣本數由舊資料的 cluster-level paired 差異變異估計與固定最小改善量決定。可用兩側 5%、80% power 的常態近似作規劃起點，再以 cluster simulation 檢查；原 60 clusters 的 CI 很寬，不能沿用後就保證可檢出 1–2 個 F1 百分點。總樣本數與運算預算要在新 outcomes 前固定，不以顯著為停止條件。
5. 完成尚未進行的 paraphrase 人工語意稽核；審核者不看 policy 名稱與 action F1。保留所有自動檢查紀錄，按事前規則標記不等義項目，報告原始與人工有效 subset 的敏感度結果，不能只刪掉使方法表現差的題目。
6. 在固定硬體與推論設定下對 actions 採交錯或隨機執行順序，記錄 warm-up、時間、功耗狀態、錯誤與重跑原因。任何排除規則均不得依答案品質制定；必要的重跑保留原始紀錄。
7. 若對兩資料集各自主張改善，主比較應事先做 multiplicity 控制，例如 Holm 校正，或明確指定單一主資料集、另一個為外部驗證。若 CI 包含零，結論為證據不足；若 fuzzy 與 ridge 相當，只能支持效益學習，不能支持 fuzzy 特有優勢。

## 7. 可交付結果與完成標準

ver2 目前的工程交付為可重現的 cache replay 程式、模型參數與 rule-support 診斷、逐筆 action／score 紀錄、主配額 fuzzy rule trace、完整 cluster 的 F1 SD／range／minimum、固定配額下全部比較，以及 paired uncertainty estimates。人工語意稽核仍待執行。合成 fixture 通過、模型 API 正確、原有單元測試通過，均只屬工程驗證。

本機目前沒有原實驗 cache；原實驗執行者將另行提供。runner 所需檔案為 root 的 `splits.json`，以及各 dataset 子目錄的 `queries.jsonl`、原始未縮放 `signals.jsonl`、涵蓋所有 action 的 `outcomes.jsonl`；原始 clusters、retrieval 與審核紀錄亦應保留供追溯。缺少外部檔案時應列出缺少項目與可執行命令，不能創造實驗成績。新方法若無收益，也應完成並保存負結果，說明是 action 缺乏 headroom、效益不可預測、樣本不足，或模型限制；「把研究做完整」不等於保證某個方法勝出。
