# 穩健性、完整性與研究改善稽核

本次使用真實 v0.4 快取，在原先四訊號、最大25% A2配額的 frozen policies 上完成兩資料集各10,000次 test cluster bootstrap，以及各100次 TRAIN／VAL cluster重抽樣重新擬合。六個方法的原始候選、逐題action與F1均和既有證據完全一致。程式與資料記帳可重現；**fuzzy 特有的效果尚未證實，整體研究仍需新的確認性實驗。**

原始證據保留在 `../evidence/v04_fresh_4sig/`；本輪新增的完整輸出、逐次選模、逐題選A2頻率與雜湊在 [evidence](evidence/)。所有小數F1以0–1表示；下表用「F1百分點」，例如+1.329表示F1增加0.01329。

比較圖：[PNG](figures/robustness_and_additive.png)／[PDF](figures/robustness_and_additive.pdf)。

## 1. 結論對統計定義敏感

| 資料集 | fuzzy−random 點估計 | 固定動作、每次同A2比例，95%區間 | 重抽batch後重新排序，95%區間 | 固定動作的8比較校正區間 |
|---|---:|---:|---:|---:|
| NQ | +0.060 | [−0.899, +1.029] | [−1.190, +0.894] | [−1.277, +1.481] |
| HotpotQA | +1.329 | [+0.135, +2.638] | [−0.084, +2.615] | [−0.335, +3.259] |

校正使用每個比較99.375%的 percentile區間（Bonferroni近似），比較族是2資料集×4個F1對照。這只是保守的探索性敏感度；不涵蓋所有額外分析，也無法消除已重複檢視測試集的偏差。

**HotpotQA 的正向訊號尚不足以支持穩健勝過random。** 原本的 frozen assignments 在不同batch可能違反25%比例；把新抽到的batch用相同frozen scores重新排序後，區間已包含0。NQ 三種定義都沒有明確優勢。fuzzy−ridge在兩資料集的95%與校正區間均包含0。HotpotQA fuzzy相對always-A1的提升較一致，但這同時付出額外A2呼叫，不能直接說是同成本的改善。

原本 `metrics.py` 的random比較固定原始A2比例，本輪另外算「每次重抽後比例」與「每次重排batch」。三種點估計相同，區間回答不同問題；完整三種結果均保存，沒有挑其中顯著者當結論。另用seed42與314159各2,000次抽樣，檢查Monte Carlo敏感度；主表固定seed20261016。

## 2. 訓練資料與選模也有不穩定性

TRAIN每次抽180個原題clusters，cluster內variants共同保留且總權重1；VAL獨立按cluster重抽，依相同網格選模與always-A1回退。每次重新計算TRAIN jitter widths。scaler因缺raw TRAIN signals維持封存值，因此尚未涵蓋scaler估計變異。

| 資料集 | fuzzy平均動作一致率 | 重抽後F1的2.5–97.5百分位 | 未優於同次數random的次數 | VAL回退always-A1 |
|---|---:|---:|---:|---:|
| NQ | 81.8% | 0.4381–0.4512 | 80 / 100 | 7 / 100 |
| HotpotQA | 84.0% | 0.4516–0.4819 | 15 / 100 | 13 / 100 |

這些百分位是**訓練／選模穩定性分布，不是泛化信賴區間**，100次使用的是同一組已曝光test。動作一致率也受多數題選A1影響，不能單獨當成穩健性勝利；例如完全回退A1仍可與原政策約75%一致。

NQ原始的微小正增益在大多數重抽fit下消失。HotpotQA比NQ穩定，但仍有回退與不同選擇。ridge也不穩定：NQ有22次、HotpotQA有34次回退A1。因此問題不只是fuzzy架構，亦包含小VAL樣本與弱改善量訊號。

## 3. 資料與缺失值檢查

| 檢查 | 結果與界線 |
|---|---|
| 原報告數值重現 | 先前完成6,494個數值比對，全部一致；原輸入32個檔案未改動 |
| cluster/query/answer metadata | 本輪兩資料集全部通過；保留的paraphrase均可對應通過的screening記錄 |
| TRAIN／VAL／old TEST／fresh文字重疊 | 全部variants經NFKC、大小寫、標點與空白正規化，跨split與cluster精確重複為0；**未證明沒有語意重複** |
| fresh完整cluster | NQ128/150；HotpotQA143/150。品質分析保留不完整cluster，成對改寫指標只能用完整者 |
| 改寫過於接近 | token Jaccard≥.9的sibling pairs：NQ8/390、HotpotQA5/436。這是詞面診斷，不能證明或否定語意等價 |
| 答案與評分重算 | cache保留各action的answer字串，先前已對5,164列重新計算EM/F1且一致；缺完整prompt/context與原始模型執行logs，仍不能從頭重現生成過程 |
| 成本 | component加總一致；尚未消除A1舊run與A2新run的時段／硬體熱狀態偏差 |

NQ只有原題的subset，fuzzy相對該subset同A2比例random變成−0.540百分點；完整cluster subset為−0.087百分點。HotpotQA原題subset為+1.465百分點，完整cluster為+1.551百分點。這些subset保留原始full-batch actions，所以subset A2比例不必等於25%；不能把差異解讀成公平重新分配後的效能，也不能只報有利subset。

按cluster等權並以相同加權A2比例比較random，NQ為+0.025百分點、HotpotQA為+1.225百分點。它們與query等權主結果量級相近；改寫數量不均並非HotpotQA效果的全部來源。HotpotQA刪除任一cluster後的random增益仍為+1.015至+1.445百分點，未由單一題完全驅動；NQ刪除一題就可能改變增益正負。

## 4. 模型結構提供了改善方向

81條規則的TRAIN fractional support總和只有180；NQ有75條、HotpotQA有72條support<5。這是membership質量，不是獨立樣本數。新診斷另按原題cluster計算Kish有效數，避免把相關variants當成獨立觀測。

主模型選prior20。TEST預測中，global TRAIN mean的平均收縮係數分別為NQ **79.3%**、HotpotQA **74.4%**。這說明局部規則大幅依賴全域平均，值得測試降低模型複雜度；不是說這個係數就是「正確率」或Bayesian posterior。

TEST的預測改善量與實際改善量Spearman相關很弱：fuzzy NQ−0.032、HotpotQA+0.075；ridge為−0.033、+0.079。它們是含相關variants的描述性相關，不是獨立樣本檢定。僅調width並沒有強力證據能解決訊號不足。

因此另開一輪 [additive fuzzy實驗](../ADDITIVE_EXPERIMENT_PROTOCOL.md)：將81個交互作用規則改為12個單訊號membership基底，固定與校準width都報告，仍用同TRAIN／VAL、同配額及ridge對照。這同時改變基底與估計器，不能只歸因於規則數；結果保存於獨立目錄，沒有覆蓋本輪主分析。

## 5. 完整性仍缺哪些證據

現有cache已足以完成離線routing研究重評，但尚不足以完成端到端與確認性主張。需要原實驗者補充：

1. raw TRAIN／VAL signals、完整retrieval結果與corpus/index快照、模型revision／GGUF hash及執行設定，才能獨立重跑預處理與檢索生成。
2. 每個action的完整prompt/context與原始生成logs，並釘選評分程式版本；現有answer字串已足以驗證封存EM/F1，補件主要用來重現生成與解釋A2成功／傷害機制。
3. 原始timing logs與執行順序；下一輪應同一環境隨機交錯A1／A2，另計signal、router及排序成本。
4. v0.4 pilot未達原門檻後仍進full run的runlog說明；不能把`passes_bar=false`回寫為成功。
5. 未曝光的新原題test與兩位獨立審核者的paraphrase語意標記。

已準備 [60題盲審樣本](evidence/blind_paraphrase_review.jsonl)，每資料集隨機抽30個有改寫的clusters、每題取一個paraphrase，隨機左右順序。請兩位審核者各自填寫一份；不要給他們 [對照key](evidence/blind_review_key.json)，也不要提供路由與F1。原題無保留paraphrase者不在此抽樣母體，eligibility已保存。此檔目前未有人評分，不能宣稱人工品質已通過。

## 6. 下一步的優先順序

先決定研究主張：若核心仍是fuzzy，下一輪需以**新test上fuzzy對同訊號ridge**為主要比較；random與always-A1只能補充操作價值。若核心改成學習額外檢索效益，應坦白調整貢獻敘述，而不是把ridge效果算成fuzzy優勢。

對1個F1百分點差異，以目前cluster差值變異、雙側alpha=.05與80% power做粗略固定policy近似，約需NQ458、HotpotQA1,018個獨立原題clusters；2個百分點則約115、255。這不是已達power、不是保證，也未包含多重比較／選模／生成隨機性。應使用保守變異範圍再做模擬規劃，不能把三個改寫當三個獨立題來灌大樣本。

優先改善A2本身的可救援性與可預測性，再增加路由複雜度。例如測試合併候選後用原問題重新rerank、保留A1有效證據；這些改動必須重新生成outcomes，不能借用舊A2分數。特徵也必須在選A2之前可取得並計入成本。完整依據與設計取捨見 [文獻與下一階段設計](LITERATURE_AND_DESIGN.md)。

## 重現

從repo根目錄執行，`--out`必須為新目錄：

```powershell
.\.venv-core\Scripts\python.exe -m experiment_ver2.robustness.run --archive runs --out experiment_ver2/runs/robustness-recheck --bootstrap 10000 --refits 100 --seed 20261016
.\.venv-core\Scripts\python.exe -m experiment_ver2.run_additive --archive runs --out experiment_ver2/runs/additive-recheck --bootstrap 10000 --seed 20261017
```
