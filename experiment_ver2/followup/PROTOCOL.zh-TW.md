# 後續實驗的固定程序與完成條件

本程序v2寫於2026-10-16，尚未抽取新原題或產生本程序的新action outcomes。它在收集前納入程式審查及樣本數模擬修正；初版準備紀錄另存`evidence/prior_preparation`，不得混用舊protocol hash。既有ver2／additive的所有test結果已曝光；本程序不把它們改名成未見資料。本地程式可以準備與執行，但「完成」需要下列各項的實際證據。

## A. 改寫語意審核

使用既有60對blind sample作第一輪品質稽核；兩位真人獨立審核，每位檢查相同information need、是否新增／刪除限制、是否文法合理且可作答。審核者不得看到router、gold答案、action F1或左右對照key。

任何一項評分分歧或abstention均需第三位不同真人裁決並留下理由。未填寫為pending，不得補成通過。軟體只能檢查審核者聲明與紀錄，無法證明身分或獨立性。保留原始匯入檔、全部同意／不同意與abstention；報各項一致率及Cohen kappa，預期一致率等於1時kappa未定義，不填1。

這60對抽樣可以估計被抽到母體的語意品質，不能把每個cluster的另一個未審查variant也視為通過，更不能自動刪除某cluster的所有結果。新確認集的人工有效subset應按同一規則事先審核全部保留pairs；原始全集結果仍需報告。

## B. A2改進：先在TRAIN做獨立action pilot

兩資料集各取34個完整TRAIN clusters（seed42、同原v0.4 pilot抽法），每題3variants，共102queries／dataset。這些題全部已曝光；其用途是action設計，不是假裝新的test。固定比較三個action：

1. A1：初次hybrid top20，用原問題cross-encode，取5段生成。
2. A2：原v0.4 V1定義，讀A1前5段產生followup；第二次hybrid top20按followup cross-encode，取A1前3段加2個新段，再用原問題生成。
3. A2-union：同followup prompt及第二次hybrid top20；合併初次top20與第二次top20、依doc ID去重，最多40段，用**原問題**cross-encode，取5段生成。原候選不先被移除，但最終top5可能仍會丟掉A1證據，必須記錄retention。

三者共用generator、revision、answer prompt、temperature0、seed42、32 answer tokens、64 followup tokens、每段text最多200words與5段context。不使用V2 reasoning prompt。新版增加的CE passes如實計費，不能宣稱與A2同成本。

每query三個action順序以seed20261019隨機排列，query順序也固定隨機；每個action獨立執行初次retrieval/rerank，不借用舊run答案或秒數。空白／原問題相同的成功followup觸發A1-context fallback，仍計followup呼叫成本；server或retrieval失敗不視為低成本成功，記錄attempt後停止。恢復執行保留失敗與重跑原因，不因慢或低F1刪除紀錄。

本實作採CPU Contriever／cross-encoder及GPU llama.cpp，避免5GB GPU相互競爭；BM25另啟其Python／JVM，啟動時間列入每次retrieval wall clock。這是公平但尚未最佳化的**本地量測環境**；不能與舊batched latency直接作加速比較。初次model/index load與固定warmup另存，不含於每題；signal、passage lookup及action內所有呼叫計入wall clock，router batch allocation需另測。

Pilot只報兩資料集各自的A2-union−A2、A2-union−A1、成本與證據保留率；F1採10,000次paired cluster bootstrap，seed20261020，兩資料集×兩參考共4比較，報未校正95%與Bonferroni邊際98.75%近似percentile intervals。不保證通過、不依顯著性追加題目。後續是否採用新action必須有新TRAIN／VAL outcomes，重新擬合router，然後鎖定另一份程序；現存original-A2 router不能直接用新action分數冒充確認結果。

## C. 新原題確認集與固定政策

現階段主比較仍固定原81-rule calibrated fuzzy vs同訊號ridge，q=.25、正預測改善量eligibility、VAL選好的原設定；additive結果為後續假說，不因已看HotpotQA較高就替換本輪主比較。`prepare`保存這兩個政策與封存scaler，測試資料不參與擬合或選擇。

使用與原資料同版本NQ-open dev、HotpotQA dev fullwiki題庫。逐一核對所有450已暴露原題的source_index／原文／Hotpot source ID，確認題序相容；排除舊TRAIN／VAL／old TEST／fresh／可見pilot資料與所有variants正規化文字，並排除新pool內精確重複。未知額外曝露及語意重複需另外提供清單或人工審核，僅ID與文字零交集不足以宣稱語意獨立。

固定NQ **1,800**、HotpotQA **3,500**個新原題clusters，seed20261018。初版600／1,300的規劃已在新資料收集前被否決：以曝光的150-cluster分布重抽未來batch、重新配置quota並納入兩資料集多重比較後，1個F1百分點的location-shift偵測機率約0.68／0.77；增加變異後更低。`evidence/planning_final/report.json`保存每個樣本數10,000次、seed20261023的規劃模擬及1／1.25／1.5倍SD情境。採1,800／3,500使1.5倍SD情境約達80%，但不是充分power保證：人審有效subset尚不存在，模擬沿用全部舊variants；資料移轉、過濾、生成變異與稀有錯誤仍可能使power不足。這是置中統計量location-shift近似，沒有捏造新答案，也不是nested-bootstrap power或觀察到的研究成功。

新樣本先封存ID、source SHA256、exposure SHA256、protocol SHA256與policy lock。改寫沿用原prompt，每原題產生5候選，temperature .7、seeds42–46、最多64tokens；CPU雙向NLI argmax entail、mpnet cosine≥.75、answer-leak及正規化重複篩選後保留前2個。另排除舊曝光文字、新樣本全部原文及已保留改寫的跨cluster精確重複；不足2個時保留原題與通過者，不能刪cluster或補抽。原gold僅供leak screen與事後評分，不進generator或gate模型prompt。

全部保留改寫先雙人盲審，review ID／左右原文SHA256必須一對一對應新題；從原始評分重算報告，分歧由第三人裁決。主分析群體固定為**每個原題＋人審有效改寫**，避免無效改寫套用錯誤gold污染結論；完整生成全集另作語意錯誤敏感度分析。先執行`freeze`把樣本、政策、全部review原始紀錄與primary mask綁定，再啟動action collection。收集會保留全集的三個action及全部失敗／重跑紀錄；A2-union只作完整紀錄，本輪確認只使用原A1／A2。缺少來源題庫或人審時停止，不能改用已曝光題或AI代填。

每個資料集主比較是上述有效population內query-weighted fuzzy−ridge F1；q=.25在此population上配置，且只允許positive predicted gain。主推論固定10,000次、seed20261021，以原題cluster為抽樣單位，每個bootstrap batch重配quota，模型／scaler／超參數均固定。報95%及Bonferroni邊際97.5%近似percentile interval；兩資料集構成family=2，某資料集校正區間下界>0才可稱該資料集有主比較優勢，不以其中一個成功推論兩者成功。這不是有限樣本精確檢定，也不包含TRAIN／VAL重訓變異。

EM、實測action wall clock、RFR／legacy HRFR、cluster內F1範圍及最低F1是次要描述端點；其95%區間固定實際assignments且未作多重性修正，不能衍生額外顯著優勢。原題subset沿用有效population配置；完整生成全集另配置q=.25並只報描述F1，不能因其結果較好改成主分析。人審一致率、失敗率、完整／不完整cluster數、實際A2比例及初始化成本均保留。對新結果不得調參、提早停止或依顯著性追加題目；未完成全集不得輸出確認性報告。

## 完成證據

| 工作 | 必須存在的證據 |
|---|---|
| 語意審核 | 兩份完整真人評分、必要的第三人裁決、原始檔hash與一致性報告 |
| 原設定可重跑 | 實際model/index/corpus與revision/hash、成功preflight及backend warmup、原始logs |
| 新action pilot | 每資料集102queries×3actions的全矩陣、未刪除的attempt journal、新生成答案／prompt/context／timing與比較報告 |
| 新確認集 | 1,800／3,500未曝光原題抽樣manifest、來源題序核對、語意稽核、frozen政策與所有實際outcomes |
| 確認性結論 | 事先固定的推論實作、各資料集完整結果與不利／不顯著結果，無test調參 |
| 交付 | 測試通過、artifact雜湊可驗、本地與遠端experiment_ver2相同commit；main不修改 |

任何一格尚未有實際證據都保持未完成。mock backend只驗證計費、資料流與續跑，不是實驗結果。
