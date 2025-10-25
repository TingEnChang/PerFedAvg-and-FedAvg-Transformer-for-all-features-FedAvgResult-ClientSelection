## Client Selection in Federated Learning: ConvergenceAnalysis and Power-of-Choice Selection Strategies

問題：傳統 FedAvg 多用隨機選客戶，收斂慢、對 non-IID 容易卡關。
主張：每輪先抽一個候選集合 d，比較「當前本地損失」，再挑損失較高的 m 個上場（加速修正在學不好的分佈）。

### 標準 FedAvg baseline：每輪從所有可用客戶中隨機抽取一部分，再做上面的加權平均。

### FL Avg
---
每一輪：伺服端會從可用客戶中隨機選一小部分 m 個（不是全部 K 個）參與本輪訓練；這 m 個把各自的本地更新回傳，伺服端做資料量加權平均（Avg）得到新的 global。
沒被選到的客戶：本輪不訓練、不上傳，照舊用「上一版或最新的 global」做推論/服務。



### Client Selection
---
Power-of-Choice：每輪先隨機抽一個候選集 d，再從候選裡挑當前損失較高的 m 個上場（pow-d=全量損失、cpow-d=小批估損、rpow-d=上一輪滾動平均損失當代理）。聚合仍然用 Avg（資料量加權平均）。

- pow-d：候選 d 個→下發 wₜ→每個用全資料算損失→回報→選 m。最準、成本最高。
- cpow-d：同上，但每個只用小批估損。較省算。
- rpow-d：候選 d 個→直接用上次訓練的平均損失排序→選 m。幾乎零通訊＋零選客計算，靠「滾動分數」＋「探索/過期校準」保持排序有效。

滾動分數：每次被選取後計算損失比並記錄下來。
冷啟：從未上場者 score 設很大（例如:+∞），確保一旦進候選就會被選到。
過期校準：若某客戶分數「太久沒更新」（超過 H 輪），當它被抽進候選時，臨時要它算一次小批平均損失刷新分數（只對該候選、開銷很小）。


「損失比」：
論文沒有要算「比值」；更精確的是「損失分數（平均損失）」，用來排序候選。
rpow-d 的分數 =「上一次該客戶本輪訓練過程的平均損失」。

---

# 聯邦學習 FedAvg：Server / Client 流程表

| 步驟 | Server 端 | 資料流向 | Client 端 |
|---|---|---|---|
| ❶ 選擇客戶端 | 選出本輪客戶 `m` 個：<br>`selected = random.sample(clients, m)` | — | — |
| ❷ 下發全局模型 | 準備並下發參數：<br>`state = global_model.state_dict()` | **S → C**：`state` | 載入參數：<br>`model.load_state_dict(state)` |
| ❸ 本地訓練 | — | — | 以本地資料訓練 `τ` 個 epoch：<br>`for e in range(τ): train_on_local_data()` |
| ❹ 上傳本地模型 | 接收各客戶回傳參數：<br>`states.append(client_state)` | **C → S**：`model.state_dict()`（或梯度） | 回傳本地參數：<br>`return model.state_dict()` |
| ❺ 聚合（FedAvg） | 加權平均得到全局：<br>`θ_global = Σ(n_i·θ_i) / Σ n_i` | — | — |
| ❻ 評估（可選） | 廣播全局權重請客戶驗證：<br>`broadcast(global_state)` | **S ↔ C**：`global_state` / `validation_loss` | 本地驗證並回傳：<br>`loss = evaluate(global_state); return loss` |
| ❼ 終止條件 | 收斂或達輪數即停止：<br>`if converged or round >= R: stop` | — | — |

---

**符號說明**
- `m`：每輪被選中的客戶數；`τ`：每個客戶本地訓練的 epoch 數。  
- `n_i`：第 *i* 個客戶本地樣本數；`θ_i`：其本地訓練後參數；`θ_global`：聚合後全局參數。  
- FedAvg 通常使用樣本數 **加權平均**（如上式）；也可用簡單平均 `avg(θ_i)`。

# 聯邦學習 FedAvg Client Selection：Server / Client 流程表


| 步驟| Server 端| 資料流向| Client 端|
|---|---|---|---|
| CS-① 抽候選 `d` | 從可用客戶**不放回**抽 `d`：`A_t = sample(online, d, probs=p_k)`  | —| —  |
| CS-② 取得排序分數  | **pow-d / cpow-d**：把 `w_t` 下發給 `A_t`，要求回報損失分數（pow=全量、cpow=小批） ；**rpow-d**：直接用「上一輪平均訓練損失」當代理（未見者設極大） | **S→C**（pow/cpow）：`w_t`、**C→S**：`loss_score` ；**rpow**：此步無通訊 | **pow**：計 `F_k(w_t)` 並回報；**cpow**：小批平均損失並回報；**rpow**：此步不計算 |
| CS-③ 選上場 `m` | 依分數（損失高優先）從 `A_t` 取 `m`：`S_t = top_m(A_t, score)`  | —  | —   |
| ② 下發全局模型     | 對 `S_t` 下發 `w_t`（pow/cpow 可與 CS-②共用；rpow 此時首次下發）| **S→C**：`w_t`  | `model.load_state_dict(w_t)` |
| ③ 本地訓練 | —   | —  | 以本地資料訓練 `τ`（或 `E` 個 epoch），並計**本輪平均訓練損失**  |
| ④ 上傳本地結果 | 收集本地更新與**本輪平均訓練損失**  | **C→S**：`state`/`update` + `avg_loss` | 回傳參數與`avg_loss` |
| ⑤ 聚合（FedAvg） | `w_{t+1} = Σ(n_k·w_k^{t+1}) / Σ n_k`（必要時可做逆機率加權＋裁剪） | —  | —    |
| ⑥ 更新分數緩存     | `loss_cache[k] ← avg_loss[k]`（工程上可選：EMA、過期校準、配額/隨機覆蓋） | —  | —   |
| ⑦ 終止條件 | 若收斂或達輪數則停止，否則進入下一輪 `t←t+1`  | —  | —   |


三種分數取得方式（CS-② 的差異）
pow-d（最精準，成本最高）
Server 對 A_t 下發 w_t
Client 用全資料計 F_k(w_t)（平均損失）回報
Server 依分數選 S_t
cpow-d（小批近似，省計算）
Server 對 A_t 下發 w_t
Client 用小批（大小 b）計小批平均損失回報（作為 F_k(w_t) 的無偏估計）
Server 依分數選 S_t
rpow-d（滾動代理，省通訊＋計算）
選客時不下發也不計算：Server 直接用 loss_cache[k]（「上一輪本地訓練的平均損失」）作分數；未曾上場者分數設為極大以保障冷啟探索
只有被選進 S_t 的客戶，才在訓練前接收 w_t
訓練結束後 Client 回傳本輪平均訓練損失，Server 更新 loss_cache
