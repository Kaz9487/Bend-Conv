# 更新紀錄

[English](CHANGELOG.md) | [繁體中文](CHANGELOG.zh-TW.md)

本文件記錄 Stelliferous 各版本的重要變更，格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.1.0/)，版本編號採用[語意化版本](https://semver.org/lang/zh-TW/)規範。

在正式發布 1.0.0 之前，公開 API 仍可能隨次版本更新而調整。

## [0.0.2] - 2026-10-11

Stelliferous 0.0.2 新增檔案讀寫與模型權重載入功能，同時改善 YOLOv5n 的推論效能。本次更新的主要內容包括

- 支援直接在 Bend 中讀寫 NumPy `.npy` 檔案。
- 支援以 Bend 程式完整定義模型架構，透過名稱取得各層權重。
- 改善 YOLOv5n 推論效能。單執行緒延遲由 153.7 毫秒降至 110 毫秒，四執行緒則由 54.8 毫秒降至 43 毫秒。

自本版本起，專案正式由 **Bend-Conv** 更名為 **Stelliferous**，並升級至 Bend v2.0.36。

### 升級注意事項

- **`conv2d_with` 更名為 `conv2d_custom`**

  原有的呼叫方式與引數均維持不變，只需更新函式名稱。

- **`activate` 不再支援自訂函式**

  將自訂的 activation 函式傳入 `activate`，現在會回傳失敗結果。如需使用自訂 activation，請改用 `apply` 或 `conv2d_custom`。

- **部分數學函式的數值結果有所調整**

  `sigmoid`、`silu`、`tanh` 與 `gelu_tanh` 已改用新的計算演算法，確保在不同機器上產生逐位元一致的結果。

  新版輸出與 0.0.1 可能存在末端位元差異。若先前使用 0.0.1 的輸出作為測試基準，請重新產生參考結果。

### 新增功能

- **檔案讀寫**

  新增 `load_npy(path)` 與 `save_npy(path, tensor)`，支援讀寫 NumPy `.npy` 格式。

  另提供 `load_raw(path, shape)` 與 `save_raw(path, tensor)`，可讀寫不含檔頭的原始 FP32 資料。

- **模型權重管理**

  新增 `load_weights(folder)`，可一次載入指定資料夾中的所有權重。模型執行時，可透過 `take(name)` 依名稱取得所需權重，再使用 `run(model, weights)` 執行完整模型。

  其他相關函式請參閱 [API 參考文件](lib/README.zh-TW.md#模型)。

- **數學函式**

  新增 `exp`、`log` 與 `softmax(tensor, axis)`。

- **比較與條件選擇**

  新增六種逐元素比較運算，包括 `less`、`less_equal`、`greater`、`greater_equal`、`equal` 與 `not_equal`。

  比較結果為由 1.0 與 0.0 組成的張量，分別表示對應位置的條件成立或不成立。六種運算皆提供 `_scalar` 版本，可直接與單一數值比較。

  新增 `select(mask, yes, no)`，依據遮罩從兩個張量中選取對應元素，以及 `clamp(tensor, low, high)`，將張量元素限制在指定範圍內。

- **完整的 YOLOv5n 範例**

  `examples/yolov5/yolov5n.bend` 使用公開 API，以單一 Bend 程式實作完整的 YOLOv5n 網路，支援邊長為 32 倍數的輸入影像。

  另新增 `examples/save_and_load.bend`，示範張量的儲存與載入方式。

### 效能改善

以下為 YOLOv5n 在相同測試機器上，使用 640 × 640 輸入影像進行單次推論的效能比較。

| 版本    |    1 執行緒 |   4 執行緒 |
| ----- | -------: | ------: |
| 0.0.1 | 153.7 毫秒 | 54.8 毫秒 |
| 0.0.2 |   110 毫秒 |   43 毫秒 |

完整測試結果請參閱 [效能測試文件](docs/performance.zh-TW.md)。

本次效能提升來自以下幾項調整。

- **卷積改為分塊執行**

  卷積內部將輸入整理與乘加運算切成固定大小的區塊，各區塊重複使用同一塊工作記憶體，不再為整層配置並初始化一大塊暫存空間。單執行緒與多執行緒採用同一套流程。

- **Activation 改為一次處理八個元素**

  `sigmoid`、`silu`、`tanh` 與 `gelu_tanh` 改以每次八個元素的方式計算，編譯後可使用 SIMD 指令。卷積後接的 activation 與單獨呼叫的 activation 皆適用。

- **不再呼叫系統數學函式庫**

  0.0.1 的 `sigmoid`、`silu`、`tanh` 與 `gelu_tanh` 透過系統提供的 `exp` 與 `tanh` 計算，過程中需轉為雙精度再轉回 FP32。新版只使用 FP32 的加減乘除與 32 位元整數運算，速度更快，結果也不再隨系統函式庫版本而改變。

- **YOLOv5n 的偵測解碼改為張量運算**

  解碼步驟改用函式庫的張量運算與 `sigmoid`，與網路其餘部分共用同一套實作。

- **工作規劃器的成本估計更新**

  規劃器改依分塊後的實際流程估算記憶體用量，並以實際指令數計算各 activation 的成本，使多執行緒的分工決策更貼近實際開銷。

- **模型程式的編譯時間縮短**

  由官方權重產生的 YOLOv5n 程式改為「運算清單加上單一迴圈」的形式。640 × 640 的程式編譯時間由約 6 分鐘縮短至約 14 秒，編譯所需記憶體由 12 GB 以上降至 1 GB 以下。

### 形式化證明與數值精度

- **新增形式化證明**

  新增六種比較運算，以及 `sigmoid`、`silu`、`tanh`、`gelu_tanh`、`exp` 與 `log` 的逐元素語意證明。

  這些證明確認每個輸出元素皆符合對應輸入元素套用指定函式後的結果。詳細內容請參閱 [形式化證明文件](docs/proofs.zh-TW.md)。

- **分塊卷積的證明**

  新的分塊執行流程已完成證明，涵蓋單一工作與平行工作兩種情況。「原生卷積對所有輸入皆符合矩陣規格」的無條件定理維持成立。

- **檔案讀寫的證明**

  已證明數值寫成位元再讀回會得到完全相同的位元，包含 NaN 的內容位元。

- **數值誤差測試**

  針對 `sigmoid`、`silu`、`tanh`、`exp` 與 `log`，已完整遍歷所有 2³² 種 FP32 位元組合，並以 MPFR 計算的正確捨入結果作為參考，評估各函式的數值誤差。

  以下為測得的最大誤差，以 ULP (Unit in the Last Place) 為單位。

  | `sigmoid` | `silu` | `tanh` | `exp` | `log` |
  | --------: | -----: | -----: | ----: | ----: |
  |      1.81 |   1.80 |   1.06 |  0.80 |  0.52 |

  可透過 `benchmarks/checks/math_accuracy.py` 重現上述測試。

### 已知限制

- **`gelu_tanh` 的負數精度仍有問題。** 輸入介於 -2 至 -0.5 時，最大誤差可達 15 ULP。當輸入低於 -2 時，計算結果接近零，目前尚無法保證其可靠性。
- `select`、`softmax` 與其他 activation 函式目前僅完成 NumPy 對照測試，尚未提供形式化證明。
- 上述數值誤差來自實際量測，尚無對應的形式化誤差界限證明。
- **目前無法直接使用 `bend --verdict` 驗證本函式庫。** Bend v2.0.36 會在正式開始驗證前中止執行。現有證明仍由官方檢查器完成驗證，可使用 `python run.py proofs` 執行。
- 目前僅支援推論，尚未實作模型訓練與自動微分。
- 僅支援 FP32 (F32) 資料型別。
- 對於權重數量較多、輸出位置較少的卷積運算，以及 batch size 大於 1 或輸出寬度小於 6 的情況，增加 worker 數量通常無法帶來明顯的效能改善。詳見 [設計文件](docs/design.zh-TW.md)。
- 目前推論效能仍落後於 PyTorch。詳細比較請參閱 [效能測試文件](docs/performance.zh-TW.md)。

## [0.0.1] - 2026-10-05

Stelliferous (原 Bend-Conv) 的首個公開版本，基於官方 Bend v2.0.35 開發。

### 新增功能

- **FP32 張量 API**

  新增 `lib/tensor_f32.bend`，提供基本張量運算與操作介面。張量形狀會在執行期間檢查，錯誤資訊透過回傳結果傳遞。

  主要功能包括

  - **張量建立** — `zeros`、`ones`、`full`、`arange`、`from_list`、`from_storage`。
  - **逐元素運算** — `add`、`sub`、`mul`、`div`、`maximum`、`minimum`，支援 NumPy 廣播規則，另提供對應的 `_scalar` 版本。
  - **Activation 函式** — `relu`、`relu6`、`leaky_relu`、`sigmoid`、`silu`、`tanh`、`hard_sigmoid`、`hard_swish`、`gelu_tanh`、`activate`，以及用於自訂函式的 `apply` 與 `apply_indexed`。
  - **歸約運算 (Reduction)** — `sum`、`mean`、`max`、`min`，以及沿指定軸運算的 `sum_axis`、`mean_axis`、`max_axis`、`min_axis`。
  - **張量形狀與佈局操作** — `reshape`、`transpose`、`slice`、`set_slice`、`concat`、`pad`、`pad_edge`、`broadcast_to`。
  - **卷積與視窗運算** — `conv2d`、`conv2d_with`、`max_pool2d`、`max_pool2d_with`、`upsample_nearest`。
  - **資料存取與狀態檢查** — `to_list`、`item`、`to_list_or`、`item_or`、`check`、`describe`、`into_storage`。

- **具備確定性結果的 FP32 卷積**

  固定 FP32 運算與累加順序，不使用 FMA (Fused Multiply-Add) 或重新結合運算，確保不同 worker 數量下的計算結果逐位元一致。

- **以 Bend 實作的工作規劃器**

  根據張量形狀與 worker 數量，決定每次卷積運算的 task 分配方式。

- **Row band 資料佈局**

  採用 Row band 儲存方式，讓多個 worker 能夠分工處理張量，無須各自複製完整資料。

  此佈局可在卷積、逐元素運算、activation、通道串接 (concat)、上採樣 (upsample) 與池化 (pooling) 等操作之間保留，減少不必要的資料複製。

- **形式化證明**

  使用未經修改的官方 Bend 檢查器驗證相關定理，包括原生卷積對所有輸入皆符合矩陣規格的無條件定理，以及張量運算的數值性質證明。

  詳細內容請參閱 [形式化證明文件](docs/proofs.zh-TW.md)。

- **YOLOv5n 推論範例**

  提供由官方預訓練 YOLOv5n 網路產生的 Bend 程式，並與 PyTorch 進行逐層數值比對。

- **正確性測試**

  建立 79 支使用公開 API 的測試程式，將運算結果與 NumPy 進行逐位元比對。

  卷積測試涵蓋 1、2、4、8 執行緒配置，檢查原生卷積在不同 worker 數量下的正確性。

- **雙語文件**

  提供英文與繁體中文文件。

### 已知限制

- 僅支援模型推論，尚未實作訓練與自動微分。
- 僅支援 FP32 (F32) 資料型別。
- 函式庫尚未提供通用的模型權重載入功能，YOLO 範例使用獨立實作的檔案讀取程式。
- 對於權重數量較多、輸出位置較少的卷積運算，以及 batch size 大於 1 或輸出寬度小於 6 的情況，增加 worker 數量通常無法帶來明顯的效能改善。詳見 [設計文件](docs/design.zh-TW.md)。
- 目前推論效能仍落後於 PyTorch。詳細比較請參閱 [效能測試文件](docs/performance.zh-TW.md)。
- 由於 Bend 尚未支援預設引數、具名引數與函式多載，部分 API 同時提供簡化與完整兩種呼叫形式。
