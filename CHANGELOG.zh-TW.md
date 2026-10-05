# 更新紀錄

[English](CHANGELOG.md) | [繁體中文](CHANGELOG.zh-TW.md)

本專案所有值得注意的變更都記錄在這裡。格式依循 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.1.0/)，版本號採用[語意化版本](https://semver.org/lang/zh-TW/)。在 1.0.0 之前，公開 API 可能在次版本之間變動。

## [0.0.1] - 2026-10-05

第一個公開版本。需要官方 Bend v2.0.35。

### 新增

- **F32 張量 API**（`lib/tensor_f32.bend`），形狀在執行期檢查，失敗由結果帶著走：
  - 建立：`zeros`、`ones`、`full`、`arange`、`from_list`、`from_storage`；
  - 逐元素運算，支援 NumPy 廣播：`add`、`sub`、`mul`、`div`、`maximum`、`minimum`，以及 `_scalar` 版本；
  - Activation：`relu`、`relu6`、`leaky_relu`、`sigmoid`、`silu`、`tanh`、`hard_sigmoid`、`hard_swish`、`gelu_tanh`、`activate`，自訂函式透過 `apply` 與 `apply_indexed`；
  - Reduction：`sum`、`mean`、`max`、`min`，以及 `sum_axis`、`mean_axis`、`max_axis`、`min_axis`；
  - 佈局：`reshape`、`transpose`、`slice`、`set_slice`、`concat`、`pad`、`pad_edge`、`broadcast_to`；
  - 視窗：`conv2d`、`conv2d_with`、`max_pool2d`、`max_pool2d_with`、`upsample_nearest`；
  - 讀值：`to_list`、`item`、`to_list_or`、`item_or`、`check`、`describe`、`into_storage`。
- **固定 FP32 順序的卷積**，不使用 FMA 或重新結合：不論 worker 數，結果逐位元相同。
- **以 Bend 寫成的規劃器**，依形狀與 worker 數決定一次卷積要用幾個 task。
- **Row band 儲存**，讓多個 worker 不必複製整個張量就能分工，並在卷積、逐元素運算、activation、通道 concat、upsample 與 pooling 之間保留。
- **由未修改的官方 Bend 檢查器檢查的證明**：原生卷積對每個輸入都等於矩陣規格的無條件定理，以及張量運算的數值定理。見 [docs/proofs.zh-TW.md](docs/proofs.zh-TW.md)。
- **YOLOv5n 範例**：官方預訓練網路產生成 Bend 程式，逐層與 PyTorch 比對。
- **檢查**：79 支使用公開 API 的程式與 NumPy 逐位元比對，以及 1、2、4、8 執行緒的原生卷積案例。
- 英文與繁體中文文件。

### 已知限制

- 只做推論：沒有訓練，沒有自動微分。
- 只支援 F32。
- 函式庫沒有載入模型權重的功能；YOLO 範例有自己的讀檔程式。
- 權重多而輸出位置少的卷積、多於一張圖的 batch、寬度小於六的輸出，增加 worker 幾乎沒有幫助。見 [docs/design.zh-TW.md](docs/design.zh-TW.md)。
- 比 PyTorch 慢。見 [docs/performance.zh-TW.md](docs/performance.zh-TW.md)。
- Bend 沒有預設引數、具名引數與多載，所以部分函式有簡短與完整兩種形式。
