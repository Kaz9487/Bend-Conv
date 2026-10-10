# 設計

[English](design.md) | [繁體中文](design.zh-TW.md)

## 分層

| 層 | 負責的內容 | 主要檔案 |
|---|---|---|
| 公開 F32 API | 程式呼叫的函式 | [tensor_f32.bend](../lib/tensor_f32.bend)、[math_activation.bend](../lib/math_activation.bend) |
| 通用張量 | 形狀、狀態、worker 數；dense 與 band 儲存；廣播、layout、視窗、FP32 檔案、具名權重 | [tensor.bend](../lib/tensor.bend)、[tensor_layout.bend](../lib/tensor_layout.bend)、[tensor_window.bend](../lib/tensor_window.bend)、[tensor_convolution.bend](../lib/tensor_convolution.bend)、[tensor_file.bend](../lib/tensor_file.bend)、[tensor_model.bend](../lib/tensor_model.bend)、`tensor_band*.bend` |
| 檢查過的運算 | 在 view 上的 `copy`、`combine`、`apply`、`reduce`、`scatter`、`sweep`，附 guard | [tensor_operations.bend](../lib/tensor_operations.bend)、[tensor_view.bend](../lib/tensor_view.bend)、[tensor_cursor.bend](../lib/tensor_cursor.bend)、[tensor_access.bend](../lib/tensor_access.bend) |
| 卷積 kernel | 權重重排、packing、4x8 GEMM、gather、規劃 | `kernel_*.bend` |
| 儲存與走訪 | 帶憑證的自有 buffer；其他部分共用的迴圈 | `storage_*.bend`、`traversal_*.bend` |
| 卷積套件 | 凍結的規格、參考實作、已證明的 Array 引擎 | [convolution/](../convolution/README.zh-TW.md) |

`lib/` 不依賴 `convolution/`、範例或 benchmarks。

## 張量

張量擁有自己的儲存、各軸長度、填充值、狀態與 worker 數。

- **運算會用掉參數。** 二元運算重用左運算元的儲存。要保留張量就用 `F.copy`。
- **失敗跟著值走。** 失敗之後，後面的運算不做事，只把第一個問題往下傳。程式在讀取結果的地方看到它。
- **形狀在執行期檢查。** 各軸長度是值，不是型別。
- **View 只在內部使用。** `transpose`、`slice` 與 `broadcast_to` 會複製到 dense 儲存；`reshape` 只改標示。

### Dense 與 band 儲存

- **Dense：**一個 row-major 的 buffer。
- **Band：**把 NCHW 張量依列切成數段。每個 band 是獨立的 buffer，順序為 `[channels][rows][width]`。

Worker 可以拿走一個 band，不必複製整個張量。多個 worker 的卷積會產生 band；逐元素運算、activation、沿通道的 concat、nearest upsample 與 max pooling 會保留 band；其他運算先一次 gather 成 dense 儲存。Layout 不會改變任何數值。

### Strided 存取

Cursor 走訪 view 時不做除法。Strided 複製把最內層的軸當成一般迴圈，每一段只推進 cursor 一次。`set_slice`、dense `concat` 與常數 `pad` 是同一個檢查過的 scatter：把連續的來源寫進 strided 的目標 view。

### 熱迴圈裡的函式

每個元素都要呼叫的函式以 template 參數傳入（`~F32.add`），不存在 record 裡。這樣編譯器會把迴圈特化。存在 record 裡的函式，每個元素都要經過 closure 呼叫，量到約慢六倍。

## 卷積

每個輸出從 +0.0 開始，依輸入通道、kernel 列、kernel 行的順序加上各個乘積，最後加上 bias。Padding 的取樣以零參與。不使用 FMA，也不重新結合，所以不論 worker 數與規劃，結果的每個位元都相同。

1. **重排權重**：一次排成八個輸出通道一組的 tile。
2. **Pack**：把一個 leaf 的輸入位置排成八個位置一組的 tile。
3. **GEMM**：每一步處理四個輸出通道乘八個位置，部分和放在區域累加器。
4. **Store 或 gather**：把 leaf 的輸出寫進結果。

多個 worker 時，輸出位置會分成數個 leaf。每次分割會複製重排後的權重與 bias，前半只拿到它會讀的輸入列。

### 規劃

規劃器在呼叫內部，依形狀與張量的 worker 數決定 leaf 數。估計有五個參數，定義在 [default_kernel_cost.bend](../lib/default_kernel_cost.bend)：

| 參數 | 值 | 意義 |
|---|---:|---|
| `mac_ns` | 0.050 | 一次乘加：以固定的 0.4 ns 參考週期計，每 32 次四個週期 |
| `operation_ns` | 0.4 | 一次計入的純量運算 |
| `copy_byte_ns` | 0.100 | 複製或初始化一個位元組 |
| `grain_ns` | 250000 | 一個 leaf 值得開 task 的最小預估工作量 |
| `wake_round_ns` | 180000 | 喚醒 worker 一輪 |

- 每個 worker 最多五個 task，每個 leaf 至少保有一組八個位置。
- 候選方案每次把 leaf 數加倍。只有整體估計（工作、複製、喚醒輪數）嚴格變小才採用分割。
- 一個 worker 時不分割。

這些數字是固定的參考尺度，不是針對某台機器的校正。估計錯誤只會多花時間：對每一種分割，結果都已證明相等。

### 已知限制

以其他網路的形狀量測，單一機器、各一次：

- **權重多、輸出位置少**（例如 512 到 512 通道、3x3、7x7 輸出）：增加 worker 幾乎沒有幫助。每次呼叫都要重排權重，每次分割都要複製它。
- **Batch 大於一：**影像一張接一張處理。
- **寬度小於六的輸出**走較慢的逐項 packing 路徑。
- **Padding 不小於 kernel，以及空的 batch**，走 dense 路徑。

## Bend 沒有提供的功能

- 沒有預設參數、具名參數與多載。簡短與完整的形式用兩個名稱（`max_pool2d` / `max_pool2d_with`）。
- `match` 只能用在參數上，不能用在呼叫結果上。函式庫提供 `to_list_or` 與 `item_or` 來讀取 `Result`。
- 沒有 FMA，也沒有 SIMD 型別。Kernel 是純量的 Bend，由 C 編譯器向量化。
- `Array.new` 會初始化每個元素。
- 共用唯讀儲存需要 `@unsafe`，本函式庫不使用。分割時會複製兩半都要讀的內容。
- 重複使用的 `List<Tensor>` 會讓編譯器把所有 constructor 標成共用，拖慢無關的程式碼。因此 `concat` 接受 `Parts`（`F.cons`、`F.single`）。

檢查器、編譯器與 runtime 的規則詳見[工具鏈規則](toolchain_rules.zh-TW.md)。
