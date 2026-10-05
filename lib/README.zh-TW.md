# API 參考

[English](README.md) | [繁體中文](README.zh-TW.md)

```python
import lib/tensor_f32.bend as F
import lib/math_activation.bend as Act
```

`F` 就是全部的公開 API。形狀是 `List<&2,U32>`；軸從零算起。張量是 `F.Tensor()`，只支援 F32。完整的程式見[開始使用](../docs/getting_started.zh-TW.md)。

每個函式都適用三條規則：

- **引數會被用掉。**運算用掉它的張量並回傳結果。二元運算重用左邊的儲存。要保留張量請用 `copy`。
- **失敗留在結果上。**之後的運算跳過計算，把第一個問題往下傳。讀值的時候才檢查。
- **順序固定。**每個加總與每次卷積都依明訂的順序相加，所以結果與 worker 數無關。

## 建立

| 函式 | 結果 |
|---|---|
| `zeros(shape)`、`ones(shape)`、`full(shape, value)` | 填滿同一個值的張量 |
| `arange(start, count)` | 一維：`start`、`start + 1`……共 `count` 個 |
| `from_list(shape, values)` | Row-major 的數值；清單長度必須正好等於形狀的元素數，否則失敗 |
| `from_storage(shape, storage)` | 包裝一個來自 `into_storage` 或檔案 I/O 的 buffer |

## 讀值

| 函式 | 結果 |
|---|---|
| `to_list_or(tensor, fallback)` | Row-major 順序的數值；張量失敗時回傳 `fallback` |
| `item_or(tensor, fallback)` | 單一元素張量的那個值，或 `fallback` |
| `to_list(tensor)`、`item(tensor)` | `Result`：`Done{values}` 或 `Fail{problem}` |
| `check(tensor)` | `Done{tensor}`，或 `Fail{(tensor, problem)}`（狀態已清除） |
| `describe(problem)` | 文字，例如 `add: shapes [2] and [3] do not match` |
| `shape(tensor)` | `(tensor, extents)`；搭配 `then_shape` 使用 |
| `into_storage(tensor)`、`expect(tensor)`、`expect_storage(tensor)` | 以 `Result` 取出 buffer；或在 `IO` 裡取出張量或 buffer，失敗時回報 |

Problem 會寫出失敗的 `F` 函式名稱，種類有三：形狀不相符、形狀不合法或超過機器容量、對該形狀而言引數不合法。

## 保留與組合

| 函式 | 結果 |
|---|---|
| `copy(tensor)` | `(原本的, 複本)` |
| `then(pair, body)` | 把一對張量交給 `body`：`F.then(F.copy(t), kept => copied => F.add(kept, F.relu(copied)))` |
| `then_shape(pair, body)` | 用於 `shape`：`F.then_shape(F.shape(t), tensor => extents => F.reshape(tensor, extents))` |
| `with_workers(tensor, count)`、`workers(tensor)` | 設定或讀取 worker 數。用 `IO.thread_count()` 讀一次 worker 數；結果保留運算元中最大的 |

Bend 只能解構參數與欄位，不能解構呼叫結果，所以成對的值要經過 `then`。

## 逐元素

| 函式 | 結果 |
|---|---|
| `add`、`sub`、`mul`、`div`、`maximum`、`minimum` `(left, right)` | 每個元素 `operation(left, right)`，支援 NumPy 廣播 |
| `add_scalar`、`sub_scalar`、`mul_scalar`、`div_scalar` `(tensor, value)` | `operation(element, value)` |
| `broadcast_to(tensor, shape)` | 把張量重複成 `shape` |

廣播從最後一軸對齊；缺少的軸視為 1；每一軸的長度要相等，或其中一個是 1。`maximum` 與 `minimum` 對 NaN 與正負零依循 Base，順序為左、右。除數為零得到無限大或 NaN，不算失敗。

## Activation

| 函式 | 說明 |
|---|---|
| `relu`、`relu6`、`sigmoid`、`silu`、`tanh`、`hard_sigmoid`、`hard_swish`、`gelu_tanh` `(tensor)` | 與 PyTorch 相同；`gelu_tanh` 是 `approximate='tanh'` |
| `leaky_relu(tensor, slope)` | |
| `activate(tensor, activation)` | 使用 `Act` 的值：`Act.identity()`、`Act.relu()`、`Act.leaky_relu(slope)` 等 |
| `apply(~operation, operations, tensor)` | 自訂的 `F32 -> F32` 函式 |
| `apply_indexed(~operation, operations, tensor)` | 自訂的 `U32 -> F32 -> F32` 函式，另外收到 row-major 索引 |

自訂函式是一個具名的 `def`（或 lambda），以 `~` 當模板傳入，並附上每個元素的純量運算數：每個算術、比較或選擇算一個，`F32.exp` 約 37，`F32.tanh` 約 82。不確定時用 `Act.unknown_operations()`。這個數字只幫助規劃器，結果不受它影響。

```python
def clipped(value: F32) -> F32:
  F32.min(Act.relu_value(value),6.0)

# F.apply(~clipped,4.0,tensor)
# F.conv2d_with(~clipped,4.0,input,weights,bias,1,1)
```

## Reduction

| 函式 | 結果 |
|---|---|
| `sum`、`mean`、`max`、`min` `(tensor)` | 對所有元素依 row-major 順序計算，得到 rank-0 張量 |
| `sum_axis`、`mean_axis`、`max_axis`、`min_axis` `(tensor, axis)` | 該軸被移除 |

每個結果依索引遞增取元素，從 `0`、`-inf` 或 `+inf` 開始；mean 是總和除以個數。順序固定，所以總和等於明寫的迴圈，最後幾個位元可能與 NumPy 的 pairwise `sum` 不同。空張量的 `mean` 是 NaN。

## 佈局

| 函式 | 結果 |
|---|---|
| `reshape(tensor, shape)` | 元素相同、形狀不同；不複製 |
| `transpose(tensor, order)` | 依指定順序排列各軸 |
| `slice(tensor, axis, start, count, step)` | 沿 `axis` 從 `start` 起每隔 `step` 取 `count` 個；`step` 為正 |
| `set_slice(target, axis, start, part)` | 把 `part` 沿 `axis` 從 `start` 寫進 `target` |
| `concat(parts, axis)` | 沿 `axis` 接起來：`F.concat(F.cons(a, F.cons(b, F.single(c))), 1)` |
| `pad(tensor, before, after, value)` | 第 `i` 軸前面加 `before[i]` 個、後面加 `after[i]` 個，值為 `value` |
| `pad_edge(tensor, before, after)` | 同上，但重複最近的元素 |

## 視窗

張量是 `[通道, 高, 寬]` 或 `[張數, 通道, 高, 寬]`。

| 函式 | 結果 |
|---|---|
| `conv2d(input, weights, bias, stride, padding, activation)` | 權重是 `[輸出通道, 輸入通道, 核高, 核寬]`，bias 是 `[輸出通道]`；兩個方向用同樣的 stride 與 padding；activation 套用在輸出上 |
| `conv2d_with(~activation, operations, input, weights, bias, stride, padding)` | 使用自訂的 activation 函式 |
| `max_pool2d(tensor, kernel)` | `kernel` 乘 `kernel` 的不重疊視窗（stride 等於 `kernel`，padding 為 0） |
| `max_pool2d_with(tensor, kernel, stride, padding)` | 完整形式；padding 視為 `-inf` |
| `upsample_nearest(tensor, scale)` | 每一列與每一行重複 `scale` 次 |

卷積的每個輸出從 +0.0 開始，依輸入通道、核的列、核的行的順序加上乘積，最後加 bias。它怎麼執行與規劃見[設計](../docs/design.zh-TW.md)；證明了什麼見[證明](../docs/proofs.zh-TW.md)。

## 函式庫內部

應用程式不需要這些；列出來是給讀原始碼的人。

| 模組 | 角色 |
|---|---|
| [tensor.bend](tensor.bend)、[tensor_layout.bend](tensor_layout.bend)、[tensor_window.bend](tensor_window.bend)、[tensor_convolution.bend](tensor_convolution.bend) | `F` 背後的泛型張量：任意元素型別、dense 與 band 儲存 |
| `tensor_band*.bend` | Row band 儲存與它的運算 |
| [tensor_operations.bend](tensor_operations.bend) | 在 view 上有檢查的 `copy`、`combine`、`apply`、`reduce`、`scatter`、`sweep`；`Accepted` 與 `Rejected` 都交回所有 owner |
| [tensor_view.bend](tensor_view.bend)、[tensor_shape.bend](tensor_shape.bend)、[tensor_cursor.bend](tensor_cursor.bend)、[tensor_access.bend](tensor_access.bend) | View（軸長度、stride、offset）、有檢查的元素數、strided 游標 |
| `kernel_*.bend` | 卷積核心：權重、packing、GEMM、gather、規劃、成本 |
| [storage_buffer.bend](storage_buffer.bend)、[storage_allocation.bend](storage_allocation.bend)、`storage_*.bend` | `Buffer`：owned array，帶邏輯長度與儲存證書 |
| `traversal_*.bend` | 迴圈：copy、update、fold、indexed output、切分樹 |
| [math_activation.bend](math_activation.bend)、[math_fp32.bend](math_fp32.bend)、[math_sequence.bend](math_sequence.bend)、[ordered_matrix.bend](ordered_matrix.bend) | Activation 的值與純量函式；證明所用的有序序列與矩陣乘積 |
| [machine_limits.bend](machine_limits.bend)、[kernel_cost.bend](kernel_cost.bend) 與它們的 `default_*` 檔 | 證明使用的機器界限；規劃器的成本參數 |
| [proofs/](proofs/) | 函式庫的證明 |
