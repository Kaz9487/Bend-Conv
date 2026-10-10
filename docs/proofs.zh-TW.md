# 證明

[English](proofs.md) | [繁體中文](proofs.zh-TW.md)

每個證明都是一個 Bend term，由未修改的官方 Bend 檢查器檢查。

```sh
python run.py proofs
```

## 根定理

[native_inference_law.bend](../convolution/proofs/native_inference_law.bend)：

```text
law native_inference_matches_matrix:
  for +problem: ConvSpec.Problem
  {NativeConv.infer(problem) == MatrixSpec.infer(problem) : ConvSpec.ConvOutcome}
```

一個 problem 包含形狀、輸入、權重與 bias。這條定律對它不設任何條件：涵蓋被拒絕的輸入、參考實作的 fallback 與原生執行，對每個分割深度（`inference_with_any_split_depth_matches_matrix`）與兩種策略（`every_strategy_matches_matrix`）都成立。

**數值契約。**每個輸出從 FP32 的 +0.0 開始，依輸入通道、kernel 列、kernel 行的順序加上各個乘積（先乘後加），最後加上 bias。Padding 的取樣也依這個順序參與。定理談的是這一串 FP32 運算，不是實數，也不允許 FMA 或重新結合。

**凍結檔。**[specification.bend](../convolution/specification.bend)、[specification_law.bend](../convolution/specification_law.bend)、[ordered_matrix.bend](../lib/ordered_matrix.bend)、[matrix_specification.bend](../convolution/matrix_specification.bend) 與 [matrix_convolution_law.bend](../convolution/proofs/matrix_convolution_law.bend) 的雜湊必須與 [frozen.json](../data/contracts/frozen.json) 相同。每次檢查證明時都會核對。

## 信任什麼

- 官方檢查器與 Base 函式庫。
- 編譯器、runtime 與其排程器、C 編譯器與硬體。定理談的是 Bend 原始碼。
- `load` 與 `save` 底下的 Bend 檔案 I/O。函式庫與卷積的證明不含任何 `@unsafe` 定義。

證明沒有涵蓋的部分由數值檢查負責：產生的程式碼、模型接線、activation、I/O 與偵測解碼，都與 NumPy 及 PyTorch 比對，數學函式則在每一個 FP32 輸入上與 MPFR 比對（[benchmarks](../benchmarks/README.zh-TW.md)）。

## 卷積證明的路線

| 義務 | 來源 |
|---|---|
| Direct（純量）執行 | [scalar_inference_refinement.bend](../convolution/proofs/scalar_inference_refinement.bend) |
| Packed 執行 | [packed_inference_refinement.bend](../convolution/proofs/packed_inference_refinement.bend) |
| 實際的流程：重排、分割、pack 與 GEMM、gather | [microkernel_inference_refinement.bend](../convolution/proofs/microkernel_inference_refinement.bend)、[microkernel_storage_execution.bend](../convolution/proofs/microkernel_storage_execution.bend) |
| 權重重排成八列一組的 tile | [weight_reorder_readback.bend](../convolution/proofs/weight_reorder_readback.bend) |
| Packing，以及它不會改動的輸入 | [packing_storage_refinement.bend](../convolution/proofs/packing_storage_refinement.bend)、[packing_contents.bend](../convolution/proofs/packing_contents.bend) |
| 把 GEMM 的儲存寫成寫入清單；邊界、數值與涵蓋範圍 | [kernel_gemm_writes.bend](../lib/proofs/kernel_gemm_writes.bend)、[gemm_output_readback.bend](../convolution/proofs/gemm_output_readback.bend) |
| 累加器等於矩陣內積 | [gemm_dot_matches_matrix.bend](../convolution/proofs/gemm_dot_matches_matrix.bend) |
| 每個 leaf 的輸出等於矩陣的值 | [microkernel_leaf_readback.bend](../convolution/proofs/microkernel_leaf_readback.bend) |
| 每次分割的輸入列視窗 | [window_partition.bend](../convolution/proofs/window_partition.bend)、[window_leaf_readback.bend](../convolution/proofs/window_leaf_readback.bend) |
| Gather 的寫入 | [gather_write_program.bend](../convolution/proofs/gather_write_program.bend) |
| 呼叫端 array 的任意容量 | [array_api_validation.bend](../convolution/proofs/array_api_validation.bend) |
| 共用的儲存契約 | [storage_refinement.bend](../lib/proofs/storage_refinement.bend) |

做法從頭到尾相同：把迴圈存入的內容寫成一份寫入清單，證明每次寫入都在界內，證明寫入的值，再證明後面的寫入不會動到它。

## 公開的卷積

`F.conv2d` 透過帶有儲存憑證的 buffer 執行同一個 kernel（[kernel_pipeline.bend](../lib/kernel_pipeline.bend)）。根定理是針對 Array 引擎敘述的；公開路徑有自己的定理：

- **Dense 路徑：**等於 Array 流程（`microkernel_storage_execution`）。
- **Band 路徑：**對程式實際選擇的規劃，輸出的 band 涵蓋每一列且不超出其配置（[band_pipeline_well_formed.bend](../convolution/proofs/band_pipeline_well_formed.bend)），而且每個存入的值都是該輸出點的有序矩陣值（[convolution_leaf_contract.bend](../convolution/proofs/convolution_leaf_contract.bend)、[band_pipeline_execution.bend](../convolution/proofs/band_pipeline_execution.bend)）。

這些與無條件的根定理是分開的。它們的前提是：輸入為 Usable、layout 良構、數值有限、形狀相符，以及程式會檢查的機器 guard。

## 張量運算

每個定理對呼叫只有一個前提：它回傳了 Usable 的結果。

| 運算 | Usable 的結果裡是什麼 | 來源 |
|---|---|---|
| `add`、`sub`、`mul`、`div`、`maximum`、`minimum`，含廣播 | 每個索引上是 `operation(left, right)`，參數順序如此 | [tensor_combine_dense_values.bend](../lib/proofs/tensor_combine_dense_values.bend) |
| `add_scalar`、`mul_scalar` 等 | `operation(x, scalar)` | [tensor_scalar_values.bend](../lib/proofs/tensor_scalar_values.bend) |
| `less`、`less_equal`、`greater`、`greater_equal`、`equal`、`not_equal` | 每個索引上依 `left` 與 `right` 的比較為 1.0 或 0.0 | [tensor_comparison_values.bend](../lib/proofs/tensor_comparison_values.bend) |
| `sigmoid`、`silu`、`tanh`、`gelu_tanh`、`exp`、`log` | [math_fp32.bend](../lib/math_fp32.bend) 中純量函式的依序 map | [tensor_function_values.bend](../lib/proofs/tensor_function_values.bend) |
| `transpose`、`slice`、`broadcast_to`、dense upsample | view 給該索引的位址上的輸入值 | [tensor_gather_values.bend](../lib/proofs/tensor_gather_values.bend) |
| `set_slice`、`pad`、dense `concat` | 透過 view 寫入的部分，其餘不變 | [tensor_set_slice_values.bend](../lib/proofs/tensor_set_slice_values.bend)、[tensor_concat_dense_values.bend](../lib/proofs/tensor_concat_dense_values.bend) |
| `sum_axis`、`max_axis`、`min_axis`、`mean_axis` | 沿該軸依索引遞增、從初始值開始的 fold | [tensor_reduce_axis_values.bend](../lib/proofs/tensor_reduce_axis_values.bend) |
| `max_pool2d` | 每個視窗先列後行、依索引遞增、從 -inf 開始的 fold | [tensor_pool_dense_values.bend](../lib/proofs/tensor_pool_dense_values.bend) |

底層：

| 主題 | 證明的內容 | 來源 |
|---|---|---|
| 邊界與不重疊 | 從程式檢查的 guard 推出：每次讀寫都在範圍內，來源與目的不重疊 | [tensor_operation_bounds.bend](../lib/proofs/tensor_operation_bounds.bend) |
| Cursor | 原生 cursor 依 row-major 順序走訪座標；以最內層軸為單位成段複製，等於逐元素走訪 | [tensor_cursor_refinement.bend](../lib/proofs/tensor_cursor_refinement.bend)、[tensor_cursor_order.bend](../lib/proofs/tensor_cursor_order.bend)、[tensor_access_refinement.bend](../lib/proofs/tensor_access_refinement.bend) |
| 儲存憑證 | Buffer 帶著一個證明：它的 array 具有模型所說的形狀。憑證的工作量每個運算是常數 | [storage_buffer_refinement.bend](../lib/proofs/storage_buffer_refinement.bend) |
| 失敗 | 每個公開函式呼叫哪個檢查過的運算；失敗的輸入維持失敗 | [tensor_public_refinement.bend](../lib/proofs/tensor_public_refinement.bend) |
| Band 遞迴 | `Bands.each` 執行所有 band 運算，只證明一次；各運算只證明自己的 leaf | [tensor_band_each.bend](../lib/proofs/tensor_band_each.bend) |
| Band layout | Band 張量何時良構；以通道、列、行表示的數值 | [tensor_band_shape_refinement.bend](../lib/proofs/tensor_band_shape_refinement.bend)、[tensor_band_values.bend](../lib/proofs/tensor_band_values.bend) |

## 沒有證明的部分

- 規劃器的成本估計、延遲與 workspace 計算。它們不會影響結果。
- 實數上的精確度。由 [math_accuracy.py](../benchmarks/checks/math_accuracy.py) 量測。
- `select` 與 `softmax`。
- Slice view 會到達哪些索引的封閉形式；敘述是跟著 cursor 的座標走的。
- YOLO 範例的運算圖、I/O 與偵測解碼。
- 編譯器與 runtime 的行為。

## 查看

`python run.py proofs` 也會把所有宣告及其前提的清單，以及從根定理出發的參照圖，寫到 `out/results/proof-audit/`。它們用來導覽；判定以檢查器為準。
