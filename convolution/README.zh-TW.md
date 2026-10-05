# 卷積套件

[English](README.md) | [繁體中文](README.zh-TW.md)

卷積的凍結規格，以及根定理所談的引擎。程式呼叫的是[函式庫](../lib/README.zh-TW.md)的 `F.conv2d`，它執行同一個 kernel。

| 檔案 | 角色 |
|---|---|
| [specification.bend](specification.bend)、[specification_law.bend](specification_law.bend) | 凍結。Problem（形狀、輸入、權重、bias）、哪些 problem 有效，以及結果 |
| [matrix_specification.bend](matrix_specification.bend) | 凍結。語意：每個輸出是一列權重與一塊輸入的有序內積，軸為 `(輸入通道, kernel 列, kernel 行)` |
| [reference.bend](reference.bend)、[ordered_reference.bend](ordered_reference.bend) | 直接依規格計算 |
| [shape.bend](shape.bend) | 原生策略共用的機器維度 |
| [scalar_kernel.bend](scalar_kernel.bend) | Direct 策略：一次算一個輸出，不用 workspace |
| [inference.bend](inference.bend) | List 上的原生入口：`infer`、`infer_with_split_depth`、`infer_with_strategy` |
| [array_api.bend](array_api.bend) | 自有 array 上的入口：`infer` 回傳 `Rejected`、`Unsupported` 或 `Accepted` |
| [planning.bend](planning.bend) | `plan`：選擇策略與分割深度 |
| [proofs/](proofs/) | 證明，最後是 `native_inference_matches_matrix` |

## 策略

- **Direct** 以共用的 reduction 迴圈計算每個輸出。
- **Packed** 重排權重、pack 輸入區塊，並執行 `lib/kernel_*.bend` 的 4x8 GEMM。

兩者使用相同的累加順序，並且對每個分割深度都已證明等於矩陣規格。

`automatic(worker_limit, workspace_budget)` 依[設計](../docs/design.zh-TW.md#規劃)裡的成本模型選擇。`None{}` 不限制 workspace；`Some{0n}` 選擇 Direct。Workspace 計入 packed tile、暫時的 leaf 輸出，以及每次分割所複製的內容。

## 規格裡的名稱

`ci/co` 是通道數，`ic/oc` 是通道索引，`oy/ox` 是輸出的列與行，`ky/kx` 是 kernel 的列與行。

[證明導覽](../docs/proofs.zh-TW.md)從根定理一路走到每個迴圈。
