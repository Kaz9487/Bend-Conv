# 效能

[English](performance.md) | [繁體中文](performance.zh-TW.md)

YOLOv5n v7.0 forward，bus 影像 640 × 640，FP32，batch 一。每個設定兩次暖機、七個樣本；中位數（毫秒）。

| 後端 | 1 執行緒 | 4 執行緒 |
|---|---:|---:|
| Bend-Conv | 271.7 | 151.4 |
| NumPy / OpenBLAS | 249.9 | 260.0 |
| PyTorch | 89.8 | 51.8 |

2026-10-05 由 [Benchmark workflow](https://github.com/Kaz9487/Bend-Conv/actions/runs/37268568117) 在 GitHub 提供的 runner 上量測。每次分配到的 runner 不盡相同，也沒有固定 CPU affinity 與頻率。每個設定都通過與官方參考的逐層、預測與偵測比對。

## 重現

從 Actions 分頁啟動 Benchmark workflow；或先完成 [YOLO 範例的設定](../examples/yolov5/README.zh-TW.md)，在本機執行：

```sh
python run.py backends      # 上面的表
python run.py benchmark     # 單一卷積形狀
```

樣本與環境寫在 `out/results/backends/`，報告是 `out/results/backend_comparison.md`。其他流程見 [benchmarks](../benchmarks/README.zh-TW.md)。

## 環境

- GitHub 提供的 `ubuntu-24.04` runner：AMD EPYC 7763，4 個虛擬 CPU（2 核心，各 2 執行緒）。
- Bend v2.0.35（官方、未修改）；Clang 18.1.3，`-O3 -march=native -ffp-contract=off -std=c11`。
- NumPy 2.4.6 與其內附的 OpenBLAS；PyTorch 2.14.0（CPU、eager、FP32），inter-op 一個執行緒。

## 計時範圍

Forward 計算：配置、所有權複製、packing、kernel、gather 與偵測解碼。載入、編譯、前處理、診斷輸出與 NMS 不計時。

## 原始樣本

| 設定 | 樣本（毫秒） |
|---|---|
| Bend-Conv，1 執行緒 | 271.7, 272.1, 272.3, 271.7, 271.3, 271.4, 271.6 |
| Bend-Conv，4 執行緒 | 151.3, 151.5, 151.2, 151.2, 151.7, 151.7, 151.4 |
| NumPy，1 執行緒 | 248.4, 249.9, 250.6, 250.8, 248.1, 249.9, 249.9 |
| NumPy，4 執行緒 | 259.2, 264.2, 260.6, 260.6, 259.1, 260.0, 260.0 |
| PyTorch，1 執行緒 | 89.3, 89.1, 89.3, 90.3, 89.9, 90.5, 89.8 |
| PyTorch，4 執行緒 | 51.5, 51.8, 52.0, 54.3, 51.8, 51.4, 51.5 |

## 其他卷積形狀

部分形狀增加 worker 的幫助不大：權重多而輸出位置少、多於一張圖的 batch，以及寬度小於六的輸出。見[設計](design.zh-TW.md#已知限制)。
