# 效能

[English](performance.md) | [繁體中文](performance.zh-TW.md)

YOLOv5n v7.0 forward，bus 影像 640 × 640，FP32，batch 一。每個設定兩次暖機、七個樣本；中位數（毫秒）。

| 後端 | 1 執行緒 | 4 執行緒 |
|---|---:|---:|
| Stelliferous 0.0.2 | 110 | 43 |
| Stelliferous 0.0.1 | 153.7 | 54.8 |
| NumPy / OpenBLAS | 126.9 | 113.3 |
| PyTorch | 49.3 | 14.8 |

2026-10-10 在 Google Cloud 的 `c4d-standard-8` 虛擬機上量測，沒有固定 CPU affinity 與頻率。每個設定都通過與官方參考的逐層、預測與偵測比對。Stelliferous 0.0.2 的時間以整數毫秒回報。Stelliferous 0.0.1 是已發布的那一版搭配它自己的 Bend v2.0.35，在同一台機器上以相同方式量測。

## 重現

先完成 [YOLO 範例的設定](../examples/yolov5/README.zh-TW.md)：

```sh
python run.py backends      # 上面的表
python run.py benchmark     # 單一卷積形狀
```

樣本與環境寫在 `out/results/backends/`，報告是 `out/results/backend_comparison.md`。Actions 分頁的 Benchmark workflow 會在 GitHub 提供的 runner 上執行同一項比較。其他流程見 [benchmarks](../benchmarks/README.zh-TW.md)。

## 環境

- Google Cloud `c4d-standard-8`，Ubuntu 24.04：AMD EPYC 9B45，8 個虛擬 CPU（4 核心，各 2 執行緒）。
- Bend v2.0.36（官方、未修改）；Clang 19.1.1，`-O3 -march=native -ffp-contract=off -std=c11`。
- NumPy 2.4.6 與其內附的 OpenBLAS；PyTorch 2.14.0（CPU、eager、FP32），inter-op 一個執行緒。

## 計時範圍

Forward 計算：配置、所有權複製、packing、kernel、gather 與偵測解碼。載入、編譯、前處理、診斷輸出與 NMS 不計時。

## 樣本

| 設定 | 中位數（毫秒） | 最小–最大（毫秒） |
|---|---:|---:|
| Stelliferous 0.0.2，1 執行緒 | 110 | 110–111 |
| Stelliferous 0.0.2，4 執行緒 | 43 | 42–43 |
| Stelliferous 0.0.1，1 執行緒 | 153.7 | 153.4–154.4 |
| Stelliferous 0.0.1，4 執行緒 | 54.8 | 54.6–56.5 |
| NumPy，1 執行緒 | 126.9 | 126.6–127.7 |
| NumPy，4 執行緒 | 113.3 | 113.0–114.5 |
| PyTorch，1 執行緒 | 49.3 | 48.7–49.5 |
| PyTorch，4 執行緒 | 14.8 | 14.5–15.6 |

## 其他卷積形狀

部分形狀增加 worker 的幫助不大：權重多而輸出位置少、多於一張圖的 batch，以及寬度小於六的輸出。見[設計](design.zh-TW.md#已知限制)。
