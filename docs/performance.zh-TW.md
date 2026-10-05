# 效能

[English](performance.md) | [繁體中文](performance.zh-TW.md)

YOLOv5n v7.0 forward，bus 影像 640 × 640，FP32，batch 一。每個設定兩次暖機、七個樣本；中位數（毫秒）。

| 後端 | 1 執行緒 | 4 執行緒 |
|---|---:|---:|
| Bend-Conv | 245.3 | 116.2 |
| NumPy / OpenBLAS | 223.0 | 232.1 |
| PyTorch | 92.7 | 45.3 |

2026-10-04 量測，當時機器上有其他工作在跑；沒有固定 CPU affinity 與頻率。每個設定都通過與官方參考的逐層、預測與偵測比對。

## 重現

先完成 [YOLO 範例的設定](../examples/yolov5/README.zh-TW.md)：

```sh
python run.py backends      # 上面的表
python run.py benchmark     # 單一卷積形狀
```

樣本與環境寫在 `out/results/backends/`，報告是 `out/results/backend_comparison.md`。其他流程見 [benchmarks](../benchmarks/README.zh-TW.md)。

## 環境

- CPU：12th Gen Intel Core i5-12500H。WSL 下的 Linux。
- Bend v2.0.35（官方、未修改）；Clang 19.1.7，`-O3 -march=native -ffp-contract=off -std=c11`。
- NumPy 2.4.6 與其內附的 OpenBLAS；PyTorch 2.14.0（CPU、eager、FP32），inter-op 一個執行緒。

## 計時範圍

Forward 計算：配置、所有權複製、packing、kernel、gather 與偵測解碼。載入、編譯、前處理、診斷輸出與 NMS 不計時。

## 原始樣本

| 設定 | 樣本（毫秒） |
|---|---|
| Bend-Conv，1 執行緒 | 231.7, 251.6, 245.1, 245.3, 245.4, 230.8, 247.1 |
| Bend-Conv，4 執行緒 | 116.2, 156.9, 117.7, 138.9, 112.6, 105.3, 114.3 |
| NumPy，1 執行緒 | 226.2, 224.3, 215.6, 205.6, 211.4, 223.0, 242.3 |
| NumPy，4 執行緒 | 232.3, 217.0, 232.1, 248.9, 236.0, 221.8, 217.3 |
| PyTorch，1 執行緒 | 97.8, 84.4, 92.7, 82.9, 93.7, 85.0, 132.3 |
| PyTorch，4 執行緒 | 61.1, 41.3, 45.3, 54.6, 49.7, 38.8, 39.4 |

## 其他卷積形狀

部分形狀增加 worker 的幫助不大：權重多而輸出位置少、多於一張圖的 batch，以及寬度小於六的輸出。見[設計](design.zh-TW.md#已知限制)。
