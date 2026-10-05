# 模型剖析

[English](README.md) | [繁體中文](README.zh-TW.md)

量測產生出來的 YOLOv5n 程式的工具。先執行 [YOLO 範例](../../examples/yolov5/README.zh-TW.md)，再於 Linux 的儲存庫根目錄執行這些工具。每次執行都很吃資源：先確認可用記憶體，一次只跑一個。輸出寫在 `out/` 底下，會覆蓋原有內容。

```sh
python -B benchmarks/model/profile_operations.py --case bus_640 --threads 4
python -B benchmarks/model/profile_allocations.py --case bus_640 --threads 4
python -B benchmarks/model/profile_timeline.py --case bus_640 --threads 4 --output out/results/timeline_4t
python -B benchmarks/model/timeline_claims.py out/results/timeline_4t
python -B benchmarks/model/report_plans.py out/models/bus_640_cpu_4t.json --output out/results/plans_4t.json
python -B benchmarks/model/compare_forward.py --baseline-models out/results/baseline/models --output out/results/forward_4t --threads 4
```

| 工具 | 量測內容 |
|---|---|
| [profile_operations.py](profile_operations.py) | 運算圖中每個運算的時間 |
| [profile_allocations.py](profile_allocations.py) | 每個運算的 heap 請求與 Array block 配置 |
| [profile_timeline.py](profile_timeline.py) | 哪些 worker 在什麼時候執行哪個運算 |
| [timeline_claims.py](timeline_claims.py) | 從時間軸算出：依忙碌 worker 數劃分的時間 |
| [report_plans.py](report_plans.py) | 規劃器對每個運算的選擇與估計 |
| [compare_forward.py](compare_forward.py) | 已存下的模型 C 與目前模型的 forward 時間 |

所有計時工具都在同一個 process 內做兩次暖機、七個樣本。

## 運算時間

依運算種類與輸出大小彙整時間，所有權複製另外列出，並計算相依關係上的關鍵路徑。結果寫在 `out/results/model_operations/`。

## 配置

計算 `blk_new`、`blk_copy`、`blk_node`、`blk_half` 的呼叫次數、payload 位元組與進位後的 heap 位元組，以及一般的 heap 請求。數字是累計值，包含容量的 padding；不是存活位元組，也不是 RSS。計時不採用。`--models` 與 `--model-stem` 可指定其他資料夾裡的模型。

## 時間軸

記錄 grow、work、wake 階段，各 worker 的 ring claim，dispatch 區段，以及 block 複製的位元組。

- Claim 表示哪個 worker 持有 runtime ring，不代表哪顆 CPU 在執行。
- Wake、copy 與 dispatch 的區間互相重疊，不可相加。
- Dispatch 區段包含被 inline 進去的 C。
- `--event-capacity` 設定 trace 大小（每個事件 48 位元組）；trace 溢位時工具會失敗。

`timeline_claims.py --kind conv` 列出某一種運算的每一個。

## 規劃器報告

[plan_observation.bend](plan_observation.bend) 以產生的運算圖的形狀呼叫函式庫的規劃選擇器，不執行任何張量資料。加上 `--operations RESULTS.json` 會附上 `profile_operations.py` 量到的各運算時間。

## Forward 比較

以相同的旗標編譯存下的 C 與目前的 C，要求每個快照逐位元相同，接著交替執行五輪，回報各輪的中位數與四分位距。Baseline 資料夾是 `out/models` 的一份複本：模型 C、運算圖 JSON 與建置紀錄。
