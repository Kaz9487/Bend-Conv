# 完整模型的後端比較

[English](README.md) | [繁體中文](README.zh-TW.md)

YOLOv5n 在 NumPy、Stelliferous 與 PyTorch 上的 forward 時間。先完成 [YOLO 範例](../../examples/yolov5/README.zh-TW.md)的設定。

```sh
python run.py backends            # CPU
python run.py backends --cuda     # 另外執行手寫的 C 與 CUDA 程式
```

結果寫在 `out/results/backends/`，報告是 `out/results/backend_comparison.md`。

## 方法

- 每個後端在同一個 process 內做兩次暖機、七個樣本。
- 計時：配置、packing、kernel、gather 與解碼。不計時：載入、編譯、前處理、診斷輸出與 NMS。
- Bend 執行的是[範例程式](../../examples/yolov5/yolov5n.bend)。每次計時的 forward 前先複製影像與權重；allocator 與 worker 保持存活。它的時鐘以毫秒計。
- 沒有固定 CPU affinity 與頻率。GPU 計時在讀取時鐘前先同步。
- 數值驗證另外執行；缺少結果就視為失敗。

## 設定

| 變數 | 意義 |
|---|---|
| `BENCH_PYTHON` | Linux 的直譯器，預設 `python3`。需要 NumPy、PyTorch 與 threadpoolctl |
| `YOLO_EXTRA` | Bend process 在存檔那一次之前先做幾次 forward，預設不做。benchmark 設為八次 |

如果直譯器的環境是唯讀的，把缺少的套件裝到 `out/python_packages`，launcher 會把它加進 `PYTHONPATH`：

```sh
python -m pip install --target out/python_packages threadpoolctl
```

## 檔案

| 檔案 | 角色 |
|---|---|
| [generate_graph.py](generate_graph.py) | 匯出每個後端都會讀的模型資料 |
| [bench_numpy.py](bench_numpy.py)、[bench_torch.py](bench_torch.py)、[run_bend.sh](run_bend.sh) | 執行單一後端並取樣 |
| [run_cpu.sh](run_cpu.sh) | 選擇 CPU 案例並記錄環境 |
| [run_cuda.sh](run_cuda.sh)、[probe_cuda.py](probe_cuda.py) | CUDA 能力檢查與 CUDA 後端 |
| [validate.py](validate.py) | 數值門檻；`--backends` 指定哪些結果必須存在 |
| [write_report.py](write_report.py) | 報告，兩種語言 |
