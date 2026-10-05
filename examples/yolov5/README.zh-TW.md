# YOLOv5n 範例

[English](README.md) | [繁體中文](README.zh-TW.md)

以本函式庫執行官方預訓練的 FP32 YOLOv5n，並與 PyTorch 比對。內附的權重與影像有各自的[出處與授權](../../data/README.zh-TW.md)。

## 設定

先完成[函式庫的設定](../../docs/getting_started.zh-TW.md)：

```sh
git clone --depth 1 --branch v7.0 https://github.com/ultralytics/yolov5.git .tools/yolov5-official
python -m venv .venv
. .venv/bin/activate                 # Windows：.venv\Scripts\activate
python -m pip install -r requirements-lock.txt
```

在 Linux 上，OpenCV 需要 libGL（Debian 或 Ubuntu：`apt install libgl1`）。

## 執行

```sh
python run.py model --case bus_640 --threads 4
```

它會在缺少時準備 PyTorch 參考結果，產生 Bend 程式，編譯、執行，並比對每一層、預測與偵測結果。結果寫在 `out/results/<case>/`。同一個編譯好的程式可用於任何 `--threads` 值。

## 檔案

| 檔案 | 角色 |
|---|---|
| [yolo_graph.py](yolo_graph.py) | 模型的運算圖 |
| [generate_yolo_graph.py](generate_yolo_graph.py) | 把運算圖寫成呼叫 [tensor_f32.bend](../../lib/tensor_f32.bend) 的 Bend 程式。張量最後一次使用時以 `Tensor.take` 取走，之前的使用以 `Tensor.get` 讀取 |
| [tensor_io.c](tensor_io.c) | 載入、快照與計時 |
| [prepare_reference.py](prepare_reference.py) | 匯出 PyTorch 參考結果與融合後的權重 |
| [numpy_yolo.py](numpy_yolo.py)、[golden_model.py](golden_model.py)、[image_pipeline.py](image_pipeline.py) | NumPy 模型與影像處理 |
| [compare_results.py](compare_results.py) | 比對與其門檻 |
| [build_run.sh](build_run.sh) | 編譯並執行產生的程式 |

偵測解碼屬於這個範例，不屬於函式庫。重複計時請用[後端 benchmark](../../benchmarks/backends/README.zh-TW.md)。
