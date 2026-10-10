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

它會在缺少時準備 PyTorch 參考結果，編譯模型、執行，並比對每一層、預測與偵測結果。結果寫在 `out/results/<case>/`。同一個編譯好的程式可用於每個案例與任何 `--threads` 值。

## 模型

[yolov5n.bend](yolov5n.bend) 就是這個網路，依官方模型一個區塊一個區塊寫成。區塊用 PyTorch 的名稱取用權重：

```python
# Convolution with its folded batch normalization, then SiLU.
def conv(x: F.Tensor(),+name: String,stride: U32,padding: U32) -> F.Model(F.Tensor()):
  do F.Model<F.Tensor()>:
    weights : F.Tensor() <- F.take(name ++ ".conv.weight")
    bias : F.Tensor() <- F.take(name ++ ".conv.bias")
    F.Model.pure(F.Tensor(),F.conv2d(x,weights,bias,stride,padding,Act.silu()))
```

```sh
node scripts/bend_launcher.mjs examples/yolov5/yolov5n.bend -o out/models/yolov5n
YOLO_OUTPUT=out ./out/models/yolov5n --threads 4      # forward ... ms，以及 out/pred.npy
```

| 變數 | 意義 | 預設 |
|---|---|---|
| `YOLO_SIZE` | 正方形影像的邊長，32 的倍數 | 640 |
| `YOLO_INPUT` | 原始 FP32 影像 `[1,3,size,size]` | `out/results/bus_640/input.bin` |
| `YOLO_OUTPUT` | 存放 `pred.npy` 的資料夾 | `out/results/bus_640/native` |
| `YOLO_LAYERS` | 有設定時，每一層的輸出也會存下來 | 未設定 |
| `YOLO_EXTRA` | 存檔那一次之前先跑幾次 forward | 0 |

## 檔案

| 檔案 | 角色 |
|---|---|
| [yolov5n.bend](yolov5n.bend) | 模型與它的程式 |
| [prepare_reference.py](prepare_reference.py) | 匯出 PyTorch 參考結果與融合後的權重，存成 `out/weights/named/<name>.npy` |
| [numpy_yolo.py](numpy_yolo.py)、[golden_model.py](golden_model.py)、[image_pipeline.py](image_pipeline.py) | NumPy 模型與影像處理 |
| [compare_results.py](compare_results.py) | 比對與其門檻 |
| [build_run.sh](build_run.sh) | 建置編譯好的模型，並在一個案例上執行 |

偵測解碼屬於這個範例，不屬於函式庫。重複計時請用[後端 benchmark](../../benchmarks/backends/README.zh-TW.md)。
