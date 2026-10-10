# 開始使用

[English](getting_started.md) | [繁體中文](getting_started.zh-TW.md)

## 安裝

需要 Git、Node.js、[Bun](https://bun.sh)、Python 3 與 Clang（或由 `CC` 指定的 C 編譯器）；Linux、macOS 或 WSL。

```sh
git clone --depth 1 --branch v2.0.36 https://github.com/bendlang/bend.git .tools/bend
git -C .tools/bend rev-parse HEAD    # 必須印出 scripts/toolchain.json 裡的 commit
```

在 Windows 上，launcher 會自己進入 WSL 執行，所以下面的指令在 PowerShell 也能用。機器相關的設定（`CC`、`BENCH_PYTHON`、`LD_LIBRARY_PATH`）寫在 `.tools/environment.sh`，Git 會忽略它。

## 第一個程式

[examples/array_basics.bend](../examples/array_basics.bend)：

```python
import Base
import ../lib/tensor_f32.bend as F
import ../lib/math_activation.bend as Act

def main() -> List<&2,F32>:
  # 1.0 到 16.0，排成 [batch, channels, height, width]
  image   = F.reshape(F.arange(1.0,16),[1,1,4,4])
  # [輸出通道, 輸入通道, kernel 高, kernel 寬]
  weights = F.ones([1,1,3,3])
  # 每個輸出通道一個值
  bias    = F.zeros([1])
  # stride 1、padding 1，再做 ReLU：大小維持 4x4
  feature = F.conv2d(image,weights,bias,1,1,Act.relu())
  # 2x2 視窗：4x4 -> 2x2
  pooled  = F.max_pool2d(feature,2)
  # 讀出數值；上面任何一步失敗就回傳 []
  F.to_list_or(pooled,[])
```

```sh
node scripts/bend_launcher.mjs examples/array_basics.bend -o out/array_basics
./out/array_basics                   # [54.0, 63.0, 90.0, 99.0]
```

一律加 `-o` 編譯。不加的話會用直譯器執行，而直譯器不計算 F32 運算。

## 同一個張量用兩次

運算會用掉它的參數：`F.add(a, b)` 之後，`a` 和 `b` 都不能再用。先複製：

```python
# kept 與 copied 是兩個數值相同的張量
F.then(F.copy(tensor), kept => copied => F.add(kept, F.relu(copied)))
```

## 讀取失敗

形狀不合不會讓程式停下來。結果會帶著問題，後面的運算把它往下傳，在讀值的地方才看到：

```python
def text(result: Result<&2,&2,F.Problem(),F32>) -> String:
  match result:
    case Done{value}: "ok"
    case Fail{problem}: F.describe(problem)

def main() -> String:
  # [2] + [3] 失敗；relu、sum、item 把失敗往下傳
  text(F.item(F.sum(F.relu(F.add(F.zeros([2]),F.zeros([3]))))))
```

會印出 `"add: shapes [2] and [3] do not match"`。

## 寫一個網路

網路就是函式，權重是傳進去的張量。[examples/small_network.bend](../examples/small_network.bend)：

```python
# 3x3 convolution with ReLU that keeps the size, then 2x2 max pooling.
def block(input: F.Tensor(),weights: F.Tensor(),bias: F.Tensor()) -> F.Tensor():
  F.max_pool2d(F.conv2d(input,weights,bias,1,1,Act.relu()),2)

# [1,1,8,8] -> [1,4,4,4] -> [1,8,2,2] -> [1,8]
def network(image: F.Tensor()) -> F.Tensor():
  first  = block(image,F.full([4,1,3,3],0.1),F.zeros([4]))
  second = block(first,F.full([8,4,3,3],0.01),F.ones([8]))
  F.mean_axis(F.mean_axis(second,3),2)
```

```sh
node scripts/bend_launcher.mjs examples/small_network.bend -o out/small_network
./out/small_network
```

## 載入與儲存數值

檔案內容是 FP32 數值，小端序、row-major 順序、沒有檔頭；NumPy 用 `array.astype('<f4').tofile(path)` 寫出。檔案讀不到，或內容與形狀不符時，得到失敗的張量。[examples/save_and_load.bend](../examples/save_and_load.bend)：

```python
def main() -> IO(Unit):
  do IO<Unit>:
    # 1.0 .. 6.0 as two rows, written as 24 bytes
    saved  : F.Tensor() <- F.save_raw("out/values.bin",F.reshape(F.arange(1.0,6),[2,3]))
    # the same file, read with its shape
    loaded : F.Tensor() <- F.load_raw("out/values.bin",[2,3])
    # the sums, or [] if a step above failed
    IO.print(List.show(~&2,~F32,~text,F.to_list_or(F.add(saved,loaded),[])))
```

```sh
node scripts/bend_launcher.mjs examples/save_and_load.bend -o out/save_and_load
./out/save_and_load                  # [2, 4, 6, 8, 10, 12]
```

`F.load_npy` 與 `F.save_npy` 讀寫 FP32 的 NumPy `.npy` 檔：

```python
# a NumPy array file: the shape comes from the file
weights : F.Tensor() <- F.load_npy("out/weights/named/model.0.conv.weight.npy")
```

## 載入模型的權重

`F.take` 依名稱從權重取出張量；`F.keep` 放一份副本進去，留給之後的步驟使用，例如跳接。[examples/yolov5/yolov5n.bend](../examples/yolov5/yolov5n.bend) 是完整的模型。

```python
# A step takes tensors by name; the lines of a do block run in order.
def block(x: F.Tensor(),+name: String) -> F.Model(F.Tensor()):
  do F.Model<F.Tensor()>:
    weights : F.Tensor() <- F.take(name ++ ".weight")
    bias : F.Tensor() <- F.take(name ++ ".bias")
    F.Model.pure(F.Tensor(),F.conv2d(x,weights,bias,1,1,Act.relu()))

def main() -> IO(Unit):
  do IO<Unit>:
    # folder/names.txt lists the names; folder/<name>.npy holds each tensor
    weights : F.Weights() <- F.load_weights("out/weights/named")
    result : F.Tensor() <- F.expect(F.run(F.Tensor(),block(F.zeros([1,3,8,8]),"stem"),weights))
    ...
```

## 使用多個 worker

張量帶著 worker 數。讀一次 pool 大小，設在輸入上，結果會沿用。

```python
def main() -> IO(Unit):
  workers : U32 <- IO.thread_count()
  image = F.with_workers(F.zeros([1,3,640,640]),workers)
  ...
```

```sh
./out/program --threads 4
```

不論 worker 數，數值的每個位元都相同。

## 接下來

- [API 參考](../lib/README.zh-TW.md)
- [YOLOv5n 範例](../examples/yolov5/README.zh-TW.md)
- [設計](design.zh-TW.md)、[證明](proofs.zh-TW.md)、[效能](performance.zh-TW.md)、[開發](development.zh-TW.md)
