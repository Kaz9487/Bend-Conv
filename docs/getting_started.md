# Getting started

[English](getting_started.md) | [繁體中文](getting_started.zh-TW.md)

## Install

Requires Git, Node.js, [Bun](https://bun.sh), Python 3 and Clang (or the C compiler named by `CC`), on Linux, macOS or WSL.

```sh
git clone --depth 1 --branch v2.0.35 https://github.com/bendlang/bend.git .tools/bend
git -C .tools/bend rev-parse HEAD    # must print the commit in scripts/toolchain.json
```

On Windows the launcher runs itself inside WSL, so the commands below also work from PowerShell. Machine settings (`CC`, `BENCH_PYTHON`, `LD_LIBRARY_PATH`) go in `.tools/environment.sh`, which Git ignores.

## A first program

[examples/array_basics.bend](../examples/array_basics.bend):

```python
import Base
import ../lib/tensor_f32.bend as F
import ../lib/math_activation.bend as Act

def main() -> List<&2,F32>:
  # 1.0 .. 16.0 as [batch, channels, height, width]
  image   = F.reshape(F.arange(1.0,16),[1,1,4,4])
  # [output channels, input channels, kernel height, kernel width]
  weights = F.ones([1,1,3,3])
  # one value per output channel
  bias    = F.zeros([1])
  # stride 1, padding 1, then ReLU: the size stays 4x4
  feature = F.conv2d(image,weights,bias,1,1,Act.relu())
  # 2x2 windows: 4x4 -> 2x2
  pooled  = F.max_pool2d(feature,2)
  # the values, or [] if any step above failed
  F.to_list_or(pooled,[])
```

```sh
node scripts/bend_launcher.mjs examples/array_basics.bend -o out/array_basics
./out/array_basics                   # [54.0, 63.0, 90.0, 99.0]
```

Always compile with `-o`. Without it the interpreter runs, and it does not evaluate F32 arithmetic.

## Use a tensor twice

An operation uses up its arguments: after `F.add(a, b)`, neither `a` nor `b` can be used again. Copy first:

```python
# kept and copied are two tensors with the same values
F.then(F.copy(tensor), kept => copied => F.add(kept, F.relu(copied)))
```

## Read a failure

A wrong shape does not stop the program. The result carries the problem, later operations pass it on, and you see it where you read the value:

```python
def text(result: Result<&2,&2,F.Problem(),F32>) -> String:
  match result:
    case Done{value}: "ok"
    case Fail{problem}: F.describe(problem)

def main() -> String:
  # [2] + [3] fails; relu, sum and item pass the failure on
  text(F.item(F.sum(F.relu(F.add(F.zeros([2]),F.zeros([3]))))))
```

It prints `"add: shapes [2] and [3] do not match"`.

## Write a network

A network is a function, and its weights are tensors passed to it. [examples/small_network.bend](../examples/small_network.bend):

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

## Use several workers

A tensor carries a worker count. Read the pool size once and set it on the input; results keep it.

```python
def main() -> IO(Unit):
  workers : U32 <- IO.thread_count()
  image = F.with_workers(F.zeros([1,3,640,640]),workers)
  ...
```

```sh
./out/program --threads 4
```

The values are the same, bit for bit, for every worker count.

## Next

- [API reference](../lib/README.md)
- [YOLOv5n example](../examples/yolov5/README.md)
- [Design](design.md), [proofs](proofs.md), [performance](performance.md), [development](development.md)
