# YOLOv5n on the public tensor API, block by block as in the official model,
# for a square image whose side is a multiple of 32. Weights carry their PyTorch names.
import Base
import ../../lib/tensor_f32.bend as F
import ../../lib/math_activation.bend as Act

# Blocks

# Convolution with its folded batch normalization, then SiLU.
def conv(x: F.Tensor(),+name: String,stride: U32,padding: U32) -> F.Model(F.Tensor()):
  do F.Model<F.Tensor()>:
    weights : F.Tensor() <- F.take(name ++ ".conv.weight")
    bias : F.Tensor() <- F.take(name ++ ".conv.bias")
    F.Model.pure(F.Tensor(),F.conv2d(x,weights,bias,stride,padding,Act.silu()))

# 1x1 then 3x3; the input is added back.
def residual(pair: F.Tensor() & F.Tensor(),+name: String) -> F.Model(F.Tensor()):
  (x,kept) = pair
  do F.Model<F.Tensor()>:
    y : F.Tensor() <- conv(x,name ++ ".cv1",1,0)
    y : F.Tensor() <- conv(y,name ++ ".cv2",1,1)
    F.Model.pure(F.Tensor(),F.add(y,kept))

def plain(x: F.Tensor(),+name: String) -> F.Model(F.Tensor()):
  do F.Model<F.Tensor()>:
    y : F.Tensor() <- conv(x,name ++ ".cv1",1,0)
    conv(y,name ++ ".cv2",1,1)

def bottleneck(x: F.Tensor(),+name: String,shortcut: Bool) -> F.Model(F.Tensor()):
  match shortcut:
    case True{}: residual(F.copy(x),name)
    case False{}: plain(x,name)

def bottlenecks(count: Nat,x: F.Tensor(),+name: String,+index: U32,+shortcut: Bool) -> F.Model(F.Tensor()):
  match count:
    case 0n: F.Model.pure(F.Tensor(),x)
    case 1n+rest:
      do F.Model<F.Tensor()>:
        y : F.Tensor() <- bottleneck(x,name ++ ".m." ++ U32.show(index),shortcut)
        bottlenecks(rest,y,name,(index + 1 : U32),shortcut)

# Two branches of the input, one through the bottlenecks, joined by a 1x1 convolution.
def branches(pair: F.Tensor() & F.Tensor(),+name: String,count: Nat,+shortcut: Bool) -> F.Model(F.Tensor()):
  (left,right) = pair
  do F.Model<F.Tensor()>:
    left : F.Tensor() <- conv(left,name ++ ".cv1",1,0)
    right : F.Tensor() <- conv(right,name ++ ".cv2",1,0)
    left : F.Tensor() <- bottlenecks(count,left,name,0,shortcut)
    conv(F.concat(F.cons(left,F.single(right)),1),name ++ ".cv3",1,0)

def c3(x: F.Tensor(),+name: String,count: Nat,+shortcut: Bool) -> F.Model(F.Tensor()):
  branches(F.copy(x),name,count,shortcut)

def pool(x: F.Tensor()) -> F.Tensor():
  F.max_pool2d_with(x,5,1,2)

def pooled_twice(pair: F.Tensor() & F.Tensor()) -> F.Parts():
  (third,copy) = pair
  F.cons(third,F.single(pool(copy)))

def pooled_once(pair: F.Tensor() & F.Tensor()) -> F.Parts():
  (second,copy) = pair
  F.cons(second,pooled_twice(F.copy(pool(copy))))

def pyramid(pair: F.Tensor() & F.Tensor()) -> F.Tensor():
  (first,copy) = pair
  F.concat(F.cons(first,pooled_once(F.copy(pool(copy)))),1)

# The input and three successive 5x5 max pools, joined.
def sppf(x: F.Tensor(),+name: String) -> F.Model(F.Tensor()):
  do F.Model<F.Tensor()>:
    first : F.Tensor() <- conv(x,name ++ ".cv1",1,0)
    conv(pyramid(F.copy(first)),name ++ ".cv2",1,0)

# The current tensor joined with one kept earlier.
def joined(x: F.Tensor(),+name: String) -> F.Model(F.Tensor()):
  do F.Model<F.Tensor()>:
    earlier : F.Tensor() <- F.take(name)
    F.Model.pure(F.Tensor(),F.concat(F.cons(x,F.single(earlier)),1))

# With trace, a copy of the tensor stays in the weights under the label.
def traced(trace: Bool,+label: String,x: F.Tensor()) -> F.Model(F.Tensor()):
  match trace:
    case True{}: F.keep(label,x)
    case False{}: F.Model.pure(F.Tensor(),x)

# Detection head

# Column numbers along the last axis and row numbers along the one before: [1,2,grid,grid].
def offsets(+grid: U32) -> F.Tensor():
  F.concat(F.cons(F.broadcast_to(F.reshape(F.arange(0.0,grid),[1,1,1,grid]),[1,1,grid,grid]),
    F.single(F.broadcast_to(F.reshape(F.arange(0.0,grid),[1,1,grid,1]),[1,1,grid,grid]))),1)

# A box centre is on the grid.
def centres(scores: F.Tensor(),+grid: U32,+stride: F32) -> F.Tensor():
  F.mul_scalar(F.add(F.sub_scalar(F.mul_scalar(scores,2.0),0.5),offsets(grid)),stride)

def squared(pair: F.Tensor() & F.Tensor()) -> F.Tensor():
  (first,second) = pair
  F.mul(first,second)

# A box size is on its anchor.
def sizes(scores: F.Tensor(),anchors: F.Tensor()) -> F.Tensor():
  F.mul(squared(F.copy(F.mul_scalar(scores,2.0))),anchors)

def boxes(pair: F.Tensor() & F.Tensor(),anchors: F.Tensor(),+grid: U32,+stride: F32) -> F.Tensor():
  (first,second) = pair
  F.concat(F.cons(centres(F.slice(first,1,0,2,1),grid,stride),F.single(sizes(F.slice(second,1,2,2,1),anchors))),1)

# The first four of the 85 fields of an anchor are its box; the others are scores.
def decoded(pair: F.Tensor() & F.Tensor(),anchors: F.Tensor(),+grid: U32,+stride: F32) -> F.Tensor():
  (scores,copy) = pair
  F.set_slice(scores,1,0,boxes(F.copy(F.slice(copy,1,0,4,1)),anchors,grid,stride))

# [1,255,grid,grid] to [1,3*grid*grid,85]; anchors is [3,2,1,1], a width and a height each.
def decode(logits: F.Tensor(),anchors: F.Tensor(),+grid: U32,+stride: F32) -> F.Tensor():
  F.reshape(F.transpose(decoded(F.copy(F.reshape(F.sigmoid(logits),[3,85,grid,grid])),anchors,grid,stride),[0,2,3,1]),[1,(3 * grid * grid : U32),85])

# The raw output of one scale: a 1x1 convolution without activation.
def logits(x: F.Tensor(),+name: String,+label: String,+trace: Bool) -> F.Model(F.Tensor()):
  do F.Model<F.Tensor()>:
    weights : F.Tensor() <- F.take(name ++ ".weight")
    bias : F.Tensor() <- F.take(name ++ ".bias")
    traced(trace,label,F.conv2d(x,weights,bias,1,0,Act.identity()))

def detect(p5: F.Tensor(),+size: U32,+trace: Bool) -> F.Model(F.Tensor()):
  do F.Model<F.Tensor()>:
    p3 : F.Tensor() <- F.take("p3")
    first : F.Tensor() <- logits(p3,"model.24.m.0","head_0",trace)
    p4 : F.Tensor() <- F.take("p4")
    second : F.Tensor() <- logits(p4,"model.24.m.1","head_1",trace)
    third : F.Tensor() <- logits(p5,"model.24.m.2","head_2",trace)
    F.Model.pure(F.Tensor(),F.concat(F.cons(decode(first,F.from_list([3,2,1,1],[10.0,13.0,16.0,30.0,33.0,23.0]),(size / 8 : U32),8.0),
      F.cons(decode(second,F.from_list([3,2,1,1],[30.0,61.0,62.0,45.0,59.0,119.0]),(size / 16 : U32),16.0),
      F.single(decode(third,F.from_list([3,2,1,1],[116.0,90.0,156.0,198.0,373.0,326.0]),(size / 32 : U32),32.0)))),1))

# Network

def layer(+trace: Bool,+label: String,step: F.Model(F.Tensor())) -> F.Model(F.Tensor()):
  do F.Model<F.Tensor()>:
    x : F.Tensor() <- step
    traced(trace,label,x)

def network(image: F.Tensor(),+size: U32,+trace: Bool) -> F.Model(F.Tensor()):
  do F.Model<F.Tensor()>:
    # backbone
    x : F.Tensor() <- layer(trace,"layer_00",conv(image,"model.0",2,2))
    x : F.Tensor() <- layer(trace,"layer_01",conv(x,"model.1",2,1))
    x : F.Tensor() <- layer(trace,"layer_02",c3(x,"model.2",1n,True{}))
    x : F.Tensor() <- layer(trace,"layer_03",conv(x,"model.3",2,1))
    x : F.Tensor() <- layer(trace,"layer_04",c3(x,"model.4",2n,True{}))
    x : F.Tensor() <- F.keep("c4",x)
    x : F.Tensor() <- layer(trace,"layer_05",conv(x,"model.5",2,1))
    x : F.Tensor() <- layer(trace,"layer_06",c3(x,"model.6",3n,True{}))
    x : F.Tensor() <- F.keep("c6",x)
    x : F.Tensor() <- layer(trace,"layer_07",conv(x,"model.7",2,1))
    x : F.Tensor() <- layer(trace,"layer_08",c3(x,"model.8",1n,True{}))
    x : F.Tensor() <- layer(trace,"layer_09",sppf(x,"model.9"))
    # top-down path
    x : F.Tensor() <- layer(trace,"layer_10",conv(x,"model.10",1,0))
    x : F.Tensor() <- F.keep("c10",x)
    x : F.Tensor() <- layer(trace,"layer_11",F.Model.pure(F.Tensor(),F.upsample_nearest(x,2)))
    x : F.Tensor() <- layer(trace,"layer_12",joined(x,"c6"))
    x : F.Tensor() <- layer(trace,"layer_13",c3(x,"model.13",1n,False{}))
    x : F.Tensor() <- layer(trace,"layer_14",conv(x,"model.14",1,0))
    x : F.Tensor() <- F.keep("c14",x)
    x : F.Tensor() <- layer(trace,"layer_15",F.Model.pure(F.Tensor(),F.upsample_nearest(x,2)))
    x : F.Tensor() <- layer(trace,"layer_16",joined(x,"c4"))
    x : F.Tensor() <- layer(trace,"layer_17",c3(x,"model.17",1n,False{}))
    x : F.Tensor() <- F.keep("p3",x)
    # bottom-up path
    x : F.Tensor() <- layer(trace,"layer_18",conv(x,"model.18",2,1))
    x : F.Tensor() <- layer(trace,"layer_19",joined(x,"c14"))
    x : F.Tensor() <- layer(trace,"layer_20",c3(x,"model.20",1n,False{}))
    x : F.Tensor() <- F.keep("p4",x)
    x : F.Tensor() <- layer(trace,"layer_21",conv(x,"model.21",2,1))
    x : F.Tensor() <- layer(trace,"layer_22",joined(x,"c10"))
    x : F.Tensor() <- layer(trace,"layer_23",c3(x,"model.23",1n,False{}))
    detect(x,size,trace)

# The forward pass has run when its result is known to be ready or failed.
def checked(pair: F.Tensor() & F.Weights()) -> IO(F.Tensor() & F.Weights()):
  (prediction,kept) = pair
  do IO<F.Tensor() & F.Weights()>:
    prediction : F.Tensor() <- F.expect(prediction)
    IO.pure(F.Tensor() & F.Weights(),(prediction,kept))

def text(result: Result<&1,&1,U32 & String,String>,fallback: String) -> String:
  match result:
    case Done{value}: value
    case Fail{error}: fallback

def set(result: Result<&1,&1,U32 & String,String>) -> Bool:
  match result:
    case Done{value}: True{}
    case Fail{error}: False{}

def variable(+name: String,fallback: String) -> IO(String):
  do IO<String>:
    result : Result<&1,&1,U32 & String,String> <- IO.get_env(name)
    IO.pure(String,text(result,fallback))

def flag(+name: String) -> IO(Bool):
  do IO<Bool>:
    result : Result<&1,&1,U32 & String,String> <- IO.get_env(name)
    IO.pure(Bool,set(result))

def number(text: Maybe<&2,U32>) -> Nat:
  match text:
    case Some{value}: U32.to_nat(value)
    case None{}: 0n

def count(+name: String) -> IO(Nat):
  do IO<Nat>:
    result : Result<&1,&1,U32 & String,String> <- IO.get_env(name)
    IO.pure(Nat,number(U32.read(text(result,"0"))))

def pixels(text: Maybe<&2,U32>) -> U32:
  match text:
    case Some{value}: value
    case None{}: 640

def side(+name: String) -> IO(U32):
  do IO<U32>:
    result : Result<&1,&1,U32 & String,String> <- IO.get_env(name)
    IO.pure(U32,pixels(U32.read(text(result,"640"))))

type Inputs is Type:
  Inputs{image: F.Tensor(),weights: F.Weights()}

# One timed forward pass.
def forward(image: F.Tensor(),weights: F.Weights(),+size: U32,+workers: U32,trace: Bool) -> IO(F.Tensor() & F.Weights()):
  do IO<F.Tensor() & F.Weights()>:
    start : Nat <- IO.now()
    result : F.Tensor() & F.Weights() <- checked(F.run_keeping(F.Tensor(),network(F.with_workers(image,workers),size,trace),weights))
    end : Nat <- IO.now()
    IO.print("forward " ++ Nat.show(Nat.sub(end,start)) ++ " ms")
    IO.pure(F.Tensor() & F.Weights(),result)

# A forward pass on copies; the image and the weights stay for the next one.
def practice(images: F.Tensor() & F.Tensor(),stores: F.Weights() & F.Weights(),+size: U32,+workers: U32) -> IO(Inputs):
  (image,next_image) = images
  (weights,next_weights) = stores
  do IO<Inputs>:
    result : F.Tensor() & F.Weights() <- forward(image,weights,size,workers,False{})
    IO.pure(Inputs,Inputs{next_image,next_weights})

def repeated(extra: Nat,ready: Inputs,+size: U32,+workers: U32,trace: Bool) -> IO(F.Tensor() & F.Weights()):
  match extra ready:
    case 0n Inputs{image,weights}: forward(image,weights,size,workers,trace)
    case 1n+rest Inputs{image,weights}:
      do IO<F.Tensor() & F.Weights()>:
        next : Inputs <- practice(F.copy(image),F.copy_weights(weights),size,workers)
        repeated(rest,next,size,workers,trace)

def finish(pair: F.Tensor() & F.Weights(),+folder: String) -> IO(Unit):
  (prediction,kept) = pair
  do IO<Unit>:
    saved : F.Tensor() <- F.save_npy(folder ++ "/pred.npy",prediction)
    layers : F.Weights() <- F.save_weights(folder,kept)
    IO.pure(Unit,Unit{})

def begin(+size: U32,input: String,+output: String,trace: Bool,extra: Nat) -> IO(Unit):
  do IO<Unit>:
    workers : U32 <- IO.thread_count()
    weights : F.Weights() <- F.load_weights("out/weights/named")
    image : F.Tensor() <- F.load_raw(input,[1,3,size,size])
    result : F.Tensor() & F.Weights() <- repeated(extra,Inputs{image,weights},size,workers,trace)
    finish(result,output)

# YOLO_SIZE: a multiple of 32. YOLO_INPUT: a raw [1,3,size,size] image. YOLO_OUTPUT: the folder for pred.npy.
# YOLO_LAYERS: when set, every layer output is saved there too.
# YOLO_EXTRA: forward passes to run before the one that is saved.
def main() -> IO(Unit):
  do IO<Unit>:
    input : String <- variable("YOLO_INPUT","out/results/bus_640/input.bin")
    output : String <- variable("YOLO_OUTPUT","out/results/bus_640/native")
    trace : Bool <- flag("YOLO_LAYERS")
    extra : Nat <- count("YOLO_EXTRA")
    size : U32 <- side("YOLO_SIZE")
    begin(size,input,output,trace,extra)
