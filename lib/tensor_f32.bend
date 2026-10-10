# The everyday F32 tensor module: import this one module and call F.add,
# F.concat, F.check, ... without element-type arguments. Every function is a
# direct specialization of the generic tensor modules; Base determines operand
# order, rounding, NaN and signed-zero behavior.
import Base
import ./storage_buffer.bend as Buffers
import ./math_activation.bend as Activations
import ./math_fp32.bend as Arithmetic
import ./math_compare.bend as Compare
import ./tensor.bend as Tensors
import ./tensor_operations.bend as Operations
import ./tensor_lanes.bend as Lanes
import ./tensor_select.bend as Select
import ./tensor_layout.bend as Layout
import ./tensor_window.bend as Window
import ./tensor_convolution.bend as TensorConvolution
import ./tensor_file.bend as Files
import ./tensor_model.bend as Models

def Tensor() -> Type:
  Tensors.Tensor<F32>

def Problem() -> Data:
  Tensors.Problem

# Storage exchanged with foreign I/O; see from_storage and into_storage.
def Storage() -> Type:
  Buffers.Buffer<F32>

# Construction

def full(+shape: List<&2,U32>,+value: F32) -> Tensor():
  Tensors.full(~F32,shape,value)

def zeros(+shape: List<&2,U32>) -> Tensor():
  Tensors.full(~F32,shape,0.0)

def ones(+shape: List<&2,U32>) -> Tensor():
  full(shape,1.0)

def from_list(+shape: List<&2,U32>,+values: List<&2,F32>) -> Tensor():
  Tensors.from_list(~F32,shape,values,0.0)

def from_storage(+shape: List<&2,U32>,storage: Storage()) -> Tensor():
  Tensors.from_storage(~F32,shape,storage,0.0)

# Raw little-endian FP32 files: the values of the shape in row-major order.
# A file that cannot be read, or does not match the shape, gives a failed tensor.
def load_raw(+path: String,+shape: List<&2,U32>) -> IO(Tensor()):
  Files.load(path,shape)

def save_raw(+path: String,tensor: Tensor()) -> IO(Tensor()):
  Files.save(path,tensor)

# NumPy .npy files of FP32 values; the file carries the shape.
def load_npy(+path: String) -> IO(Tensor()):
  Files.load_npy(path)

def save_npy(+path: String,tensor: Tensor()) -> IO(Tensor()):
  Files.save_npy(path,tensor)

# Models: tensors by name, loaded once. In `do Model<T>:` a step takes a tensor
# from the weights with take, or puts a copy there with keep for a later take.
def Weights() -> Type:
  Models.Store

def Model(A: Type) -> Type:
  Models.Model(A)

def Model.pure(-A: Type,value: A) -> Model(A):
  Models.Model.pure(A,value)

def Model.bind(-A: Type,-B: Type,step: Model(A),next: A -> Model(B)) -> Model(B):
  Models.Model.bind(A,B,step,next)

def take(+name: String) -> Model(Tensor()):
  Models.take(name)

def keep(+name: String,tensor: Tensor()) -> Model(Tensor()):
  Models.keep(name,tensor)

# folder/names.txt lists one name per line; folder/<name>.npy holds its tensor.
def load_weights(+folder: String) -> IO(Weights()):
  Models.load(folder)

def copy_weights(weights: Weights()) -> Weights() & Weights():
  Models.copy(weights)

def run(-A: Type,model: Model(A),weights: Weights()) -> A:
  Models.result(A,model,weights)

# The result with what remains of the weights, kept tensors included.
def run_keeping(-A: Type,model: Model(A),weights: Weights()) -> A & Weights():
  Models.run(A,model,weights)

def save_weights(+folder: String,weights: Weights()) -> IO(Weights()):
  Models.save(folder,weights)

# Execution context: operations may use count workers, like a tensor's device.
# Read the runtime's pool once with IO.thread_count(); results keep the count.
def with_workers(tensor: Tensor(),+count: U32) -> Tensor():
  Tensors.with_workers(~F32,tensor,count)

def workers(tensor: Tensor()) -> Tensor() & U32:
  Tensors.workers(~F32,tensor)

# Observation and exits

def shape(tensor: Tensor()) -> Tensor() & List<&2,U32>:
  Tensors.shape(~F32,tensor)

def check(tensor: Tensor()) -> Result<&1,&1,Tensor() & Problem(),Tensor()>:
  Tensors.check(~F32,tensor)

def describe(problem: Problem()) -> String:
  Tensors.describe(problem)

def expect(tensor: Tensor()) -> IO(Tensor()):
  Tensors.expect(~F32,tensor)

def to_list(tensor: Tensor()) -> Result<&2,&2,Problem(),List<&2,F32>>:
  Tensors.to_list(~F32,tensor)

def item(tensor: Tensor()) -> Result<&2,&2,Problem(),F32>:
  Tensors.item(~F32,tensor)

# Reads with a fallback for a failed tensor, for callers that do not inspect
# the problem.
def to_list_or(tensor: Tensor(),fallback: List<&2,F32>) -> List<&2,F32>:
  Tensors.value_or(~List<&2,F32>,Tensors.to_list(~F32,tensor),fallback)

def item_or(tensor: Tensor(),fallback: F32) -> F32:
  Tensors.value_or(~F32,Tensors.item(~F32,tensor),fallback)

def into_storage(tensor: Tensor()) -> Result<&2,&1,Problem(),Storage()>:
  Tensors.into_storage(~F32,tensor)

def expect_storage(tensor: Tensor()) -> IO(Storage()):
  Tensors.expect_storage(~F32,tensor)

# Operations consume their tensor arguments and return one result. Keep a
# tensor by copying it first; copy uses then, and shape uses then_shape.

def then(pair: Tensor() & Tensor(),body: Tensor() -> Tensor() -> Tensor()) -> Tensor():
  Tensors.then(~Tensor(),~Tensor(),~Tensor(),pair,body)

def then_shape(pair: Tensor() & List<&2,U32>,body: Tensor() -> List<&2,U32> -> Tensor()) -> Tensor():
  Tensors.then(~Tensor(),~List<&2,U32>,~Tensor(),pair,body)

def copy(tensor: Tensor()) -> Tensor() & Tensor():
  Layout.copy(~F32,tensor)

# Binary elementwise operations reuse the left tensor's storage. The number
# after the name is the operation's scalar work for the band schedule.
def add(left: Tensor(),right: Tensor()) -> Tensor():
  Tensors.combine(~F32,~F32.add,"add",1.0,left,right)

def sub(left: Tensor(),right: Tensor()) -> Tensor():
  Tensors.combine(~F32,~F32.sub,"sub",1.0,left,right)

def mul(left: Tensor(),right: Tensor()) -> Tensor():
  Tensors.combine(~F32,~F32.mul,"mul",1.0,left,right)

# Base division: a zero divisor gives an infinity or NaN, never a failure.
def div(left: Tensor(),right: Tensor()) -> Tensor():
  Tensors.combine(~F32,~F32.div,"div",1.0,left,right)

def maximum(left: Tensor(),right: Tensor()) -> Tensor():
  Tensors.combine(~F32,~F32.max,"maximum",2.0,left,right)

def minimum(left: Tensor(),right: Tensor()) -> Tensor():
  Tensors.combine(~F32,~F32.min,"minimum",2.0,left,right)

def add_scalar(tensor: Tensor(),+value: F32) -> Tensor():
  Tensors.combine_scalar(~F32,~F32.add,"add_scalar",1.0,tensor,value)

def sub_scalar(tensor: Tensor(),+value: F32) -> Tensor():
  Tensors.combine_scalar(~F32,~F32.sub,"sub_scalar",1.0,tensor,value)

def mul_scalar(tensor: Tensor(),+value: F32) -> Tensor():
  Tensors.combine_scalar(~F32,~F32.mul,"mul_scalar",1.0,tensor,value)

def div_scalar(tensor: Tensor(),+value: F32) -> Tensor():
  Tensors.combine_scalar(~F32,~F32.div,"div_scalar",1.0,tensor,value)

def less(left: Tensor(),right: Tensor()) -> Tensor():
  Tensors.combine_with(~F32,~Lanes.combine(~Compare.less_elements),"less",2.625,left,right)

def less_equal(left: Tensor(),right: Tensor()) -> Tensor():
  Tensors.combine_with(~F32,~Lanes.combine(~Compare.less_equal_elements),"less_equal",2.625,left,right)

def greater(left: Tensor(),right: Tensor()) -> Tensor():
  Tensors.combine_with(~F32,~Lanes.combine(~Compare.greater_elements),"greater",2.625,left,right)

def greater_equal(left: Tensor(),right: Tensor()) -> Tensor():
  Tensors.combine_with(~F32,~Lanes.combine(~Compare.greater_equal_elements),"greater_equal",2.625,left,right)

def equal(left: Tensor(),right: Tensor()) -> Tensor():
  Tensors.combine_with(~F32,~Lanes.combine(~Compare.equal_elements),"equal",1.375,left,right)

def not_equal(left: Tensor(),right: Tensor()) -> Tensor():
  Tensors.combine_with(~F32,~Lanes.combine(~Compare.not_equal_elements),"not_equal",1.375,left,right)

# Scalar comparisons share the eight-lane map and preserve the tensor layout.
# Costs count AVX2 arithmetic, comparison, selection and value-bit instructions
# in one eight-element round, divided by eight; moves and addressing are excluded.
def less_scalar(tensor: Tensor(),+value: F32) -> Tensor():
  Tensors.apply_context(~F32,~F32,~Operations.lanes(~F32,~Compare.less),value,"less_scalar",2.75,tensor)

def less_equal_scalar(tensor: Tensor(),+value: F32) -> Tensor():
  Tensors.apply_context(~F32,~F32,~Operations.lanes(~F32,~Compare.less_equal),value,"less_equal_scalar",2.625,tensor)

def greater_scalar(tensor: Tensor(),+value: F32) -> Tensor():
  Tensors.apply_context(~F32,~F32,~Operations.lanes(~F32,~Compare.greater),value,"greater_scalar",2.625,tensor)

def greater_equal_scalar(tensor: Tensor(),+value: F32) -> Tensor():
  Tensors.apply_context(~F32,~F32,~Operations.lanes(~F32,~Compare.greater_equal),value,"greater_equal_scalar",2.75,tensor)

def equal_scalar(tensor: Tensor(),+value: F32) -> Tensor():
  Tensors.apply_context(~F32,~F32,~Operations.lanes(~F32,~Compare.equal),value,"equal_scalar",1.5,tensor)

def not_equal_scalar(tensor: Tensor(),+value: F32) -> Tensor():
  Tensors.apply_context(~F32,~F32,~Operations.lanes(~F32,~Compare.not_equal),value,"not_equal_scalar",1.5,tensor)

def select(mask: Tensor(),yes: Tensor(),no: Tensor()) -> Tensor():
  Select.select(mask,yes,no)

def clamp(tensor: Tensor(),+low: F32,+high: F32) -> Tensor():
  Tensors.apply_context(~F32,~Compare.Bounds,~Operations.lanes(~Compare.Bounds,~Compare.clamp),Compare.Bounds{low,high},"clamp",0.5,tensor)

# Activations: Act.silu() and the other values of lib/math_activation.bend
# carry their own planner cost. activate applies one to a tensor; relu, silu,
# ... below are its shorthands. Custom carries only a cost; its function
# reaches conv2d_custom or apply, and activate fails on it.
def activate(tensor: Tensor(),+activation: Activations.Activation) -> Tensor():
  match activation:
    case Activations.Identity{}: tensor
    case Activations.ReLU{}: Tensors.apply(~F32,~Operations.mapped(~F32,~Activations.relu_value),"relu",Activations.operations(Activations.ReLU{}),tensor)
    case Activations.ReLU6{}: Tensors.apply(~F32,~Operations.mapped(~F32,~Activations.relu6_value),"relu6",Activations.operations(Activations.ReLU6{}),tensor)
    case Activations.LeakyReLU{+slope}: Tensors.combine_scalar(~F32,~Activations.leaky_relu_value,"leaky_relu",Activations.operations(Activations.LeakyReLU{slope}),tensor,slope)
    case Activations.Sigmoid{}: Tensors.apply(~F32,~Operations.mapped_lanes(~Activations.sigmoid_value),"sigmoid",Activations.operations(Activations.Sigmoid{}),tensor)
    case Activations.SiLU{}: Tensors.apply(~F32,~Operations.mapped_lanes(~Activations.silu_value),"silu",Activations.operations(Activations.SiLU{}),tensor)
    case Activations.Tanh{}: Tensors.apply(~F32,~Operations.mapped_lanes(~Activations.tanh_value),"tanh",Activations.operations(Activations.Tanh{}),tensor)
    case Activations.HardSigmoid{}: Tensors.apply(~F32,~Operations.mapped(~F32,~Activations.hard_sigmoid_value),"hard_sigmoid",Activations.operations(Activations.HardSigmoid{}),tensor)
    case Activations.HardSwish{}: Tensors.apply(~F32,~Operations.mapped(~F32,~Activations.hard_swish_value),"hard_swish",Activations.operations(Activations.HardSwish{}),tensor)
    case Activations.GELUTanh{}: Tensors.apply(~F32,~Operations.mapped_lanes(~Activations.gelu_tanh_value),"gelu_tanh",Activations.operations(Activations.GELUTanh{}),tensor)
    case Activations.Custom{count}: Tensors.rejected(~F32,"activate",tensor)

def relu(tensor: Tensor()) -> Tensor():
  activate(tensor,Activations.relu())

def relu6(tensor: Tensor()) -> Tensor():
  activate(tensor,Activations.relu6())

def leaky_relu(tensor: Tensor(),slope: F32) -> Tensor():
  activate(tensor,Activations.leaky_relu(slope))

def sigmoid(tensor: Tensor()) -> Tensor():
  activate(tensor,Activations.sigmoid())

def silu(tensor: Tensor()) -> Tensor():
  activate(tensor,Activations.silu())

def tanh(tensor: Tensor()) -> Tensor():
  activate(tensor,Activations.tanh())

def hard_sigmoid(tensor: Tensor()) -> Tensor():
  activate(tensor,Activations.hard_sigmoid())

def hard_swish(tensor: Tensor()) -> Tensor():
  activate(tensor,Activations.hard_swish())

# PyTorch's gelu(approximate='tanh').
def gelu_tanh(tensor: Tensor()) -> Tensor():
  activate(tensor,Activations.gelu_tanh())

# Ordered convolution over CHW or NCHW; the activation runs on each finished
# leaf while it is in cache.
def conv2d(input: Tensor(),weights: Tensor(),bias: Tensor(),stride: U32,padding: U32,activation: Activations.Activation) -> Tensor():
  TensorConvolution.conv2d(~Activations.identity_value,activation,input,weights,bias,stride,padding)

# conv2d with a custom activation of the given scalar operations per element.
def conv2d_custom(~activation: F32 -> F32,operations: F32,input: Tensor(),weights: Tensor(),bias: Tensor(),stride: U32,padding: U32) -> Tensor():
  TensorConvolution.conv2d(~activation,Activations.Custom{operations},input,weights,bias,stride,padding)

# Custom elementwise functions are passed as direct templates, never records,
# with their scalar operations per element.
def apply(~operation: F32 -> F32,operations: F32,tensor: Tensor()) -> Tensor():
  Tensors.apply(~F32,~Operations.mapped(~F32,~operation),"apply",operations,tensor)

# The function also receives the element's row-major index.
def apply_indexed(~operation: U32 -> F32 -> F32,operations: F32,tensor: Tensor()) -> Tensor():
  Tensors.apply_indexed(~F32,~operation,operations,tensor)

# start, start + 1, ... as a rank-1 tensor of count elements.
def arange(+start: F32,+count: U32) -> Tensor():
  add_scalar(Tensors.apply_indexed(~F32,~(index => value => U32.to_f32(index)),1.0,zeros([count])),start)

# Reductions return a rank-0 tensor, so failures propagate like any result.

def sum(tensor: Tensor()) -> Tensor():
  Tensors.reduce_all(~F32,~F32.add,"sum",tensor,0.0)

def max(tensor: Tensor()) -> Tensor():
  Tensors.reduce_all(~F32,~F32.max,"max",tensor,F32.div(F32.neg(1.0),0.0))

def min(tensor: Tensor()) -> Tensor():
  Tensors.reduce_all(~F32,~F32.min,"min",tensor,F32.div(1.0,0.0))

# sum divided by the number of elements; an empty tensor gives NaN.
def mean(tensor: Tensor()) -> Tensor():
  then_shape(shape(tensor),tensor => extents =>
    Tensors.combine_scalar(~F32,~F32.div,"mean",1.0,Tensors.reduce_all(~F32,~F32.add,"mean",tensor,0.0),
      U32.to_f32(Tensors.count_or_zero(extents))))

# Layout: results are new dense tensors, except reshape, which relabels.

def reshape(tensor: Tensor(),+shape: List<&2,U32>) -> Tensor():
  Layout.reshape(~F32,tensor,shape)

# The result's axis i is the argument's axis order[i].
def transpose(tensor: Tensor(),order: List<&2,U32>) -> Tensor():
  Layout.transpose(~F32,tensor,order)

# count elements from start along axis, every step-th one.
def slice(tensor: Tensor(),axis: U32,start: U32,count: U32,step: U32) -> Tensor():
  Layout.slice(~F32,tensor,axis,start,count,step)

# target with part written at start along axis.
def set_slice(target: Tensor(),axis: U32,start: U32,part: Tensor()) -> Tensor():
  Layout.set_slice(~F32,target,axis,start,part)

def Parts() -> Type:
  Tensors.Parts<F32>

# Build the parts of concat: F.cons(a, F.cons(b, F.single(c))).
def single(tensor: Tensor()) -> Parts():
  Tensors.Single{tensor}

def cons(tensor: Tensor(),rest: Parts()) -> Parts():
  Tensors.Cons{tensor,rest}

def concat(parts: Parts(),+axis: U32) -> Tensor():
  Layout.concat(~F32,parts,axis)

# Repeats each element of the last two axes scale times in each direction.
def upsample_nearest(tensor: Tensor(),+scale: U32) -> Tensor():
  Layout.upsample_nearest(~F32,tensor,scale)

# Over the last two axes; positions outside the input count as -inf.
def max_pool2d_with(tensor: Tensor(),+kernel: U32,+stride: U32,+padding: U32) -> Tensor():
  Window.pool2d(~F32,~F32.max,"max_pool2d",F32.div(F32.neg(1.0),0.0),tensor,kernel,stride,padding)

# Windows of kernel by kernel that do not overlap: stride kernel, padding 0.
def max_pool2d(tensor: Tensor(),+kernel: U32) -> Tensor():
  max_pool2d_with(tensor,kernel,kernel,0)

# Broadcasting and axis operations.

# The tensor repeated to shape under NumPy broadcasting; add, sub, mul, max
# and min broadcast their operands in the same way.
def broadcast_to(tensor: Tensor(),+shape: List<&2,U32>) -> Tensor():
  Layout.broadcast_to(~F32,tensor,shape)

# Reductions along one axis remove that axis. Each result element takes the
# elements of the axis in increasing index, starting from 0, -inf or +inf.
def sum_axis(tensor: Tensor(),+axis: U32) -> Tensor():
  Layout.reduce_axis(~F32,~F32.add,"sum_axis",0.0,tensor,axis)

def max_axis(tensor: Tensor(),+axis: U32) -> Tensor():
  Layout.reduce_axis(~F32,~F32.max,"max_axis",F32.div(F32.neg(1.0),0.0),tensor,axis)

def min_axis(tensor: Tensor(),+axis: U32) -> Tensor():
  Layout.reduce_axis(~F32,~F32.min,"min_axis",F32.div(1.0,0.0),tensor,axis)

# Elementwise functions share the eight-value traversal used by activations.
def exp(tensor: Tensor()) -> Tensor():
  Tensors.apply(~F32,~Operations.mapped_lanes(~Arithmetic.exp),"exp",4.5,tensor)

def log(tensor: Tensor()) -> Tensor():
  Tensors.apply(~F32,~Operations.mapped_lanes(~Arithmetic.log),"log",8.25,tensor)

# Keep a singleton reduction axis for broadcasting. Both reductions use the
# existing ordered fold; reduction gathers row bands to a dense owner.
def softmax(tensor: Tensor(),+axis: U32) -> Tensor():
  then_shape(shape(tensor),tensor => extents =>
    +unit = Layout.with_extent(extents,U32.to_nat(axis),1)
    then(copy(tensor),values => maximum_input =>
      then(copy(exp(sub(values,reshape(Layout.reduce_axis(~F32,~F32.max,"softmax",F32.div(F32.neg(1.0),0.0),maximum_input,axis),unit)))),
        values => sum_input => div(values,reshape(Layout.reduce_axis(~F32,~F32.add,"softmax",0.0,sum_input,axis),unit)))))

# sum_axis divided by the extent of the axis.
def mean_axis(tensor: Tensor(),+axis: U32) -> Tensor():
  then_shape(shape(tensor),tensor => extents =>
    Tensors.combine_scalar(~F32,~F32.div,"mean_axis",1.0,Layout.reduce_axis(~F32,~F32.add,"mean_axis",0.0,tensor,axis),
      U32.to_f32(Layout.extent_at(extents,U32.to_nat(axis)))))

# before[i] new elements in front of axis i and after[i] behind it, all equal
# to value.
def pad(tensor: Tensor(),+before: List<&2,U32>,+after: List<&2,U32>,+value: F32) -> Tensor():
  Layout.pad(~F32,Layout.Constant{value},tensor,before,after)

# The same, each new element repeating the nearest element of the tensor.
def pad_edge(tensor: Tensor(),+before: List<&2,U32>,+after: List<&2,U32>) -> Tensor():
  Layout.pad(~F32,Layout.Edge{},tensor,before,after)
