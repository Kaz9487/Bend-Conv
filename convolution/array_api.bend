import ./shape.bend as ConvShape
import Base
import ../lib/machine_limits.bend as Machine
import ../lib/default_machine_limits.bend as Default
import ../lib/storage_buffer.bend as Storage
import ./specification.bend as ConvSpec
import ./inference.bend as NativeConv
import ../lib/kernel_shape.bend as KernelShape

type Outcome is Type:
  Rejected{}
  Unsupported{}
  Accepted{height: Nat, width: Nat, values: Array<F32>}

law valid:
  for shape: ConvSpec.Shape
  for input_length: Nat
  for weight_length: Nat
  for bias_length: Nat
  Bool

def valid(shape,input_length,weight_length,bias_length):
  ConvSpec.Shape{ci,co,height,width,kernel,stride,padding} = shape
  KernelShape.valid(KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding},input_length,weight_length,bias_length)

def supported_with_limits(+limits: Machine.Limits,shape: ConvSpec.Shape) -> Bool:
  ConvSpec.Shape{+input_channel_count,+output_channel_count,+height,+width,+kernel,stride,padding} = shape
  ConvShape.supports_packed(limits,shape,Nat.mul(input_channel_count,Nat.mul(height,width)),Nat.mul(output_channel_count,Nat.mul(input_channel_count,Nat.mul(kernel,kernel))))

def supported(shape: ConvSpec.Shape) -> Bool:
  supported_with_limits(Default.limits(),shape)

def execute(ok: Bool, +shape: ConvSpec.Shape, strategy: NativeConv.Strategy, input: Array<F32>, weights: Array<F32>, bias: Array<F32>) -> Outcome:
  match ok:
    case False{}:
      Rejected{}
    case True{}:
      ConvSpec.Shape{input_channels,output_channels,+height,+width,+kernel_size,+stride,+padding} = shape
      +output_height = ConvSpec.axis_size(height,kernel_size,stride,padding)
      +output_width = ConvSpec.axis_size(width,kernel_size,stride,padding)
      Accepted{output_height,output_width,NativeConv.execute(strategy,False{},ConvShape.from_spec(shape),input,weights,bias)}

def capacities(shape: ConvSpec.Shape, strategy: NativeConv.Strategy, input_length: Nat, weight_length: Nat, bias_length: Nat, input_sized: Array<F32> & U32, weights_sized: Array<F32> & U32,
  bias_sized: Array<F32> & U32) -> Outcome:
  (input,input_capacity) = input_sized
  (weights,weight_capacity) = weights_sized
  (bias,bias_capacity) = bias_sized
  execute(Nat.is_le(input_length,U32.to_nat(input_capacity)) && Nat.is_le(weight_length,U32.to_nat(weight_capacity)) && Nat.is_le(bias_length,U32.to_nat(bias_capacity)),shape,strategy,input,weights,bias)

def machine(ok: Bool, shape: ConvSpec.Shape, strategy: NativeConv.Strategy, input: Storage.Buffer<F32>, weights: Storage.Buffer<F32>, bias: Storage.Buffer<F32>) -> Outcome:
  match ok:
    case False{}:
      Unsupported{}
    case True{}:
      Storage.Buffer{input_storage,input_length,input_certificate} = input
      Storage.Buffer{weight_storage,weight_length,weight_certificate} = weights
      Storage.Buffer{bias_storage,bias_length,bias_certificate} = bias
      capacities(shape,strategy,input_length,weight_length,bias_length,Array.size(F32,input_storage),Array.size(F32,weight_storage),Array.size(F32,bias_storage))

def checked(ok: Bool, +shape: ConvSpec.Shape, strategy: NativeConv.Strategy, input: Storage.Buffer<F32>, weights: Storage.Buffer<F32>, bias: Storage.Buffer<F32>) -> Outcome:
  match ok:
    case False{}:
      Rejected{}
    case True{}:
      machine(supported(shape),shape,strategy,input,weights,bias)

def infer(+shape: ConvSpec.Shape, strategy: NativeConv.Strategy, input: Storage.Buffer<F32>, weights: Storage.Buffer<F32>, bias: Storage.Buffer<F32>) -> Outcome:
  Storage.Buffer{input_storage,+input_length,input_certificate} = input
  Storage.Buffer{weight_storage,+weight_length,weight_certificate} = weights
  Storage.Buffer{bias_storage,+bias_length,bias_certificate} = bias
  checked(valid(shape,input_length,weight_length,bias_length),shape,strategy,Storage.Buffer{input_storage,input_length,input_certificate},Storage.Buffer{weight_storage,weight_length,weight_certificate},Storage.Buffer{bias_storage,bias_length,bias_certificate})
