import ./shape.bend as ConvShape
import ../lib/machine_limits.bend as Machine
import ../lib/default_machine_limits.bend as Default
# Scalar inference is proved equivalent to the ordered matrix specification in
# proofs/scalar_inference_refinement.bend, including fallback paths.
import Base
import ../lib/storage_buffer.bend as Storage
import ./specification.bend as ConvSpec
import ./reference.bend as Reference
import ../lib/math_fp32.bend as Fp32
import ../lib/traversal_output.bend as OutputTraversal

def pack(+values: List<&2,F32>) -> Array<F32>:
  Storage.write_list(F32,values,0,Array.new(F32,Storage.allocation_depth(U32.from_nat(ConvSpec.length(values))),0.0))


def sample(~Storage: Type,~read: Storage -> U32 -> U32 -> Storage & F32,
  inside: Bool,storage: Storage,index: U32,count: U32) -> Storage & F32:
  match inside:
    case False{}: (storage,0.0)
    case True{}: read(storage,index,count)

type SampleContext is Data:
  SampleContext{shape: ConvShape.Shape,kernel_row: U32,input_channel: U32,output_channel: U32,output_row: U32,output_column: U32}

type SampleAddress is Data:
  SampleAddress{inside: Bool,index: U32,length: U32}

def input_address(context: SampleContext,kernel_column: U32) -> SampleAddress:
  SampleContext{shape,kernel_row_index,input_channel_index,output_channel_index,output_row_index,output_column_index} = context
  ConvShape.Shape{+input_channel_count,output_channel_count,+height,+width,kernel,+stride,+padding,output_height,output_width} = shape
  +padded_row = (output_row_index * stride + kernel_row_index : U32)
  +padded_column = (output_column_index * stride + kernel_column : U32)
  inside = U32.is_le(padding,padded_row) && U32.is_lt(padded_row,(padding + height : U32)) && U32.is_le(padding,padded_column) && U32.is_lt(padded_column,(padding + width : U32))
  SampleAddress{inside,(input_channel_index * height * width + (padded_row - padding) * width + padded_column - padding : U32),(input_channel_count * height * width : U32)}

def read_address(~Storage: Type,~read: Storage -> U32 -> U32 -> Storage & F32,
  input: Storage,address: SampleAddress) -> Storage & F32:
  SampleAddress{inside,index,length} = address
  sample(~Storage,~read,inside,input,index,length)

def read_input(~Storage: Type,~read: Storage -> U32 -> U32 -> Storage & F32,
  input: Storage,context: SampleContext,kernel_column: U32) -> Storage & F32:
  read_address(~Storage,~read,input,input_address(context,kernel_column))

# Direct sampling is a policy of the common output driver. K is traversed in
# input-channel, kernel-row, kernel-column order; every output starts at +0.
type DirectContext is Data:
  DirectContext{shape: ConvShape.Shape,index: U32,channel: U32,row: U32,column: U32}

def direct_job(shape: ConvShape.Shape,index: U32) -> OutputTraversal.Job<DirectContext,F32>:
  ConvShape.Shape{+input_channel_count,output_channel_count,input_height,input_width,+kernel,stride,padding,+output_height,+output_width} = shape
  +index = index
  OutputTraversal.Job{DirectContext{ConvShape.Shape{input_channel_count,output_channel_count,input_height,input_width,kernel,stride,padding,output_height,output_width},index,(index / (output_height * output_width) : U32),(index % (output_height * output_width) / output_width : U32),(index % output_width : U32)},
    U32.to_nat((input_channel_count * kernel * kernel : U32)),0,0,0.0}

def direct_address(context: DirectContext,reduction: U32) -> SampleAddress:
  DirectContext{shape,index,channel,row,column} = context
  ConvShape.Shape{input_channel_count,output_channel_count,input_height,input_width,+kernel,stride,padding,output_height,output_width} = shape
  +reduction = reduction
  input_address(SampleContext{ConvShape.Shape{input_channel_count,output_channel_count,input_height,input_width,kernel,stride,padding,output_height,output_width},(reduction / kernel % kernel : U32),(reduction / (kernel * kernel) : U32),channel,row,column},
    (reduction % kernel : U32))

def direct_input(~Storage: Type,~read: Storage -> U32 -> U32 -> Storage & F32,
  input: Storage,context: DirectContext,reduction: U32) -> Storage & F32:
  read_address(~Storage,~read,input,direct_address(context,reduction))

def weight_address(context: DirectContext,reduction: U32) -> SampleAddress:
  DirectContext{shape,index,channel,row,column} = context
  ConvShape.Shape{+input_channel_count,output_channel_count,input_height,input_width,+kernel,stride,padding,output_height,output_width} = shape
  +reduction_length = (input_channel_count * kernel * kernel : U32)
  SampleAddress{True{},(channel * reduction_length + reduction : U32),(output_channel_count * reduction_length : U32)}

def direct_weight(~Storage: Type,~read: Storage -> U32 -> U32 -> Storage & F32,
  weights: Storage,context: DirectContext,reduction: U32) -> Storage & F32:
  read_address(~Storage,~read,weights,weight_address(context,reduction))

def direct_update(context: DirectContext,accumulator: F32,input: F32,weight: F32) -> F32:
  Fp32.multiply_add(accumulator,input,weight)

def direct_written(~Storage: Type,~write: Storage -> U32 -> F32 -> Storage,
  output: Storage,index: U32,accumulator: F32,read: Storage & F32) -> Storage & Storage:
  (bias,value) = read
  (bias,write(output,index,(accumulator + value : F32)))

def direct_write(~Storage: Type,~read: Storage -> U32 -> U32 -> Storage & F32,~write: Storage -> U32 -> F32 -> Storage,
  context: DirectContext,accumulator: F32,bias: Storage,output: Storage) -> Storage & Storage:
  DirectContext{shape,index,channel,row,column} = context
  ConvShape.Shape{input_channel_count,output_channel_count,input_height,input_width,kernel,stride,padding,output_height,output_width} = shape
  direct_written(~Storage,~write,output,index,accumulator,read(bias,channel,output_channel_count))

def run(+shape: ConvShape.Shape,input: Array<F32>,weights: Array<F32>,bias: Array<F32>) -> Array<F32>:
  ConvShape.Shape{input_channel_count,+output_channel_count,input_height,width,kernel,stride,padding,+output_height,+output_width} = shape
  +count = (output_channel_count * output_height * output_width : U32)
  OutputTraversal.output(~&1,~Array<F32>,~Array<F32>,
    OutputTraversal.run(~&1,~Array<F32>,~Array<F32>,~F32,~F32,~ConvShape.Shape,~DirectContext,~F32,
      ~direct_job,~direct_input(~Array<F32>,~Storage.read_or(F32,0.0)),~direct_weight(~Array<F32>,~Storage.read_or(F32,0.0)),~direct_update,
      ~direct_write(~Array<F32>,~Storage.read_or(F32,0.0),~Array.set(F32)),
      U32.to_nat(count),0,shape,OutputTraversal.Buffers{input,weights,bias,Array.new(F32,Storage.allocation_depth(count),0.0)}))

# Conservative finite machine envelope, evaluated in Nat before any narrowing.
# It is deliberately stricter than the semantic SPEC; other valid problems
# retain the verified List implementation instead of being rejected.
def supported_with_limits(+limits: Machine.Limits,problem: ConvSpec.Problem) -> Bool:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  ConvShape.supports_direct(limits,shape,ConvSpec.length(input),ConvSpec.length(weights))

def supported(+problem: ConvSpec.Problem) -> Bool:
  supported_with_limits(Default.limits(),problem)

def native(+problem: ConvSpec.Problem) -> ConvSpec.ConvOutcome:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  +output_height = ConvSpec.out_height(problem)
  +output_width = ConvSpec.out_width(problem)
  ConvSpec.Accepted{output_height,output_width,Storage.read_list(F32,Nat.mul(ConvSpec.outputs(problem),Nat.mul(output_height,output_width)),0,
    run(ConvShape.from_spec(shape),pack(input),pack(weights),pack(bias)))}

def choose(ok: Bool, +problem: ConvSpec.Problem) -> ConvSpec.ConvOutcome:
  match ok:
    case False{}:
      Reference.infer(problem)
    case True{}:
      native(problem)

def validated(ok: Bool, +problem: ConvSpec.Problem) -> ConvSpec.ConvOutcome:
  match ok:
    case False{}:
      ConvSpec.Rejected{}
    case True{}:
      choose(supported(problem),problem)

def infer(+problem: ConvSpec.Problem) -> ConvSpec.ConvOutcome:
  validated(ConvSpec.valid(problem),problem)
