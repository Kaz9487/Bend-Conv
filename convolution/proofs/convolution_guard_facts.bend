# Original validation extraction and parameterized support facts.
# Machine support facts are in concrete_support_guard.bend.
import Base
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/nat_order.bend as Order
import ../specification.bend as ConvSpec

def input_channel_count(problem: ConvSpec.Problem) -> Nat:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  ConvSpec.Shape{input_channels,output_channels,input_height,input_width,kernel_size,stride,padding} = shape
  input_channels

def output_channel_count(problem: ConvSpec.Problem) -> Nat:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  ConvSpec.Shape{input_channels,output_channels,input_height,input_width,kernel_size,stride,padding} = shape
  output_channels

def height(problem: ConvSpec.Problem) -> Nat:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  ConvSpec.Shape{input_channels,output_channels,input_height,input_width,kernel_size,stride,padding} = shape
  input_height

def width(problem: ConvSpec.Problem) -> Nat:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  ConvSpec.Shape{input_channels,output_channels,input_height,input_width,kernel_size,stride,padding} = shape
  input_width

def kernel(problem: ConvSpec.Problem) -> Nat:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  ConvSpec.Shape{input_channels,output_channels,input_height,input_width,kernel_size,stride,padding} = shape
  kernel_size

def stride(problem: ConvSpec.Problem) -> Nat:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  ConvSpec.Shape{input_channels,output_channels,input_height,input_width,kernel_size,stride,padding} = shape
  stride

def pad(problem: ConvSpec.Problem) -> Nat:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  ConvSpec.Shape{input_channels,output_channels,input_height,input_width,kernel_size,stride,padding} = shape
  padding

def input(problem: ConvSpec.Problem) -> List<&2,F32>:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  input

def weights(problem: ConvSpec.Problem) -> List<&2,F32>:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  weights

def bias(problem: ConvSpec.Problem) -> List<&2,F32>:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  bias

def reflect(+a: Nat,+b: Nat,h: {Nat.is_eq(a,b) == True{} : Bool}) -> {a == b : Nat}:
  Order.equal_of_is_eq(a,b,h)

def input_channels_positive(+problem: ConvSpec.Problem) -> Bool:
  Nat.is_lt(0n,input_channel_count(problem))

def output_channels_positive(+problem: ConvSpec.Problem) -> Bool:
  Nat.is_lt(0n,output_channel_count(problem))

def height_positive(+problem: ConvSpec.Problem) -> Bool:
  Nat.is_lt(0n,height(problem))

def width_positive(+problem: ConvSpec.Problem) -> Bool:
  Nat.is_lt(0n,width(problem))

def kernel_positive(+problem: ConvSpec.Problem) -> Bool:
  Nat.is_lt(0n,kernel(problem))

def stride_positive(+problem: ConvSpec.Problem) -> Bool:
  Nat.is_lt(0n,stride(problem))

def kernel_fits_height(+problem: ConvSpec.Problem) -> Bool:
  Nat.is_le(kernel(problem),Nat.add(height(problem),Nat.mul(2n,pad(problem))))

def kernel_fits_width(+problem: ConvSpec.Problem) -> Bool:
  Nat.is_le(kernel(problem),Nat.add(width(problem),Nat.mul(2n,pad(problem))))

def input_length_matches_shape(+problem: ConvSpec.Problem) -> Bool:
  Nat.is_eq(ConvSpec.length(input(problem)),Nat.mul(input_channel_count(problem),Nat.mul(height(problem),width(problem))))

def weight_length_matches_shape(+problem: ConvSpec.Problem) -> Bool:
  Nat.is_eq(ConvSpec.length(weights(problem)),Nat.mul(output_channel_count(problem),Nat.mul(input_channel_count(problem),Nat.mul(kernel(problem),kernel(problem)))))

def bias_length_matches_channels(+problem: ConvSpec.Problem) -> Bool:
  Nat.is_eq(ConvSpec.length(bias(problem)),output_channel_count(problem))

type Valid<-problem: ConvSpec.Problem> is Data:
  Valid{input_channels_positive: {input_channels_positive(problem) == True{} : Bool},
    output_channels_positive: {output_channels_positive(problem) == True{} : Bool},height_positive: {height_positive(problem) == True{} : Bool},
    width_positive: {width_positive(problem) == True{} : Bool},kernel_positive: {kernel_positive(problem) == True{} : Bool},
    stride_positive: {stride_positive(problem) == True{} : Bool},kernel_fits_height: {kernel_fits_height(problem) == True{} : Bool},
    kernel_fits_width: {kernel_fits_width(problem) == True{} : Bool},input_length_matches_shape: {input_length_matches_shape(problem) == True{} : Bool},
    weight_length_matches_shape: {weight_length_matches_shape(problem) == True{} : Bool},
    bias_length_matches_channels: {bias_length_matches_channels(problem) == True{} : Bool}}

def valid_conditions(+problem: ConvSpec.Problem) -> List<&2,Bool>:
  [bias_length_matches_channels(problem),weight_length_matches_shape(problem),input_length_matches_shape(problem),kernel_fits_width(problem),kernel_fits_height(problem),
    stride_positive(problem),kernel_positive(problem),width_positive(problem),height_positive(problem),output_channels_positive(problem),input_channels_positive(problem)]

def valid(+problem: ConvSpec.Problem,+h: {ConvSpec.valid(problem) == True{} : Bool}) -> Valid<problem>:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+input_channels,+output_channels,+input_height,+input_width,+kernel_size,+stride,+padding} = shape
  Valid{
    Boolean.every_condition_holds(valid_conditions(problem),10n,h),
    Boolean.every_condition_holds(valid_conditions(problem),9n,h),
    Boolean.every_condition_holds(valid_conditions(problem),8n,h),
    Boolean.every_condition_holds(valid_conditions(problem),7n,h),
    Boolean.every_condition_holds(valid_conditions(problem),6n,h),
    Boolean.every_condition_holds(valid_conditions(problem),5n,h),
    Boolean.every_condition_holds(valid_conditions(problem),4n,h),
    Boolean.every_condition_holds(valid_conditions(problem),3n,h),
    Boolean.every_condition_holds(valid_conditions(problem),2n,h),
    Boolean.every_condition_holds(valid_conditions(problem),1n,h),
    Boolean.every_condition_holds(valid_conditions(problem),0n,h)}

def input_channels_supported(+problem: ConvSpec.Problem,+d: U32,+m: U32,+c: U32) -> Bool:
  Nat.is_le(input_channel_count(problem),U32.to_nat(d))

def output_channels_supported(+problem: ConvSpec.Problem,+d: U32,+m: U32,+c: U32) -> Bool:
  Nat.is_le(output_channel_count(problem),U32.to_nat(d))

def height_supported(+problem: ConvSpec.Problem,+d: U32,+m: U32,+c: U32) -> Bool:
  Nat.is_le(height(problem),U32.to_nat(d))

def width_supported(+problem: ConvSpec.Problem,+d: U32,+m: U32,+c: U32) -> Bool:
  Nat.is_le(width(problem),U32.to_nat(d))

def kernel_supported(+problem: ConvSpec.Problem,+d: U32,+m: U32,+c: U32) -> Bool:
  Nat.is_le(kernel(problem),U32.to_nat(d))

def stride_supported(+problem: ConvSpec.Problem,+d: U32,+m: U32,+c: U32) -> Bool:
  Nat.is_le(stride(problem),U32.to_nat(d))

def padding_supported(+problem: ConvSpec.Problem,+d: U32,+m: U32,+c: U32) -> Bool:
  Nat.is_le(pad(problem),U32.to_nat(d))

def input_buffer_supported(+problem: ConvSpec.Problem,+d: U32,+m: U32,+c: U32) -> Bool:
  Nat.is_le(ConvSpec.length(input(problem)),U32.to_nat(m))

def weight_buffer_supported(+problem: ConvSpec.Problem,+d: U32,+m: U32,+c: U32) -> Bool:
  Nat.is_le(ConvSpec.length(weights(problem)),U32.to_nat(m))

def output_buffer_supported(+problem: ConvSpec.Problem,+d: U32,+m: U32,+c: U32) -> Bool:
  Nat.is_le(Nat.mul(output_channel_count(problem),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))),U32.to_nat(m))

def vertical_coordinate_supported(+problem: ConvSpec.Problem,+d: U32,+m: U32,+c: U32) -> Bool:
  Nat.is_le(Nat.add(Nat.mul(ConvSpec.out_height(problem),stride(problem)),kernel(problem)),U32.to_nat(c))

def horizontal_coordinate_supported(+problem: ConvSpec.Problem,+d: U32,+m: U32,+c: U32) -> Bool:
  Nat.is_le(Nat.add(Nat.mul(ConvSpec.out_width(problem),stride(problem)),kernel(problem)),U32.to_nat(c))

def supported_formula(+problem: ConvSpec.Problem,+d: U32,+m: U32,+c: U32) -> Bool:
  input_channels_supported(problem,d,m,c) && output_channels_supported(problem,d,m,c) && height_supported(problem,d,m,c) && width_supported(problem,d,m,
    c) && kernel_supported(problem,d,m,c) && stride_supported(problem,d,m,c) && padding_supported(problem,d,m,c) && input_buffer_supported(problem,d,m,
    c) && weight_buffer_supported(problem,d,m,c) && output_buffer_supported(problem,d,m,c) && vertical_coordinate_supported(problem,d,m,
    c) && horizontal_coordinate_supported(problem,d,m,c)

type Supported<-problem: ConvSpec.Problem,-d: U32,-m: U32,-c: U32> is Data:
  Supported{input_channels_supported: {input_channels_supported(problem,d,m,c) == True{} : Bool},
    output_channels_supported: {output_channels_supported(problem,d,m,c) == True{} : Bool},height_supported: {height_supported(problem,d,m,
        c) == True{} : Bool},width_supported: {width_supported(problem,d,m,c) == True{} : Bool},kernel_supported: {kernel_supported(problem,d,m,
        c) == True{} : Bool},stride_supported: {stride_supported(problem,d,m,c) == True{} : Bool},padding_supported: {padding_supported(problem,d,m,
        c) == True{} : Bool},input_buffer_supported: {input_buffer_supported(problem,d,m,c) == True{} : Bool},
    weight_buffer_supported: {weight_buffer_supported(problem,d,m,c) == True{} : Bool},output_buffer_supported: {output_buffer_supported(problem,d,m,
        c) == True{} : Bool},vertical_coordinate_supported: {vertical_coordinate_supported(problem,d,m,c) == True{} : Bool},
    horizontal_coordinate_supported: {horizontal_coordinate_supported(problem,d,m,c) == True{} : Bool}}

def supported_conditions(+problem: ConvSpec.Problem,+d: U32,+m: U32,+c: U32) -> List<&2,Bool>:
  [horizontal_coordinate_supported(problem,d,m,c),vertical_coordinate_supported(problem,d,m,c),output_buffer_supported(problem,d,m,c),weight_buffer_supported(problem,d,
      m,c),input_buffer_supported(problem,d,m,c),padding_supported(problem,d,m,c),stride_supported(problem,d,m,c),kernel_supported(problem,d,m,c),width_supported(problem,d,m,
      c),height_supported(problem,d,m,c),output_channels_supported(problem,d,m,c),input_channels_supported(problem,d,m,c)]

def supported_generic(+problem: ConvSpec.Problem,+d: U32,+m: U32,+c: U32,+h: {supported_formula(problem,d,m,c) == True{} : Bool}) -> Supported<problem,d,m,c>:
  Supported{
    Boolean.every_condition_holds(supported_conditions(problem,d,m,c),11n,h),
    Boolean.every_condition_holds(supported_conditions(problem,d,m,c),10n,h),
    Boolean.every_condition_holds(supported_conditions(problem,d,m,c),9n,h),
    Boolean.every_condition_holds(supported_conditions(problem,d,m,c),8n,h),
    Boolean.every_condition_holds(supported_conditions(problem,d,m,c),7n,h),
    Boolean.every_condition_holds(supported_conditions(problem,d,m,c),6n,h),
    Boolean.every_condition_holds(supported_conditions(problem,d,m,c),5n,h),
    Boolean.every_condition_holds(supported_conditions(problem,d,m,c),4n,h),
    Boolean.every_condition_holds(supported_conditions(problem,d,m,c),3n,h),
    Boolean.every_condition_holds(supported_conditions(problem,d,m,c),2n,h),
    Boolean.every_condition_holds(supported_conditions(problem,d,m,c),1n,h),
    Boolean.every_condition_holds(supported_conditions(problem,d,m,c),0n,h)}
