# Machine dimensions shared by all native convolution strategies. No storage.
import Base
import ./specification.bend as ConvSpec
import ../lib/machine_limits.bend as Machine
import ../lib/kernel_shape.bend as KernelShape

type Shape is Data:
  Shape{input_channels: U32,output_channels: U32,height: U32,width: U32,kernel: U32,stride: U32,padding: U32,output_height: U32,output_width: U32}

# Narrow only after the caller establishes the machine envelope below.
def from_spec(+shape: ConvSpec.Shape) -> Shape:
  ConvSpec.Shape{input_channels,output_channels,+height,+width,+kernel,+stride,+padding} = shape
  Shape{U32.from_nat(input_channels),U32.from_nat(output_channels),U32.from_nat(height),U32.from_nat(width),
    U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),
    U32.from_nat(ConvSpec.axis_size(height,kernel,stride,padding)),U32.from_nat(ConvSpec.axis_size(width,kernel,stride,padding))}

def from_problem(problem: ConvSpec.Problem) -> Shape:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  from_spec(shape)

# A single machine envelope for both public entry points. Semantic validation
# establishes the logical lengths; these checks precede all U32 narrowing.
def supports_direct(+limits: Machine.Limits,shape: ConvSpec.Shape,input_length: Nat,weight_length: Nat) -> Bool:
  ConvSpec.Shape{ci,co,height,width,kernel,stride,padding} = shape
  KernelShape.supports_direct(limits,KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding},input_length,weight_length)

def supports_packed(+limits: Machine.Limits,+shape: ConvSpec.Shape,input_length: Nat,weight_length: Nat) -> Bool:
  ConvSpec.Shape{ci,co,height,width,kernel,stride,padding} = shape
  KernelShape.supports_packed(limits,KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding},input_length,weight_length)
