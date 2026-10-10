# The logical value of every output element of a banded convolution: decode
# the NCHW index, build that image's problem from the input's logical values
# and apply the activation to its matrix value. Band Values of the output use
# this function, so "computed on bands" means "the dense result, split".
import Base
import ../../lib/proofs/kernel_pipeline_activation.bend as ActivationProofs
import ../specification.bend as ConvSpec
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/tensor_band_coordinates.bend as BandCoordinates
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/nat_order.bend as Order
import ./output_values.bend as Values
import ./band_window_contents.bend as WindowContents

type Context<-Logical: Data> is Data:
  Context{logical: Logical,activation: Activations.Activation,channels: Nat,outputs: Nat,height: Nat,width: Nat,kernel: Nat,stride: Nat,padding: Nat,
    weights: List<&2,F32>,bias: List<&2,F32>}

# Image `image` of the input as a convolution problem.
def problem(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat) -> ConvSpec.Problem:
  +area = Nat.mul(channels,Nat.mul(height,width))
  ConvSpec.Problem{ConvSpec.Shape{channels,outputs,height,width,kernel,stride,padding},
    WindowContents.image_values(~Logical,~original,area,Nat.mul(image,area),logical),weights,bias}

def at_position(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,
  +channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +image: Nat,+channel: Nat,+position: Nat) -> F32:
  +output_width = ConvSpec.axis_size(width,kernel,stride,padding)
  ActivationProofs.activation_value(~custom,activation,
    Values.matrix_value(problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),channel,
      Nat.div(position,output_width),Nat.mod(position,output_width)))

def plane(+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat) -> Nat:
  Nat.mul(ConvSpec.axis_size(height,kernel,stride,padding),ConvSpec.axis_size(width,kernel,stride,padding))

def at_rest(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,
  +channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +image: Nat,+rest: Nat) -> F32:
  +area = plane(height,width,kernel,stride,padding)
  at_position(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,Nat.div(rest,area),Nat.mod(rest,area))

def value_at(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,
  +channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,+flat: Nat) -> F32:
  +volume = Nat.mul(outputs,plane(height,width,kernel,stride,padding))
  at_rest(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,Nat.div(flat,volume),Nat.mod(flat,volume))

def value(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,context: Context<Logical>,index: U32) -> F32:
  Context{+logical,+activation,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias} = context
  value_at(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,U32.to_nat(index))

# An output index built from (image, channel, position) decodes to them.
def decoded(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,
  +channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +image: Nat,+channel: Nat,+position: Nat,
  +channel_inside: {Nat.is_lt(channel,outputs) == True{} : Bool},
  +position_inside: {Nat.is_lt(position,plane(height,width,kernel,stride,padding)) == True{} : Bool}) ->
  {value_at(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,
      Nat.add(Nat.mul(image,Nat.mul(outputs,plane(height,width,kernel,stride,padding))),Nat.add(Nat.mul(channel,plane(height,width,kernel,stride,padding)),position))) ==
    at_position(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,channel,position) : F32}:
  +area = plane(height,width,kernel,stride,padding)
  +volume = Nat.mul(outputs,area)
  +inner = Nat.add(Nat.mul(channel,area),position)
  +flat = Nat.add(Nat.mul(image,volume),inner)
  +area_positive = Order.le_lt_trans(0n,position,area,Order.zero_le(position),position_inside)
  +inner_inside = RowMajor.radix(channel,outputs,position,area,channel_inside,position_inside)
  +volume_positive = Order.le_lt_trans(0n,inner,volume,Order.zero_le(inner),inner_inside)
  Equal.trans(F32,at_rest(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,Nat.div(flat,volume),Nat.mod(flat,volume)),
    at_rest(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,inner),
    at_position(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,channel,position),
    Equal.trans(F32,at_rest(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,Nat.div(flat,volume),Nat.mod(flat,volume)),
      at_rest(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,Nat.mod(flat,volume)),
      at_rest(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,inner),
      Equal.cong(Nat,F32,value => at_rest(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,value,Nat.mod(flat,volume)),
        Nat.div(flat,volume),image,BandCoordinates.quotient(image,inner,volume,volume_positive,inner_inside)),
      Equal.cong(Nat,F32,value => at_rest(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,value),
        Nat.mod(flat,volume),inner,BandCoordinates.remainder(image,inner,volume,volume_positive,inner_inside))),
    Equal.trans(F32,at_rest(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,inner),
      at_position(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,channel,Nat.mod(inner,area)),
      at_position(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,channel,position),
      Equal.cong(Nat,F32,value => at_position(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,value,Nat.mod(inner,area)),
        Nat.div(inner,area),channel,BandCoordinates.quotient(channel,position,area,area_positive,position_inside)),
      Equal.cong(Nat,F32,value => at_position(~custom,~Logical,~original,logical,activation,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,channel,value),
        Nat.mod(inner,area),position,BandCoordinates.remainder(channel,position,area,area_positive,position_inside))))
