# Natural dimensions at the validation boundary, before any machine narrowing.
# Convolution specifications and public tensor metadata use the same envelope.
import Base
import ./machine_limits.bend as Machine

type Dimensions is Data:
  Dimensions{input_channels: Nat,output_channels: Nat,height: Nat,width: Nat,kernel: Nat,stride: Nat,padding: Nat}

def axis_size(+size: Nat,+kernel: Nat,stride: Nat,+padding: Nat) -> Nat:
  1n+Nat.div(Nat.sub(Nat.add(size,Nat.mul(2n,padding)),kernel),stride)

def valid(shape: Dimensions,input_length: Nat,weight_length: Nat,bias_length: Nat) -> Bool:
  Dimensions{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Nat.is_lt(0n,ci) && Nat.is_lt(0n,co) && Nat.is_lt(0n,height) && Nat.is_lt(0n,width) && Nat.is_lt(0n,kernel) && Nat.is_lt(0n,stride) &&
  Nat.is_le(kernel,Nat.add(height,Nat.mul(2n,padding))) && Nat.is_le(kernel,Nat.add(width,Nat.mul(2n,padding))) &&
  Nat.is_eq(input_length,Nat.mul(ci,Nat.mul(height,width))) &&
  Nat.is_eq(weight_length,Nat.mul(co,Nat.mul(ci,Nat.mul(kernel,kernel)))) && Nat.is_eq(bias_length,co)

def direct_bounds(+dimension: U32,+capacity: U32,+coordinate: U32,shape: Dimensions,input_length: Nat,weight_length: Nat) -> Bool:
  Dimensions{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  +oh = axis_size(height,kernel,stride,padding)
  +ow = axis_size(width,kernel,stride,padding)
  Nat.is_le(ci,U32.to_nat(dimension)) && Nat.is_le(co,U32.to_nat(dimension)) &&
  Nat.is_le(height,U32.to_nat(dimension)) && Nat.is_le(width,U32.to_nat(dimension)) &&
  Nat.is_le(kernel,U32.to_nat(dimension)) && Nat.is_le(stride,U32.to_nat(dimension)) && Nat.is_le(padding,U32.to_nat(dimension)) &&
  Nat.is_le(input_length,U32.to_nat(capacity)) && Nat.is_le(weight_length,U32.to_nat(capacity)) &&
  Nat.is_le(Nat.mul(co,Nat.mul(oh,ow)),U32.to_nat(capacity)) &&
  Nat.is_le(Nat.add(Nat.mul(oh,stride),kernel),U32.to_nat(coordinate)) &&
  Nat.is_le(Nat.add(Nat.mul(ow,stride),kernel),U32.to_nat(coordinate))

def packed_bounds(dimension: U32,+capacity: U32,coordinate: U32,+shape: Dimensions,input_length: Nat,weight_length: Nat) -> Bool:
  Dimensions{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  +oh = axis_size(height,kernel,stride,padding)
  +ow = axis_size(width,kernel,stride,padding)
  +padded = Nat.mul(Nat.div(Nat.add(Nat.mul(oh,ow),7n),8n),8n)
  direct_bounds(dimension,capacity,coordinate,shape,input_length,weight_length) &&
  Nat.is_le(Nat.mul(padded,Nat.mul(ci,Nat.mul(kernel,kernel))),U32.to_nat(capacity)) &&
  Nat.is_le(Nat.mul(padded,co),U32.to_nat(capacity))

# Never duplicate the Limits owner: its Facts field depends on its values.
def supports_direct(limits: Machine.Limits,shape: Dimensions,input_length: Nat,weight_length: Nat) -> Bool:
  Machine.Limits{dimension,capacity,coordinate,half_word,facts} = limits
  direct_bounds(dimension,capacity,coordinate,shape,input_length,weight_length)

def supports_packed(limits: Machine.Limits,shape: Dimensions,input_length: Nat,weight_length: Nat) -> Bool:
  Machine.Limits{dimension,capacity,coordinate,half_word,facts} = limits
  packed_bounds(dimension,capacity,coordinate,shape,input_length,weight_length)
