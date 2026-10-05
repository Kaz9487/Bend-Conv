import Base
import ./traversal_array.bend as Traversal

# Clip once per leaf. A channel is one contiguous copy; padding never becomes
# a store into the next channel. Both owners stay affine across the channel loop.
def gather_width_when(inside: Bool,offset: U32,output_positions: U32,span: U32) -> U32:
  match inside:
    case False{}: 0
    case True{}: U32.min(span,(output_positions - offset : U32))

def gather_width(+offset: U32,+output_positions: U32,span: U32) -> U32:
  gather_width_when(U32.is_lt(offset,output_positions),offset,output_positions,span)

def gather_channel(+channel: U32,+width: U32,+offset: U32,+output_positions: U32,+span: U32,
  owners: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  (input_storage,output_storage) = owners
  Traversal.copy(~F32,U32.to_nat(width),(channel * span : U32),
    (channel * output_positions + offset : U32),input_storage,output_storage)

def gather_channels(remaining: Nat,+channel: U32,+width: U32,+offset: U32,+output_positions: U32,+span: U32,
  owners: Array<F32> & Array<F32>) -> Array<F32> & Array<F32>:
  match remaining:
    case 0n: owners
    case 1n+rest:
      gather_channels(rest,(channel + 1 : U32),width,offset,output_positions,span,
        gather_channel(channel,width,offset,output_positions,span,owners))

def gathered_output(owners: Array<F32> & Array<F32>) -> Array<F32>:
  (input_storage,output_storage) = owners
  output_storage

def gather_block(channels: U32,+offset: U32,+output_positions: U32,+span: U32,input_storage: Array<F32>,output_storage: Array<F32>) -> Array<F32>:
  gathered_output(gather_channels(U32.to_nat(channels),0,gather_width(offset,output_positions,span),offset,output_positions,span,
    (input_storage,output_storage)))
