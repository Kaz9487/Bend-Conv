import ../../lib/proofs/storage_shape.bend as Shape
# Native aligned-packer pixel lookup -> logical dense NCHW lookup.
# Loop-counter refinement to convolution (oy,ox,ky,kx) is a separate obligation.
import Base
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./input_storage.bend as Inputs
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ./list_array_allocation.bend as ListArrayAllocation
import ./packing_block_reads.bend as Reads
import ../../lib/proofs/row_major_index.bend as RowMajor
import ./native_convolution_index.bend as NativeConvolutionIndex
import ../../lib/proofs/padded_interval.bend as PaddedInterval
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ./storage_capacity.bend as StorageCapacity
import ../specification.bend as ConvSpec
import ../scalar_kernel.bend as ScalarConv
import ../../lib/kernel_packing_samples.bend as PackingSamples
import ../../lib/kernel_spatial_packing.bend as Packing

def index(+input_channel_index: U32,+height: U32,+width: U32,+iy: U32,+ix: U32) -> Nat:
  RowMajor.index(U32.to_nat(input_channel_index),U32.to_nat(height),U32.to_nat(width),U32.to_nat(iy),U32.to_nat(ix))

def guard(+iy: U32,+ix: U32,+height: U32,+width: U32) -> Bool:
  Nat.is_lt(U32.to_nat(iy),U32.to_nat(height)) && Nat.is_lt(U32.to_nat(ix),U32.to_nat(width))

def value(xs: List<&2,F32>,+input_channel_index: U32,+height: U32,+width: U32,+iy: U32,+ix: U32) -> F32:
  ConvSpec.sample_if(guard(iy,ix,height,width),index(input_channel_index,height,width,iy,ix),xs)

def guard_equal(+iy: U32,+ix: U32,+height: U32,+width: U32) -> {U32.is_lt(iy,height) && U32.is_lt(ix,width) == guard(iy,ix,height,width) : Bool}:
  %Comparison.lt(iy,height) : {U32.is_lt(iy,height) && U32.is_lt(ix,width) == _ && Nat.is_lt(U32.to_nat(ix),U32.to_nat(width)) : Bool}
  %Comparison.lt(ix,width) : {U32.is_lt(iy,height) && U32.is_lt(ix,width) == U32.is_lt(iy,height) && _ : Bool}
  {==}

def bounds(+xs: List<&2,F32>,+input_channel_index: U32,+input_channel_count: U32,+height: U32,+width: U32,+iy: U32,+ix: U32,shape: {ConvSpec.length(xs) == Nat.mul(U32.to_nat(input_channel_count),
      Nat.mul(U32.to_nat(height),U32.to_nat(width))) : Nat},channel: {Nat.is_lt(U32.to_nat(input_channel_index),U32.to_nat(input_channel_count)) == True{} : Bool},
  pair: Shape.Both<{U32.is_lt(iy,height) == True{} : Bool},{U32.is_lt(ix,width) == True{} : Bool}>) -> {Nat.is_lt(index(input_channel_index,height,width,
      iy,ix),ConvSpec.length(xs)) == True{} : Bool}:
  Shape.Both{row,col} = pair
  %Equal.sym(Nat,ConvSpec.length(xs),Nat.mul(U32.to_nat(input_channel_count),Nat.mul(U32.to_nat(height),U32.to_nat(width))),shape) : {Nat.is_lt(index(input_channel_index,height,
        width,iy,ix),_) == True{} : Bool}
  RowMajor.in_bounds(U32.to_nat(input_channel_index),U32.to_nat(input_channel_count),U32.to_nat(height),U32.to_nat(width),U32.to_nat(iy),U32.to_nat(ix),channel,
    Bounds.as_nat_lt(iy,height,row),Bounds.as_nat_lt(ix,width,col))

def when(+inside: Bool,+xs: List<&2,F32>,+evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(xs),xs>,+address: U32,+at: Nat,proof: ListArrayAllocation.Ready(xs),encoded: {address == U32.from_nat(at) : U32},
  valid: {inside == True{} : Bool} -> {Nat.is_lt(at,ConvSpec.length(xs)) == True{} : Bool}) -> {PackingSamples.masked(Tree.read(F32,
      StorageLaws.sequence_storage(F32,ConvSpec.length(xs),xs,evidence),address),inside) == ConvSpec.sample_if(inside,at,xs) : F32}:
  match inside:
    case False{}:
      {==}
    case True{}:
      %Equal.sym(U32,address,U32.from_nat(at),encoded) :
        {PackingSamples.masked(Tree.read(F32,StorageLaws.sequence_storage(F32,ConvSpec.length(xs),xs,evidence),_),True{}) == ConvSpec.sample_if(True{},at,xs) : F32}
      Inputs.stored_value(xs,evidence,at,valid({==}))

law scalar_value:
  for +xs: List<&2,F32>
  for +evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(xs),xs>
  for +input_channel_index: U32
  for +input_channel_count: U32
  for +height: U32
  for +width: U32
  for +iy: U32
  for +ix: U32
  for +proof: ListArrayAllocation.Ready(xs)
  for +shape: {ConvSpec.length(xs) == Nat.mul(U32.to_nat(input_channel_count),Nat.mul(U32.to_nat(height),U32.to_nat(width))) : Nat}
  for +channel: {Nat.is_lt(U32.to_nat(input_channel_index),U32.to_nat(input_channel_count)) == True{} : Bool}
  for +positive_width: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool}
  {Reads.value(StorageLaws.sequence_storage(F32,ConvSpec.length(xs),xs,evidence),iy,ix,(input_channel_index * height * width : U32),height,width) == value(xs,input_channel_index,
      height,width,iy,ix) : F32}
def scalar_value(xs,evidence,input_channel_index,input_channel_count,height,width,iy,ix,proof,shape,channel,positive_width):
  Shape.Both{+limit,+room} = proof
  %guard_equal(iy,ix,height,width) : {Reads.value(StorageLaws.sequence_storage(F32,ConvSpec.length(xs),xs,evidence),iy,ix,(input_channel_index * height * width : U32),
      height,width) == ConvSpec.sample_if(_,index(input_channel_index,height,width,iy,ix),xs) : F32}
  when(U32.is_lt(iy,height) && U32.is_lt(ix,width),xs,evidence,(input_channel_index * height * width + iy * width + ix : U32),index(input_channel_index,height,width,iy,ix),
    Shape.Both{limit,room},
    NativeConvolutionIndex.nchw(input_channel_index,height,width,iy,ix,Tree.capacity(F32,ListArrayAllocation.initial(xs)),
      StorageCapacity.small(ListArrayAllocation.depth(xs),ListArrayAllocation.initial(xs),Shape.fresh(F32,ListArrayAllocation.depth(xs),0.0),limit),
      RowMajor.row_product(U32.to_nat(input_channel_index),U32.to_nat(input_channel_count),U32.to_nat(height),U32.to_nat(width),U32.to_nat(Tree.capacity(F32,
            ListArrayAllocation.initial(xs))),channel,positive_width,
        %shape : {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,ListArrayAllocation.initial(xs)))) == True{} : Bool}
        room)),
    h => bounds(xs,input_channel_index,input_channel_count,height,width,iy,ix,shape,channel,PaddedInterval.parts(U32.is_lt(iy,height),U32.is_lt(ix,width),h)))

law native_pixel:
  for +xs: List<&2,F32>
  for +evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(xs),xs>
  for +input_channel_index: U32
  for +input_channel_count: U32
  for +height: U32
  for +width: U32
  for +iy: U32
  for +ix: U32
  for +proof: ListArrayAllocation.Ready(xs)
  for +shape: {ConvSpec.length(xs) == Nat.mul(U32.to_nat(input_channel_count),Nat.mul(U32.to_nat(height),U32.to_nat(width))) : Nat}
  for +channel: {Nat.is_lt(U32.to_nat(input_channel_index),U32.to_nat(input_channel_count)) == True{} : Bool}
  for +positive_width: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool}
  {Packing.pixel(StorageLaws.sequence_owner(F32,ConvSpec.length(xs),xs,evidence),iy,ix,(input_channel_index * height * width : U32),height,width) == (StorageLaws.sequence_owner(F32,ConvSpec.length(xs),xs,evidence),value(xs,input_channel_index,height,width,iy,
        ix)) : Array<F32> & F32}
def native_pixel(xs,evidence,input_channel_index,input_channel_count,height,width,iy,ix,proof,shape,channel,positive_width):
  %scalar_value(xs,evidence,input_channel_index,input_channel_count,height,width,iy,ix,proof,shape,channel,positive_width) : {Packing.pixel(Tree.pack(F32,
        StorageLaws.sequence_storage(F32,ConvSpec.length(xs),xs,evidence)),iy,ix,(input_channel_index * height * width : U32),height,width) == (Tree.pack(F32,
        StorageLaws.sequence_storage(F32,ConvSpec.length(xs),xs,evidence)),_) : Array<F32> & F32}
  Reads.pixel(StorageLaws.sequence_storage(F32,ConvSpec.length(xs),xs,evidence),iy,ix,(input_channel_index * height * width : U32),height,width)
