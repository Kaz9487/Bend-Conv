import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/storage_shape.bend as Shape
# Scalar padding and guarded reads refine the same pixel abstraction as packing.
import Base
import ../specification.bend as ConvSpec
import ../scalar_kernel.bend as ScalarConv
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./encoded_array_read.bend as EncodedRead
import ./list_array_allocation.bend as ListArrayAllocation
import ./native_convolution_index.bend as NativeConvolutionIndex
import ./packed_pixel_address.bend as PackedPixelAddress
import ../../lib/proofs/padded_interval.bend as PaddedInterval
import ./native_sample_refinement.bend as NativeSampleRefinement
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/row_major_index.bend as RowMajor
import ./storage_capacity.bend as StorageCapacity

def pixel_matches_packed(+xs: List<&2,F32>,+channel: U32,+channels: U32,+height: U32,+width: U32,+row: U32,+column: U32,+count: U32,
  +ready: ListArrayAllocation.Ready(xs),+shape: {ConvSpec.length(xs) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))) : Nat},
  +channel_bound: {Nat.is_lt(U32.to_nat(channel),U32.to_nat(channels)) == True{} : Bool},
  positive_width: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},
  length_exact: {U32.to_nat(count) == ConvSpec.length(xs) : Nat}) ->
  {ScalarConv.sample(~Array<F32>,~Storage.read_or(F32,0.0),U32.is_lt(row,height) && U32.is_lt(column,width),ScalarConv.pack(xs),(channel * height * width + row * width + column : U32),count) ==
    Packing.pixel(ScalarConv.pack(xs),row,column,(channel * height * width : U32),height,width) : Array<F32> & F32}:
  Shape.Both{+depth_bound,+room} = ready
  EncodedRead.sample_matches_masked_get(U32.is_lt(row,height) && U32.is_lt(column,width),xs,
    (channel * height * width + row * width + column : U32),PackedPixelAddress.index(channel,height,width,row,column),count,ready,
    NativeConvolutionIndex.nchw(channel,height,width,row,column,Tree.capacity(F32,ListArrayAllocation.initial(xs)),
      StorageCapacity.small(ListArrayAllocation.depth(xs),ListArrayAllocation.initial(xs),Shape.fresh(F32,ListArrayAllocation.depth(xs),0.0),depth_bound),
      RowMajor.row_product(U32.to_nat(channel),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width),U32.to_nat(Tree.capacity(F32,ListArrayAllocation.initial(xs))),
        channel_bound,positive_width,
        %shape : {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,ListArrayAllocation.initial(xs)))) == True{} : Bool}
        room)),
    inside => PackedPixelAddress.bounds(xs,channel,channels,height,width,row,column,shape,channel_bound,
      PaddedInterval.parts(U32.is_lt(row,height),U32.is_lt(column,width),inside)),length_exact)

def padding_guard(+row: U32,+column: U32,+padding: U32,+height: U32,+width: U32,+capacity: U32,
  +capacity_bound: {U32.is_le(capacity,2147483648) == True{} : Bool},
  height_bound: {Nat.is_le(Nat.add(U32.to_nat(height),U32.to_nat(padding)),U32.to_nat(capacity)) == True{} : Bool},
  width_bound: {Nat.is_le(Nat.add(U32.to_nat(width),U32.to_nat(padding)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.is_le(padding,row) && U32.is_lt(row,(padding + height : U32)) && U32.is_le(padding,column) && U32.is_lt(column,(padding + width : U32)) ==
    U32.is_lt((row - padding : U32),height) && U32.is_lt((column - padding : U32),width) : Bool}:
  %PaddedInterval.native_interval(row,padding,height,capacity,capacity_bound,height_bound) :
    {U32.is_le(padding,row) && U32.is_lt(row,(padding + height : U32)) && U32.is_le(padding,column) && U32.is_lt(column,(padding + width : U32)) ==
      _ && U32.is_lt((column - padding : U32),width) : Bool}
  %PaddedInterval.native_interval(column,padding,width,capacity,capacity_bound,width_bound) :
    {U32.is_le(padding,row) && U32.is_lt(row,(padding + height : U32)) && U32.is_le(padding,column) && U32.is_lt(column,(padding + width : U32)) ==
      (U32.is_le(padding,row) && U32.is_lt(row,(padding + height : U32))) && _ : Bool}
  Equal.sym(Bool,(U32.is_le(padding,row) && U32.is_lt(row,(padding + height : U32))) && (U32.is_le(padding,column) && U32.is_lt(column,(padding + width : U32))),
    U32.is_le(padding,row) && U32.is_lt(row,(padding + height : U32)) && U32.is_le(padding,column) && U32.is_lt(column,(padding + width : U32)),
    NativeSampleRefinement.regroup(U32.is_le(padding,row),U32.is_lt(row,(padding + height : U32)),U32.is_le(padding,column),U32.is_lt(column,(padding + width : U32))))

def padded_pixel_matches_packed(+xs: List<&2,F32>,+channel: U32,+channels: U32,+height: U32,+width: U32,+row: U32,+column: U32,+padding: U32,+count: U32,
  ready: ListArrayAllocation.Ready(xs),shape: {ConvSpec.length(xs) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))) : Nat},
  channel_bound: {Nat.is_lt(U32.to_nat(channel),U32.to_nat(channels)) == True{} : Bool},
  positive_width: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},length_exact: {U32.to_nat(count) == ConvSpec.length(xs) : Nat},
  +capacity: U32,capacity_bound: {U32.is_le(capacity,2147483648) == True{} : Bool},
  height_bound: {Nat.is_le(Nat.add(U32.to_nat(height),U32.to_nat(padding)),U32.to_nat(capacity)) == True{} : Bool},
  width_bound: {Nat.is_le(Nat.add(U32.to_nat(width),U32.to_nat(padding)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {ScalarConv.sample(~Array<F32>,~Storage.read_or(F32,0.0),U32.is_le(padding,row) && U32.is_lt(row,(padding + height : U32)) && U32.is_le(padding,column) && U32.is_lt(column,(padding + width : U32)),
      ScalarConv.pack(xs),(channel * height * width + (row - padding) * width + column - padding : U32),count) ==
    Packing.pixel(ScalarConv.pack(xs),(row - padding : U32),(column - padding : U32),(channel * height * width : U32),height,width) : Array<F32> & F32}:
  %Equal.sym(Bool,
    U32.is_le(padding,row) && U32.is_lt(row,(padding + height : U32)) && U32.is_le(padding,column) && U32.is_lt(column,(padding + width : U32)),
    U32.is_lt((row - padding : U32),height) && U32.is_lt((column - padding : U32),width),
    padding_guard(row,column,padding,height,width,capacity,capacity_bound,height_bound,width_bound)) :
    {ScalarConv.sample(~Array<F32>,~Storage.read_or(F32,0.0),_,ScalarConv.pack(xs),(channel * height * width + (row - padding) * width + column - padding : U32),count) ==
      Packing.pixel(ScalarConv.pack(xs),(row - padding : U32),(column - padding : U32),(channel * height * width : U32),height,width) : Array<F32> & F32}
  %Equal.sym(U32,(channel * height * width + (row - padding) * width + column - padding : U32),
    (channel * height * width + (row - padding) * width + (column - padding) : U32),
    Arithmetic.subtract_from_sum((channel * height * width + (row - padding) * width : U32),column,padding)) :
    {ScalarConv.sample(~Array<F32>,~Storage.read_or(F32,0.0),U32.is_lt((row - padding : U32),height) && U32.is_lt((column - padding : U32),width),ScalarConv.pack(xs),_,count) ==
      Packing.pixel(ScalarConv.pack(xs),(row - padding : U32),(column - padding : U32),(channel * height * width : U32),height,width) : Array<F32> & F32}
  pixel_matches_packed(xs,channel,channels,height,width,(row - padding : U32),(column - padding : U32),count,ready,shape,channel_bound,positive_width,length_exact)
