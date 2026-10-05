import ../../lib/proofs/storage_shape.bend as Shape
import Base
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ./packed_pixel_address.bend as PackedPixelAddress
import ./list_array_allocation.bend as ListArrayAllocation
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/padded_interval.bend as PaddedInterval
import ../specification.bend as ConvSpec
import ../scalar_kernel.bend as ScalarConv
import ../../lib/kernel_spatial_packing.bend as Packing

def axis(+p: U32,+pad: U32,+extent: U32) -> Bool:
  Nat.is_le(U32.to_nat(pad),U32.to_nat(p)) && Nat.is_lt(U32.to_nat(p),Nat.add(U32.to_nat(pad),U32.to_nat(extent)))

def native_guard(+padded_row: U32,+padded_column: U32,+pad: U32,+h: U32,+w: U32) -> Bool:
  U32.is_lt((padded_row - pad : U32),h) && U32.is_lt((padded_column - pad : U32),w)

def index(+input_channel_index: U32,+h: U32,+w: U32,+padded_row: U32,+padded_column: U32,+pad: U32) -> Nat:
  RowMajor.index(U32.to_nat(input_channel_index),U32.to_nat(h),U32.to_nat(w),Nat.sub(U32.to_nat(padded_row),U32.to_nat(pad)),Nat.sub(U32.to_nat(padded_column),U32.to_nat(pad)))

def value(+xs: List<&2,F32>,+input_channel_index: U32,+h: U32,+w: U32,+padded_row: U32,+padded_column: U32,+pad: U32) -> F32:
  ConvSpec.sample_if(axis(padded_row,pad,h) && axis(padded_column,pad,w),index(input_channel_index,h,w,padded_row,padded_column,pad),xs)

def index_equal(+input_channel_index: U32,+h: U32,+w: U32,+padded_row: U32,+padded_column: U32,+pad: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  row_bound: {Nat.is_le(Nat.add(U32.to_nat(h),U32.to_nat(pad)),U32.to_nat(cap)) == True{} : Bool},col_bound: {Nat.is_le(Nat.add(U32.to_nat(w),
        U32.to_nat(pad)),U32.to_nat(cap)) == True{} : Bool},pair: Shape.Both<{U32.is_lt((padded_row - pad : U32),h) == True{} : Bool},
  {U32.is_lt((padded_column - pad : U32),w) == True{} : Bool}>) -> {PackedPixelAddress.index(input_channel_index,h,w,(padded_row - pad : U32),(padded_column - pad : U32)) == index(input_channel_index,h,w,padded_row,
    padded_column,pad) : Nat}:
  Shape.Both{row,col} = pair
  %PaddedInterval.offset(padded_row,pad,h,cap,cap_ok,row_bound,row) : {PackedPixelAddress.index(input_channel_index,h,w,(padded_row - pad : U32),
      (padded_column - pad : U32)) == RowMajor.index(U32.to_nat(input_channel_index),U32.to_nat(h),U32.to_nat(w),_,Nat.sub(U32.to_nat(padded_column),U32.to_nat(pad))) : Nat}
  %PaddedInterval.offset(padded_column,pad,w,cap,cap_ok,col_bound,col) : {PackedPixelAddress.index(input_channel_index,h,w,(padded_row - pad : U32),
      (padded_column - pad : U32)) == RowMajor.index(U32.to_nat(input_channel_index),U32.to_nat(h),U32.to_nat(w),U32.to_nat((padded_row - pad : U32)),_) : Nat}
  {==}

def when(+inside: Bool,+xs: List<&2,F32>,+a: Nat,+b: Nat,eq: {inside == True{} : Bool} -> {a == b : Nat}) -> {ConvSpec.sample_if(inside,a,
    xs) == ConvSpec.sample_if(inside,b,xs) : F32}:
  match inside:
    case False{}:
      {==}
    case True{}:
      %eq({==}) : {ConvSpec.at(a,xs) == ConvSpec.at(_,xs) : F32}
      {==}

law translated:
  for +xs: List<&2,F32>
  for +input_channel_index: U32
  for +h: U32
  for +w: U32
  for +padded_row: U32
  for +padded_column: U32
  for +pad: U32
  for +cap: U32
  for +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool}
  for +row_bound: {Nat.is_le(Nat.add(U32.to_nat(h),U32.to_nat(pad)),U32.to_nat(cap)) == True{} : Bool}
  for +col_bound: {Nat.is_le(Nat.add(U32.to_nat(w),U32.to_nat(pad)),U32.to_nat(cap)) == True{} : Bool}
  {PackedPixelAddress.value(xs,input_channel_index,h,w,(padded_row - pad : U32),(padded_column - pad : U32)) == value(xs,input_channel_index,h,w,padded_row,padded_column,pad) : F32}
def translated(xs,input_channel_index,h,w,padded_row,padded_column,pad,cap,cap_ok,row_bound,col_bound):
  %PackedPixelAddress.guard_equal((padded_row - pad : U32),(padded_column - pad : U32),h,w) : {ConvSpec.sample_if(_,PackedPixelAddress.index(input_channel_index,h,w,
        (padded_row - pad : U32),(padded_column - pad : U32)),xs) == value(xs,input_channel_index,h,w,padded_row,padded_column,pad) : F32}
  %PaddedInterval.range(padded_row,pad,h,cap,cap_ok,row_bound) : {ConvSpec.sample_if(native_guard(padded_row,padded_column,pad,h,w),PackedPixelAddress.index(input_channel_index,h,
        w,(padded_row - pad : U32),(padded_column - pad : U32)),xs) == ConvSpec.sample_if(_ && axis(padded_column,pad,w),index(input_channel_index,h,w,padded_row,padded_column,pad),xs) : F32}
  %PaddedInterval.range(padded_column,pad,w,cap,cap_ok,col_bound) : {ConvSpec.sample_if(native_guard(padded_row,padded_column,pad,h,w),PackedPixelAddress.index(input_channel_index,h,
        w,(padded_row - pad : U32),(padded_column - pad : U32)),xs) == ConvSpec.sample_if(U32.is_lt((padded_row - pad : U32),h) && _,index(input_channel_index,h,w,padded_row,padded_column,pad),xs) : F32}
  when(native_guard(padded_row,padded_column,pad,h,w),xs,PackedPixelAddress.index(input_channel_index,h,w,(padded_row - pad : U32),(padded_column - pad : U32)),index(input_channel_index,h,w,padded_row,padded_column,pad),
    yes => index_equal(input_channel_index,h,w,padded_row,padded_column,pad,cap,cap_ok,row_bound,col_bound,PaddedInterval.parts(U32.is_lt((padded_row - pad : U32),h),
        U32.is_lt((padded_column - pad : U32),w),yes)))

law native_pixel:
  for +xs: List<&2,F32>
  for +evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(xs),xs>
  for +input_channel_index: U32
  for +input_channel_count: U32
  for +h: U32
  for +w: U32
  for +padded_row: U32
  for +padded_column: U32
  for +pad: U32
  for +proof: ListArrayAllocation.Ready(xs)
  for +shape: {ConvSpec.length(xs) == Nat.mul(U32.to_nat(input_channel_count),Nat.mul(U32.to_nat(h),U32.to_nat(w))) : Nat}
  for +channel: {Nat.is_lt(U32.to_nat(input_channel_index),U32.to_nat(input_channel_count)) == True{} : Bool}
  for +positive_width: {Nat.is_lt(0n,U32.to_nat(w)) == True{} : Bool}
  for +cap: U32
  for +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool}
  for +row_bound: {Nat.is_le(Nat.add(U32.to_nat(h),U32.to_nat(pad)),U32.to_nat(cap)) == True{} : Bool}
  for +col_bound: {Nat.is_le(Nat.add(U32.to_nat(w),U32.to_nat(pad)),U32.to_nat(cap)) == True{} : Bool}
  {Packing.pixel(StorageLaws.sequence_owner(F32,ConvSpec.length(xs),xs,evidence),(padded_row - pad : U32),(padded_column - pad : U32),(input_channel_index * h * w : U32),h,w) == (StorageLaws.sequence_owner(F32,ConvSpec.length(xs),xs,evidence),value(xs,input_channel_index,h,w,padded_row,padded_column,
        pad)) : Array<F32> & F32}
def native_pixel(xs,evidence,input_channel_index,input_channel_count,h,w,padded_row,padded_column,pad,proof,shape,channel,positive_width,cap,cap_ok,row_bound,col_bound):
  %translated(xs,input_channel_index,h,w,padded_row,padded_column,pad,cap,cap_ok,row_bound,col_bound) : {Packing.pixel(StorageLaws.sequence_owner(F32,ConvSpec.length(xs),xs,evidence),(padded_row - pad : U32),(padded_column - pad : U32),
      (input_channel_index * h * w : U32),h,w) == (StorageLaws.sequence_owner(F32,ConvSpec.length(xs),xs,evidence),_) : Array<F32> & F32}
  PackedPixelAddress.native_pixel(xs,evidence,input_channel_index,input_channel_count,h,w,(padded_row - pad : U32),(padded_column - pad : U32),proof,shape,channel,positive_width)
