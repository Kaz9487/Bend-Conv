import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
# A tabulated packed pixel -> convolution sample, with real guard-derived bounds.
import Base
import ./input_storage.bend as Inputs
import ../../lib/storage_buffer.bend as Storage
import ../../lib/math_sequence.bend as Sequence
import ./packing_tile_bounds.bend as PackingBounds
import ../inference.bend as NativeConv
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./list_array_storage.bend as ListStorage
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./reduction_coordinate_bounds.bend as Coordinates
import ./native_weight_refinement.bend as NativeWeightRefinement
import ./native_sample_refinement.bend as NativeSampleRefinement
import ./packing_lane_reads.bend as LaneReads
import ./packing_readback.bend as PackingReadback
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../specification.bend as ConvSpec
import ../scalar_kernel.bend as ScalarConv
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/kernel_packing_samples.bend as PackingSamples
import ../../lib/kernel_tile.bend as Tile
import ./packing_tabulation.bend as PackingTabulation
import ./packed_reduction_samples.bend as ReductionSamples

def masks(+inside: Bool,pair: Array<F32> & F32) -> {PackingSamples.mask_pixel(inside,pair) == Packing.masked(inside,pair) : Array<F32> & F32}:
  (x,v) = pair
  {==}

def decoded(+x: Tree.Tree<F32>,+pos: U32,+q: U32,+h: U32,+w: U32,+k: U32,+stride: U32,+pad: U32,+output_width: U32,+input_channel_index: U32,+output_row_index: U32,+output_column_index: U32,
  +kernel_row_index: U32,+kernel_column_index: U32,eic: {(q / (k * k) : U32) == input_channel_index : U32},eoy: {(pos / output_width : U32) == output_row_index : U32},eox: {(pos % output_width : U32) == output_column_index : U32},
  eky: {((q / k) % k : U32) == kernel_row_index : U32},ekx: {(q % k : U32) == kernel_column_index : U32}) -> {PackingSamples.pixel(Tree.pack(F32,x),pos,q,h,w,k,stride,pad,
    output_width) == Packing.pixel(Tree.pack(F32,x),(output_row_index * stride + kernel_row_index - pad : U32),(output_column_index * stride + kernel_column_index - pad : U32),(input_channel_index * h * w : U32),h,
    w) : Array<F32> & F32}:
  %eic : {PackingSamples.pixel(Tree.pack(F32,x),pos,q,h,w,k,stride,pad,output_width) == Packing.pixel(Tree.pack(F32,x),(output_row_index * stride + kernel_row_index - pad : U32),
      (output_column_index * stride + kernel_column_index - pad : U32),(_ * h * w : U32),h,w) : Array<F32> & F32}
  %eoy : {PackingSamples.pixel(Tree.pack(F32,x),pos,q,h,w,k,stride,pad,output_width) == Packing.pixel(Tree.pack(F32,x),(_ * stride + kernel_row_index - pad : U32),
      (output_column_index * stride + kernel_column_index - pad : U32),((q / (k * k) : U32) * h * w : U32),h,w) : Array<F32> & F32}
  %eox : {PackingSamples.pixel(Tree.pack(F32,x),pos,q,h,w,k,stride,pad,output_width) == Packing.pixel(Tree.pack(F32,x),
      ((pos / output_width : U32) * stride + kernel_row_index - pad : U32),(_ * stride + kernel_column_index - pad : U32),((q / (k * k) : U32) * h * w : U32),h,w) : Array<F32> & F32}
  %eky : {PackingSamples.pixel(Tree.pack(F32,x),pos,q,h,w,k,stride,pad,output_width) == Packing.pixel(Tree.pack(F32,x),
      ((pos / output_width : U32) * stride + _ - pad : U32),((pos % output_width : U32) * stride + kernel_column_index - pad : U32),((q / (k * k) : U32) * h * w : U32),h,
      w) : Array<F32> & F32}
  %ekx : {PackingSamples.pixel(Tree.pack(F32,x),pos,q,h,w,k,stride,pad,output_width) == Packing.pixel(Tree.pack(F32,x),
      ((pos / output_width : U32) * stride + ((q / k) % k : U32) - pad : U32),((pos % output_width : U32) * stride + _ - pad : U32),
      ((q / (k * k) : U32) * h * w : U32),h,w) : Array<F32> & F32}
  masks(U32.is_lt(((pos / output_width) * stride + (q / k) % k - pad : U32),h) && U32.is_lt(((pos % output_width) * stride + q % k - pad : U32),w),Array.get(F32,
      Tree.pack(F32,x),
      ((q / (k * k)) * h * w + ((pos / output_width) * stride + (q / k) % k - pad : U32) * w + ((pos % output_width) * stride + q % k - pad : U32) : U32)))

def native_pixel_decoded(+limits: Machine.Limits,+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+pos: U32,+q: U32,+input_channel_index: Nat,+output_row_index: Nat,+output_column_index: Nat,+kernel_row_index: Nat,+kernel_column_index: Nat,
  +valid: {ConvSpec.valid(p) == True{} : Bool},+supported: {ScalarConv.supported_with_limits(limits,p) == True{} : Bool},
  +epos: {U32.to_nat(pos) == Coordinates.position(p,output_row_index,output_column_index) : Nat},+input_channel_in_bounds: {Nat.is_lt(input_channel_index,Guards.input_channel_count(p)) == True{} : Bool},+output_row_in_bounds: {Nat.is_lt(output_row_index,
      ConvSpec.out_height(p)) == True{} : Bool},+output_column_in_bounds: {Nat.is_lt(output_column_index,ConvSpec.out_width(p)) == True{} : Bool},+kernel_row_in_bounds: {Nat.is_lt(kernel_row_index,
      Guards.kernel(p)) == True{} : Bool},+kernel_column_in_bounds: {Nat.is_lt(kernel_column_index,
      Guards.kernel(p)) == True{} : Bool},
  coordinates: Coordinates.DecodedReduction<q,U32.from_nat(Guards.kernel(p)),input_channel_index,kernel_row_index,kernel_column_index>) -> {PackingSamples.pixel(Inputs.input_owner(p,storage_contract),pos,q,
    U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),
    U32.from_nat(Guards.stride(p)),U32.from_nat(Guards.pad(p)),
    U32.from_nat(ConvSpec.out_width(p))) == (Inputs.input_owner(p,storage_contract),ConvSpec.sample(p,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index)) : Array<F32> & F32}:
  Coordinates.DecodedReduction{ec,ey,ex} = coordinates
  +vf = Guards.valid(p,valid)
  +sf = Support.scalar(limits,p,supported)
  +kn = Guards.kernel(p)
  +uk = U32.from_nat(kn)
  +wn = ConvSpec.out_width(p)
  +uw = U32.from_nat(wn)
  +we = Coordinates.width_exact(limits,p,vf,sf)
  +wp = Coordinates.width_positive(p)
  +wd = Limits.double_bound(limits,wn,Coordinates.width_bound(limits,p,vf,sf))
  +er = Encoded.encoding((pos / uw : U32),output_row_index,Radix.quotient_word(pos,uw,output_row_index,output_column_index,wn,Machine.half_word(limits),
      Limits.half_word_fits(limits),epos,we,wp,output_column_in_bounds,wd))
  +es = Encoded.encoding((pos % uw : U32),output_column_index,Radix.remainder_word(pos,uw,output_row_index,output_column_index,wn,Machine.half_word(limits),
      Limits.half_word_fits(limits),epos,we,wp,output_column_in_bounds,wd))
  %NativeSampleRefinement.native_sample(limits,p,storage_contract,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index,valid,supported,input_channel_in_bounds,output_row_in_bounds,output_column_in_bounds,kernel_row_in_bounds,
    kernel_column_in_bounds) : {PackingSamples.pixel(Inputs.input_owner(p,storage_contract),pos,q,U32.from_nat(Guards.height(p)),
      U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),U32.from_nat(Guards.stride(p)),
      U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p))) == _ : Array<F32> & F32}
  decoded(Inputs.input_tree(p,storage_contract),pos,q,U32.from_nat(Guards.height(p)),
    U32.from_nat(Guards.width(p)),uk,U32.from_nat(Guards.stride(p)),U32.from_nat(Guards.pad(p)),uw,
    U32.from_nat(input_channel_index),U32.from_nat(output_row_index),U32.from_nat(output_column_index),U32.from_nat(kernel_row_index),U32.from_nat(kernel_column_index),ec,er,es,ey,ex)

def native_pixel(+limits: Machine.Limits,+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+pos: U32,+q: U32,+input_channel_index: Nat,+output_row_index: Nat,+output_column_index: Nat,+kernel_row_index: Nat,+kernel_column_index: Nat,
  +valid: {ConvSpec.valid(p) == True{} : Bool},+supported: {ScalarConv.supported_with_limits(limits,p) == True{} : Bool},
  +epos: {U32.to_nat(pos) == Coordinates.position(p,output_row_index,output_column_index) : Nat},+eq: {U32.to_nat(q) == Coordinates.reduction(input_channel_index,kernel_row_index,
      kernel_column_index,Guards.kernel(p)) : Nat},+input_channel_in_bounds: {Nat.is_lt(input_channel_index,Guards.input_channel_count(p)) == True{} : Bool},+output_row_in_bounds: {Nat.is_lt(output_row_index,
      ConvSpec.out_height(p)) == True{} : Bool},+output_column_in_bounds: {Nat.is_lt(output_column_index,ConvSpec.out_width(p)) == True{} : Bool},+kernel_row_in_bounds: {Nat.is_lt(kernel_row_index,
      Guards.kernel(p)) == True{} : Bool},+kernel_column_in_bounds: {Nat.is_lt(kernel_column_index,
      Guards.kernel(p)) == True{} : Bool}) -> {PackingSamples.pixel(Inputs.input_owner(p,storage_contract),pos,q,
    U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),
    U32.from_nat(Guards.stride(p)),U32.from_nat(Guards.pad(p)),
    U32.from_nat(ConvSpec.out_width(p))) == (Inputs.input_owner(p,storage_contract),ConvSpec.sample(p,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index)) : Array<F32> & F32}:
  native_pixel_decoded(limits,p,storage_contract,pos,q,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index,valid,supported,epos,input_channel_in_bounds,output_row_in_bounds,output_column_in_bounds,kernel_row_in_bounds,kernel_column_in_bounds,
    Coordinates.decode_reduction(limits,p,q,input_channel_index,kernel_row_index,kernel_column_index,
      Guards.valid(p,valid),Support.scalar(limits,p,supported),eq,kernel_row_in_bounds,kernel_column_in_bounds))

def second(pair: Array<F32> & F32) -> F32:
  (x,v) = pair
  v

def pure_value(+limits: Machine.Limits,+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+pos: U32,+q: U32,+input_channel_index: Nat,+output_row_index: Nat,+output_column_index: Nat,+kernel_row_index: Nat,+kernel_column_index: Nat,
  +valid: {ConvSpec.valid(p) == True{} : Bool},+supported: {ScalarConv.supported_with_limits(limits,p) == True{} : Bool},
  +epos: {U32.to_nat(pos) == Coordinates.position(p,output_row_index,output_column_index) : Nat},+eq: {U32.to_nat(q) == Coordinates.reduction(input_channel_index,kernel_row_index,
      kernel_column_index,Guards.kernel(p)) : Nat},+input_channel_in_bounds: {Nat.is_lt(input_channel_index,Guards.input_channel_count(p)) == True{} : Bool},+output_row_in_bounds: {Nat.is_lt(output_row_index,
      ConvSpec.out_height(p)) == True{} : Bool},+output_column_in_bounds: {Nat.is_lt(output_column_index,ConvSpec.out_width(p)) == True{} : Bool},+kernel_row_in_bounds: {Nat.is_lt(kernel_row_index,
      Guards.kernel(p)) == True{} : Bool},+kernel_column_in_bounds: {Nat.is_lt(kernel_column_index,
      Guards.kernel(p)) == True{} : Bool}) -> {LaneReads.value(Inputs.input_tree(p,storage_contract),pos,q,U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),U32.from_nat(Guards.stride(p)),U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p))) == ConvSpec.sample(p,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index) : F32}:
  %LaneReads.pixel(Inputs.input_tree(p,storage_contract),pos,q,
    U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),
    U32.from_nat(Guards.stride(p)),U32.from_nat(Guards.pad(p)),
    U32.from_nat(ConvSpec.out_width(p))) : {second(_) == ConvSpec.sample(p,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index) : F32}
  Equal.cong(Array<F32> & F32,F32,second,PackingSamples.pixel(Inputs.input_owner(p,storage_contract),pos,q,
      U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),
      U32.from_nat(Guards.stride(p)),U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p))),
    (Inputs.input_owner(p,storage_contract),ConvSpec.sample(p,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index)),native_pixel(limits,p,storage_contract,pos,q,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index,valid,
      supported,epos,eq,input_channel_in_bounds,output_row_in_bounds,output_column_in_bounds,kernel_row_in_bounds,kernel_column_in_bounds))

law allocated_packing_lane_matches_sample:
  for +limits: Machine.Limits
  for +p: ConvSpec.Problem
  for +storage_contract: Inputs.Contract<p>
  for +count: U32
  for +start: U32
  for +inner: U32
  for +lane: Sequence.Index(8n)
  for +query: U32
  for +input_channel_index: Nat
  for +output_row_index: Nat
  for +output_column_index: Nat
  for +kernel_row_index: Nat
  for +kernel_column_index: Nat
  for +valid: {ConvSpec.valid(p) == True{} : Bool}
  for +supported: {ScalarConv.supported_with_limits(limits,p) == True{} : Bool}
  for +positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}
  for +bounded: {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  for +inside: {Nat.is_lt(U32.to_nat(query),U32.to_nat(count)) == True{} : Bool}
  for +position: {U32.to_nat(PackingReadback.lane_position((start + (query / inner) * 8 : U32),lane)) == Coordinates.position(p,output_row_index,
      output_column_index) : Nat}
  for +reduction: {U32.to_nat((query % inner : U32)) == Coordinates.reduction(input_channel_index,kernel_row_index,kernel_column_index,Guards.kernel(p)) : Nat}
  for +input_channel_in_bounds: {Nat.is_lt(input_channel_index,Guards.input_channel_count(p)) == True{} : Bool}
  for +output_row_in_bounds: {Nat.is_lt(output_row_index,ConvSpec.out_height(p)) == True{} : Bool}
  for +output_column_in_bounds: {Nat.is_lt(output_column_index,ConvSpec.out_width(p)) == True{} : Bool}
  for +kernel_row_in_bounds: {Nat.is_lt(kernel_row_index,Guards.kernel(p)) == True{} : Bool}
  for +kernel_column_in_bounds: {Nat.is_lt(kernel_column_index,Guards.kernel(p)) == True{} : Bool}
  {ReductionSamples.select(lane,Tree.read(Tile.Tile8,PackingTabulation.fill(U32.to_nat(count),0,start,Inputs.input_tree(p,storage_contract),PackingReadback.allocated(count),
        U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),
        U32.from_nat(Guards.stride(p)),U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p)),inner),query)) == ConvSpec.sample(p,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index) : F32}

def allocated_packing_lane_matches_sample(limits,p,storage_contract,count,start,inner,lane,query,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index,valid,supported,positive,bounded,inside,position,
  reduction,input_channel_in_bounds,output_row_in_bounds,output_column_in_bounds,kernel_row_in_bounds,kernel_column_in_bounds):
  +input = Inputs.input_tree(p,storage_contract)
  +height = U32.from_nat(Guards.height(p))
  +width = U32.from_nat(Guards.width(p))
  +kernel = U32.from_nat(Guards.kernel(p))
  +stride = U32.from_nat(Guards.stride(p))
  +padding = U32.from_nat(Guards.pad(p))
  +output_width = U32.from_nat(ConvSpec.out_width(p))
  %pure_value(limits,p,storage_contract,PackingReadback.lane_position((start + (query / inner) * 8 : U32),lane),(query % inner : U32),input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index,valid,
    supported,position,reduction,input_channel_in_bounds,output_row_in_bounds,output_column_in_bounds,kernel_row_in_bounds,kernel_column_in_bounds) : {ReductionSamples.select(lane,
      Tree.read(Tile.Tile8,PackingTabulation.fill(U32.to_nat(count),0,start,input,PackingReadback.allocated(count),height,width,kernel,stride,padding,output_width,inner),query)) == _ : F32}
  PackingReadback.allocated_readback(limits,count,start,input,height,width,kernel,stride,padding,output_width,inner,lane,query,positive,bounded,
    inside)

# The physical vector index uses group-major / reduction-minor order.
def packing_index(+p: ConvSpec.Problem,group: Nat,input_channel_index: Nat,kernel_row_index: Nat,kernel_column_index: Nat) -> Nat:
  Nat.add(Nat.mul(group,Coordinates.inner(p)),Coordinates.reduction(input_channel_index,kernel_row_index,kernel_column_index,Guards.kernel(p)))

def packing_index_exact(+limits: Machine.Limits,+p: ConvSpec.Problem,+count: U32,+group: Nat,+input_channel_index: Nat,+kernel_row_index: Nat,+kernel_column_index: Nat,
  bounded: {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},inside: {Nat.is_lt(packing_index(p,group,
        input_channel_index,kernel_row_index,kernel_column_index),U32.to_nat(count)) == True{} : Bool}) -> {U32.to_nat(U32.from_nat(packing_index(p,group,input_channel_index,kernel_row_index,kernel_column_index))) == packing_index(p,group,
    input_channel_index,kernel_row_index,kernel_column_index) : Nat}:
  Bounds.roundtrip(packing_index(p,group,input_channel_index,kernel_row_index,kernel_column_index),Machine.capacity(limits),Limits.capacity_fits(limits),
    Bounds.strict_le(packing_index(p,group,input_channel_index,kernel_row_index,kernel_column_index),U32.to_nat(Machine.capacity(limits)),Bounds.nat_trans(packing_index(p,
          group,input_channel_index,kernel_row_index,kernel_column_index),U32.to_nat(count),U32.to_nat(Machine.capacity(limits)),inside,bounded)))

def packing_group(+limits: Machine.Limits,+p: ConvSpec.Problem,+count: U32,+group: Nat,+input_channel_index: Nat,+kernel_row_index: Nat,+kernel_column_index: Nat,
  +valid: Guards.Valid<p>,+support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),Machine.coordinate(limits)>,bounded: {Nat.is_le(U32.to_nat(count),
      U32.to_nat(Machine.capacity(limits))) == True{} : Bool},inside: {Nat.is_lt(packing_index(p,group,input_channel_index,kernel_row_index,kernel_column_index),
      U32.to_nat(count)) == True{} : Bool},input_channel_in_bounds: {Nat.is_lt(input_channel_index,Guards.input_channel_count(p)) == True{} : Bool},kernel_row_in_bounds: {Nat.is_lt(kernel_row_index,
      Guards.kernel(p)) == True{} : Bool},kernel_column_in_bounds: {Nat.is_lt(kernel_column_index,
      Guards.kernel(p)) == True{} : Bool}) -> {U32.to_nat((U32.from_nat(packing_index(p,group,input_channel_index,kernel_row_index,
        kernel_column_index)) / Coordinates.native_inner(p) : U32)) == group : Nat}:
  Radix.quotient_word(U32.from_nat(packing_index(p,group,input_channel_index,kernel_row_index,kernel_column_index)),Coordinates.native_inner(p),group,
    Coordinates.reduction(input_channel_index,kernel_row_index,kernel_column_index,Guards.kernel(p)),Coordinates.inner(p),
    Machine.half_word(limits),Limits.half_word_fits(limits),packing_index_exact(limits,p,count,group,input_channel_index,kernel_row_index,kernel_column_index,bounded,inside),
    Coordinates.inner_exact(limits,p,valid,support),Coordinates.inner_positive(p,valid),
    Coordinates.reduction_inside(p,input_channel_index,kernel_row_index,kernel_column_index,input_channel_in_bounds,kernel_row_in_bounds,kernel_column_in_bounds),Limits.double_bound(limits,
      Coordinates.inner(p),Coordinates.inner_bound(limits,p,valid,support)))

def packing_reduction(+limits: Machine.Limits,+p: ConvSpec.Problem,+count: U32,+group: Nat,+input_channel_index: Nat,+kernel_row_index: Nat,+kernel_column_index: Nat,
  +valid: Guards.Valid<p>,+support: Guards.Supported<p,Machine.dimension(limits),
  Machine.capacity(limits),Machine.coordinate(limits)>,bounded: {Nat.is_le(U32.to_nat(count),
      U32.to_nat(Machine.capacity(limits))) == True{} : Bool},inside: {Nat.is_lt(packing_index(p,group,input_channel_index,kernel_row_index,kernel_column_index),
      U32.to_nat(count)) == True{} : Bool},input_channel_in_bounds: {Nat.is_lt(input_channel_index,Guards.input_channel_count(p)) == True{} : Bool},kernel_row_in_bounds: {Nat.is_lt(kernel_row_index,
      Guards.kernel(p)) == True{} : Bool},kernel_column_in_bounds: {Nat.is_lt(kernel_column_index,
      Guards.kernel(p)) == True{} : Bool}) -> {U32.to_nat((U32.from_nat(packing_index(p,group,input_channel_index,kernel_row_index,
        kernel_column_index)) % Coordinates.native_inner(p) : U32)) == Coordinates.reduction(input_channel_index,kernel_row_index,kernel_column_index,
    Guards.kernel(p)) : Nat}:
  Radix.remainder_word(U32.from_nat(packing_index(p,group,input_channel_index,kernel_row_index,kernel_column_index)),Coordinates.native_inner(p),group,
    Coordinates.reduction(input_channel_index,kernel_row_index,kernel_column_index,Guards.kernel(p)),Coordinates.inner(p),
    Machine.half_word(limits),Limits.half_word_fits(limits),packing_index_exact(limits,p,count,group,input_channel_index,kernel_row_index,kernel_column_index,bounded,inside),
    Coordinates.inner_exact(limits,p,valid,support),Coordinates.inner_positive(p,valid),
    Coordinates.reduction_inside(p,input_channel_index,kernel_row_index,kernel_column_index,input_channel_in_bounds,kernel_row_in_bounds,kernel_column_in_bounds),Limits.double_bound(limits,
      Coordinates.inner(p),Coordinates.inner_bound(limits,p,valid,support)))

law logical_packing_lane_matches_sample:
  for +limits: Machine.Limits
  for +p: ConvSpec.Problem
  for +storage_contract: Inputs.Contract<p>
  for +count: U32
  for +start: U32
  for +lane: Sequence.Index(8n)
  for +group: Nat
  for +input_channel_index: Nat
  for +output_row_index: Nat
  for +output_column_index: Nat
  for +kernel_row_index: Nat
  for +kernel_column_index: Nat
  for +valid: {ConvSpec.valid(p) == True{} : Bool}
  for +supported: {ScalarConv.supported_with_limits(limits,p) == True{} : Bool}
  for +bounded: {Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  for +inside: {Nat.is_lt(packing_index(p,group,input_channel_index,kernel_row_index,kernel_column_index),U32.to_nat(count)) == True{} : Bool}
  for +position: {PackingReadback.logical_position(start,group,lane) == Coordinates.position(p,output_row_index,output_column_index) : Nat}
  for +input_channel_in_bounds: {Nat.is_lt(input_channel_index,Guards.input_channel_count(p)) == True{} : Bool}
  for +output_row_in_bounds: {Nat.is_lt(output_row_index,ConvSpec.out_height(p)) == True{} : Bool}
  for +output_column_in_bounds: {Nat.is_lt(output_column_index,ConvSpec.out_width(p)) == True{} : Bool}
  for +kernel_row_in_bounds: {Nat.is_lt(kernel_row_index,Guards.kernel(p)) == True{} : Bool}
  for +kernel_column_in_bounds: {Nat.is_lt(kernel_column_index,Guards.kernel(p)) == True{} : Bool}
  {ReductionSamples.select(lane,Tree.read(Tile.Tile8,PackingTabulation.fill(U32.to_nat(count),0,start,Inputs.input_tree(p,storage_contract),PackingReadback.allocated(count),
        U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),
        U32.from_nat(Guards.stride(p)),U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p)),Coordinates.native_inner(p)),
      U32.from_nat(packing_index(p,group,input_channel_index,kernel_row_index,kernel_column_index)))) == ConvSpec.sample(p,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index) : F32}

def logical_packing_lane_matches_sample(limits,p,storage_contract,count,start,lane,group,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index,valid,supported,bounded,inside,position,input_channel_in_bounds,output_row_in_bounds,output_column_in_bounds,kernel_row_in_bounds,kernel_column_in_bounds):
  +valid_facts = Guards.valid(p,valid)
  +support_facts = Support.scalar(limits,p,supported)
  +query = U32.from_nat(packing_index(p,group,input_channel_index,kernel_row_index,kernel_column_index))
  +inner = Coordinates.native_inner(p)
  allocated_packing_lane_matches_sample(limits,p,storage_contract,count,start,inner,lane,query,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index,valid,supported,Order.le_lt_trans(0n,
      packing_index(p,group,input_channel_index,kernel_row_index,kernel_column_index),U32.to_nat(count),Order.zero_le(packing_index(p,group,input_channel_index,kernel_row_index,kernel_column_index)),inside),bounded,
    %Equal.sym(Nat,U32.to_nat(query),packing_index(p,group,input_channel_index,kernel_row_index,kernel_column_index),packing_index_exact(limits,p,count,group,input_channel_index,kernel_row_index,kernel_column_index,bounded,
        inside)) : {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool}
    inside,
    %Equal.sym(U32,(query / inner : U32),U32.from_nat(group),Encoded.encoding((query / inner : U32),group,packing_group(limits,p,
          count,group,input_channel_index,kernel_row_index,kernel_column_index,valid_facts,support_facts,bounded,inside,input_channel_in_bounds,kernel_row_in_bounds,
          kernel_column_in_bounds))) : {U32.to_nat(PackingReadback.lane_position((start + _ * 8 : U32),lane)) == Coordinates.position(p,output_row_index,output_column_index) : Nat}
    Equal.trans(Nat,U32.to_nat(PackingReadback.lane_position((start + U32.from_nat(group) * 8 : U32),lane)),
      PackingReadback.logical_position(start,group,lane),Coordinates.position(p,output_row_index,output_column_index),PackingReadback.position_exact(start,group,
        lane,Machine.capacity(limits),Limits.capacity_fits(limits),
      %Equal.sym(Nat,PackingReadback.logical_position(start,group,lane),Coordinates.position(p,output_row_index,output_column_index),position) : {Nat.is_le(_,
          U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
      Coordinates.position_bound(limits,p,output_row_index,output_column_index,valid_facts,support_facts,output_row_in_bounds,output_column_in_bounds)),position),
    packing_reduction(limits,p,count,group,input_channel_index,kernel_row_index,kernel_column_index,valid_facts,support_facts,bounded,inside,input_channel_in_bounds,kernel_row_in_bounds,kernel_column_in_bounds),input_channel_in_bounds,output_row_in_bounds,output_column_in_bounds,kernel_row_in_bounds,kernel_column_in_bounds)
