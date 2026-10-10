import ../../lib/proofs/storage_shape.bend as Shape
# Every leaf of actual prepare has the frozen sample.
import Base
import ./tile_array_projection.bend as ArrayProjection
import ./input_storage.bend as Inputs
import ../../lib/traversal_partition.bend as Partition
import ../../lib/storage_buffer.bend as Storage
import ../../lib/math_sequence.bend as Sequence
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../../lib/kernel_tile.bend as Tile
import ../scalar_kernel.bend as ScalarConv
import ../inference.bend as NativeConv
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./list_array_storage.bend as ListStorage
import ./packing_storage_refinement.bend as PackingStorage
import ./packing_tile_bounds.bend as PackingBounds
import ./tile_partition_bounds.bend as TilePartitionBounds
import ./reduction_coordinate_bounds.bend as Coordinates
import ./packed_reduction_samples.bend as ReductionSamples
import ./packing_readback.bend as PackingReadback
import ./packing_sample_refinement.bend as PackingSampleRefinement
import ./supported_padded_coordinates.bend as Padded
import ./packing_tabulation.bend as PackingTabulation
import ./packing_contents.bend as PackingContents
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/nat_division_order.bend as DivisionOrder

def original_input(+p: ConvSpec.Problem,storage_contract: Inputs.Contract<p>) -> Tree.Tree<F32>:
  Inputs.input_tree(p,storage_contract)

def workspace_pack(+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+offset: U32,+span: U32,workspace: Tree.Tree<Tile.Tile8>) -> PackingStorage.Packed:
  PackingStorage.packing(U32.to_nat(PackingBounds.count(span,p)),offset,original_input(p,storage_contract),workspace,
    U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),U32.from_nat(Guards.stride(p)),
    U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p)),Coordinates.native_inner(p))

def allocated_pack(+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+offset: U32,+span: U32) -> PackingStorage.Packed:
  workspace_pack(p,storage_contract,offset,span,PackingReadback.allocated(PackingBounds.count(span,p)))

def packed_output(pair: PackingStorage.Packed) -> Tree.Tree<Tile.Tile8>:
  PackingStorage.Packed{input,output} = pair
  output

def LeafSamples(+p: ConvSpec.Problem,+offset: U32,+span: U32,+data: Tree.Tree<Tile.Tile8>) -> Type:
  @+lane: Sequence.Index(8n) ->
  @+group: Nat -> @+input_channel_index: Nat -> @+output_row_index: Nat -> @+output_column_index: Nat -> @+kernel_row_index: Nat -> @+kernel_column_index: Nat ->
  @group_inside: {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool} ->
  @position: {PackingReadback.logical_position(offset,group,lane) == Coordinates.position(p,output_row_index,output_column_index) : Nat} ->
  @input_channel_in_bounds: {Nat.is_lt(input_channel_index,Guards.input_channel_count(p)) == True{} : Bool} ->
  @output_row_in_bounds: {Nat.is_lt(output_row_index,ConvSpec.out_height(p)) == True{} : Bool} ->
  @output_column_in_bounds: {Nat.is_lt(output_column_index,ConvSpec.out_width(p)) == True{} : Bool} ->
  @kernel_row_in_bounds: {Nat.is_lt(kernel_row_index,Guards.kernel(p)) == True{} : Bool} ->
  @kernel_column_in_bounds: {Nat.is_lt(kernel_column_index,Guards.kernel(p)) == True{} : Bool} ->
  {ReductionSamples.select(lane,Tree.read(Tile.Tile8,data,U32.from_nat(PackingSampleRefinement.packing_index(p,group,input_channel_index,kernel_row_index,kernel_column_index)))) == ConvSpec.sample(p,input_channel_index,output_row_index,output_column_index,
      kernel_row_index,kernel_column_index) : F32}

# Duplicable evidence records the actual buffer's provenance, not a closure.
# Sample and dot proofs can independently consume this same checked fact.
type LeafEvidence<-p: ConvSpec.Problem,-offset: U32,-span: U32,-data: Tree.Tree<Tile.Tile8>> is Data:
  LeafEvidence{storage_contract: Inputs.Contract<p>,span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(p)) == True{} : Bool},alignment: TilePartitionBounds.BlockAligned<offset>,
    span_alignment: TilePartitionBounds.BlockAligned<span>,range_bound: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(p)) == True{} : Bool},
    contents: {data == packed_output(allocated_pack(p,storage_contract,offset,span)) : Tree.Tree<Tile.Tile8>}}

def LeafCertificate(p: ConvSpec.Problem,offset: U32,span: U32,data: Tree.Tree<Tile.Tile8>) -> Data:
  LeafEvidence<p,offset,span,data>

def AllCertificates(+p: ConvSpec.Problem,tiles: PackingStorage.Tiles()) -> Data:
  match tiles:
    case Partition.Leaf{data,offset,span,Unit{}}:
      LeafCertificate(p,offset,span,data)
    case Partition.Fork{left,right}:
      Shape.Both<AllCertificates(p,left),AllCertificates(p,right)>

def PreparedCertificates(+p: ConvSpec.Problem,result: PackingStorage.Prepared) -> Data:
  PackingStorage.Prepared{input,tiles} = result
  AllCertificates(p,tiles)

def workspace_tabulated(+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+offset: U32,+span: U32,workspace: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  PackingTabulation.fill(U32.to_nat(PackingBounds.count(span,p)),0,offset,original_input(p,storage_contract),workspace,
    U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),U32.from_nat(Guards.stride(p)),
    U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p)),Coordinates.native_inner(p))

# The packed buffer of a leaf is the flat tabulation of its entries.
def workspace_contents(+limits: Machine.Limits,+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+offset: U32,+span: U32,+workspace: Tree.Tree<Tile.Tile8>,
  +valid: Guards.Valid<p>,+support: Support.FastSupported<limits,p>,+span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(p)) == True{} : Bool},
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),Nat.mul(PackingBounds.groups(span),8n)),Nat.add(PackingContents.grid(p),7n)) == True{} : Bool}) ->
  {packed_output(workspace_pack(p,storage_contract,offset,span,workspace)) == workspace_tabulated(p,storage_contract,offset,span,workspace) : Tree.Tree<Tile.Tile8>}:
  +count = PackingBounds.count(span,p)
  +input = original_input(p,storage_contract)
  +buffer = workspace
  +entries = Nat.mul(PackingBounds.groups(span),Coordinates.inner(p))
  %Equal.sym(Nat,U32.to_nat(count),entries,PackingBounds.count_exact(limits,p,span,valid,support,span_bound)) :
    {packed_output(PackingStorage.packing(_,offset,input,buffer,U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),U32.from_nat(Guards.stride(p)),
    U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p)),Coordinates.native_inner(p))) ==
      PackingTabulation.fill(_,0,offset,input,buffer,U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),U32.from_nat(Guards.stride(p)),
    U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p)),Coordinates.native_inner(p)) : Tree.Tree<Tile.Tile8>}
  Equal.cong(PackingStorage.Packed,Tree.Tree<Tile.Tile8>,pair => packed_output(pair),PackingStorage.packing(entries,offset,input,buffer,U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),U32.from_nat(Guards.stride(p)),
    U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p)),Coordinates.native_inner(p)),
    PackingStorage.Packed{input,PackingTabulation.fill(entries,0,offset,input,buffer,U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),U32.from_nat(Guards.stride(p)),
    U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p)),Coordinates.native_inner(p))},
    PackingContents.packing_contents(limits,p,PackingBounds.groups(span),offset,input,buffer,valid,Padded.scalar(limits,p,support),
      PackingBounds.elements_bound(limits,p,span,support,span_bound),range))

def tabulated(+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+offset: U32,+span: U32) -> Tree.Tree<Tile.Tile8>:
  workspace_tabulated(p,storage_contract,offset,span,PackingReadback.allocated(PackingBounds.count(span,p)))

def allocated_contents(+limits: Machine.Limits,+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+offset: U32,+span: U32,
  +valid: Guards.Valid<p>,+support: Support.FastSupported<limits,p>,+span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(p)) == True{} : Bool},
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),Nat.mul(PackingBounds.groups(span),8n)),Nat.add(PackingContents.grid(p),7n)) == True{} : Bool}) ->
  {packed_output(allocated_pack(p,storage_contract,offset,span)) == tabulated(p,storage_contract,offset,span) : Tree.Tree<Tile.Tile8>}:
  workspace_contents(limits,p,storage_contract,offset,span,PackingReadback.allocated(PackingBounds.count(span,p)),valid,support,span_bound,range)

def sample_inside(+limits: Machine.Limits,+p: ConvSpec.Problem,+span: U32,+valid: Guards.Valid<p>,+support: Support.FastSupported<limits,p>,
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(p)) == True{} : Bool},+group: Nat,+input_channel: Nat,+kernel_row: Nat,+kernel_column: Nat,
  group_inside: {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool},channel_inside: {Nat.is_lt(input_channel,Guards.input_channel_count(p)) == True{} : Bool},
  row_inside: {Nat.is_lt(kernel_row,Guards.kernel(p)) == True{} : Bool},column_inside: {Nat.is_lt(kernel_column,Guards.kernel(p)) == True{} : Bool}) ->
  {Nat.is_lt(PackingSampleRefinement.packing_index(p,group,input_channel,kernel_row,kernel_column),U32.to_nat(PackingBounds.count(span,p))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(PackingBounds.count(span,p)),Nat.mul(PackingBounds.groups(span),Coordinates.inner(p)),PackingBounds.count_exact(limits,p,span,valid,support,span_bound)) :
    {Nat.is_lt(PackingSampleRefinement.packing_index(p,group,input_channel,kernel_row,kernel_column),_) == True{} : Bool}
  RowMajor.radix(group,PackingBounds.groups(span),Coordinates.reduction(input_channel,kernel_row,kernel_column,Guards.kernel(p)),Coordinates.inner(p),group_inside,
    Coordinates.reduction_inside(p,input_channel,kernel_row,kernel_column,channel_inside,row_inside,column_inside))

def leaf_lane(+limits: Machine.Limits,+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+offset: U32,+span: U32,
  +valid: {ConvSpec.valid(p) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,p) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(p)) == True{} : Bool},+lane: Sequence.Index(8n),
  +group: Nat,+input_channel_index: Nat,+output_row_index: Nat,+output_column_index: Nat,+kernel_row_index: Nat,+kernel_column_index: Nat,
  +group_inside: {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool},
  +position: {PackingReadback.logical_position(offset,group,lane) == Coordinates.position(p,output_row_index,output_column_index) : Nat},
  +input_channel_in_bounds: {Nat.is_lt(input_channel_index,Guards.input_channel_count(p)) == True{} : Bool},
  +output_row_in_bounds: {Nat.is_lt(output_row_index,ConvSpec.out_height(p)) == True{} : Bool},
  +output_column_in_bounds: {Nat.is_lt(output_column_index,ConvSpec.out_width(p)) == True{} : Bool},
  +kernel_row_in_bounds: {Nat.is_lt(kernel_row_index,Guards.kernel(p)) == True{} : Bool},
  +kernel_column_in_bounds: {Nat.is_lt(kernel_column_index,Guards.kernel(p)) == True{} : Bool}) ->
  {ReductionSamples.select(lane,Tree.read(Tile.Tile8,tabulated(p,storage_contract,offset,span),U32.from_nat(PackingSampleRefinement.packing_index(p,group,input_channel_index,kernel_row_index,kernel_column_index)))) ==
    ConvSpec.sample(p,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index) : F32}:
  +valid_facts = Guards.valid(p,valid)
  +support_facts = Support.fast(limits,p,supported)
  +count = PackingBounds.count(span,p)
  PackingSampleRefinement.logical_packing_lane_matches_sample(limits,p,storage_contract,count,offset,lane,group,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index,
    valid,Support.scalar_guard(limits,p,supported),PackingBounds.count_bound(limits,p,span,valid_facts,support_facts,span_bound),
    sample_inside(limits,p,span,valid_facts,support_facts,span_bound,group,input_channel_index,kernel_row_index,kernel_column_index,group_inside,input_channel_in_bounds,kernel_row_in_bounds,kernel_column_in_bounds),
    position,input_channel_in_bounds,output_row_in_bounds,output_column_in_bounds,kernel_row_in_bounds,kernel_column_in_bounds)

def workspace_lane(+limits: Machine.Limits,+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+offset: U32,+span: U32,
  +workspace: Tree.Tree<Tile.Tile8>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,workspace),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(PackingBounds.count(span,p)),U32.to_nat(Tree.capacity(Tile.Tile8,workspace))) == True{} : Bool},
  +valid: {ConvSpec.valid(p) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,p) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(p)) == True{} : Bool},+lane: Sequence.Index(8n),
  +group: Nat,+input_channel: Nat,+output_row: Nat,+output_column: Nat,+kernel_row: Nat,+kernel_column: Nat,
  +group_inside: {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool},
  +position: {PackingReadback.logical_position(offset,group,lane) == Coordinates.position(p,output_row,output_column) : Nat},
  +channel_inside: {Nat.is_lt(input_channel,Guards.input_channel_count(p)) == True{} : Bool},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(p)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column,ConvSpec.out_width(p)) == True{} : Bool},
  +kernel_row_inside: {Nat.is_lt(kernel_row,Guards.kernel(p)) == True{} : Bool},+kernel_column_inside: {Nat.is_lt(kernel_column,Guards.kernel(p)) == True{} : Bool}) ->
  {ReductionSamples.select(lane,Tree.read(Tile.Tile8,workspace_tabulated(p,storage_contract,offset,span,workspace),U32.from_nat(PackingSampleRefinement.packing_index(p,group,input_channel,kernel_row,kernel_column)))) ==
    ConvSpec.sample(p,input_channel,output_row,output_column,kernel_row,kernel_column) : F32}:
  +count = PackingBounds.count(span,p)
  +index = PackingSampleRefinement.packing_index(p,group,input_channel,kernel_row,kernel_column)
  +query = U32.from_nat(index)
  +bounded = PackingBounds.count_bound(limits,p,span,Guards.valid(p,valid),Support.fast(limits,p,supported),span_bound)
  +inside = sample_inside(limits,p,span,Guards.valid(p,valid),Support.fast(limits,p,supported),span_bound,group,input_channel,kernel_row,kernel_column,group_inside,channel_inside,kernel_row_inside,kernel_column_inside)
  +query_exact = PackingSampleRefinement.packing_index_exact(limits,p,count,group,input_channel,kernel_row,kernel_column,bounded,inside)
  Equal.trans(F32,ReductionSamples.select(lane,Tree.read(Tile.Tile8,workspace_tabulated(p,storage_contract,offset,span,workspace),query)),
    ReductionSamples.select(lane,Tree.read(Tile.Tile8,tabulated(p,storage_contract,offset,span),query)),ConvSpec.sample(p,input_channel,output_row,output_column,kernel_row,kernel_column),
    PackingReadback.workspace_readback(limits,count,offset,original_input(p,storage_contract),workspace,U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),U32.from_nat(Guards.stride(p)),U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p)),Coordinates.native_inner(p),lane,query,depth,
      ArrayProjection.balanced(depth,workspace,lane,balanced),depth_ok,
      %Equal.sym(U32,Tree.capacity(F32,ArrayProjection.project(workspace,lane)),Tree.capacity(Tile.Tile8,workspace),ArrayProjection.capacity(workspace,lane)) : {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool} room,
      Order.le_lt_trans(0n,index,U32.to_nat(count),Order.zero_le(index),inside),bounded,
      %Equal.sym(Nat,U32.to_nat(query),index,query_exact) : {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool} inside),
    leaf_lane(limits,p,storage_contract,offset,span,valid,supported,span_bound,lane,group,input_channel,output_row,output_column,kernel_row,kernel_column,group_inside,position,channel_inside,row_inside,column_inside,kernel_row_inside,kernel_column_inside))

def workspace_samples(+limits: Machine.Limits,+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+offset: U32,+span: U32,
  +workspace: Tree.Tree<Tile.Tile8>,+depth: Nat,+balanced: Shape.Balanced(Tile.Tile8,depth,workspace),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(PackingBounds.count(span,p)),U32.to_nat(Tree.capacity(Tile.Tile8,workspace))) == True{} : Bool},
  +valid: {ConvSpec.valid(p) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,p) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(p)) == True{} : Bool},
  +range: {Nat.is_le(Nat.add(U32.to_nat(offset),Nat.mul(PackingBounds.groups(span),8n)),Nat.add(PackingContents.grid(p),7n)) == True{} : Bool}) ->
  LeafSamples(p,offset,span,packed_output(workspace_pack(p,storage_contract,offset,span,workspace))):
  %Equal.sym(Tree.Tree<Tile.Tile8>,packed_output(workspace_pack(p,storage_contract,offset,span,workspace)),workspace_tabulated(p,storage_contract,offset,span,workspace),
    workspace_contents(limits,p,storage_contract,offset,span,workspace,Guards.valid(p,valid),Support.fast(limits,p,supported),span_bound,range)) : LeafSamples(p,offset,span,_)
  lane => group => input_channel => output_row => output_column => kernel_row => kernel_column => group_inside => position => channel_inside => row_inside => column_inside => kernel_row_inside => kernel_column_inside =>
    workspace_lane(limits,p,storage_contract,offset,span,workspace,depth,balanced,depth_ok,room,valid,supported,span_bound,lane,group,input_channel,output_row,output_column,kernel_row,kernel_column,group_inside,position,channel_inside,row_inside,column_inside,kernel_row_inside,kernel_column_inside)

# Every leaf of prepare, at any offset and output width, has the frozen sample.
def leaf_samples(+limits: Machine.Limits,+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+offset: U32,+span: U32,
  +valid: {ConvSpec.valid(p) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,p) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(p)) == True{} : Bool},
  +range_bound: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Nat.add(PackingContents.grid(p),7n)) == True{} : Bool}) ->
  LeafSamples(p,offset,span,packed_output(allocated_pack(p,storage_contract,offset,span))):
  +valid_facts = Guards.valid(p,valid)
  +support_facts = Support.fast(limits,p,supported)
  +range = Order.transitive(Nat.add(U32.to_nat(offset),Nat.mul(PackingBounds.groups(span),8n)),Nat.add(U32.to_nat(offset),U32.to_nat(span)),Nat.add(PackingContents.grid(p),7n),
    Order.offset_le(U32.to_nat(offset),Nat.mul(PackingBounds.groups(span),8n),U32.to_nat(span),TilePartitionBounds.quotient_product(span,8n,{==})),range_bound)
  %Equal.sym(Tree.Tree<Tile.Tile8>,packed_output(allocated_pack(p,storage_contract,offset,span)),tabulated(p,storage_contract,offset,span),
    allocated_contents(limits,p,storage_contract,offset,span,valid_facts,support_facts,span_bound,range)) : LeafSamples(p,offset,span,_)
  lane => group => input_channel_index => output_row_index => output_column_index => kernel_row_index => kernel_column_index => group_inside => position => input_channel_in_bounds => output_row_in_bounds => output_column_in_bounds => kernel_row_in_bounds => kernel_column_in_bounds =>
    leaf_lane(limits,p,storage_contract,offset,span,valid,supported,span_bound,lane,group,input_channel_index,output_row_index,output_column_index,kernel_row_index,kernel_column_index,group_inside,position,input_channel_in_bounds,output_row_in_bounds,output_column_in_bounds,kernel_row_in_bounds,kernel_column_in_bounds)

# A leaf range under the padded grid ends within seven of the grid.
def padded_range(+p: ConvSpec.Problem,+offset: U32,+span: U32,range_bound: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(p)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Nat.add(PackingContents.grid(p),7n)) == True{} : Bool}:
  Order.transitive(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(p),Nat.add(PackingContents.grid(p),7n),range_bound,
    DivisionOrder.product_below(Nat.add(PackingContents.grid(p),7n),8n,{==}))

def prepare(+p: ConvSpec.Problem,depth: Nat,offset: U32,span: U32,input: Tree.Tree<F32>) -> PackingStorage.Prepared:
  PackingStorage.prepare(depth,offset,span,input,U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),
    U32.from_nat(Guards.stride(p)),U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p)),Coordinates.native_inner(p))

def leaf_certificate(+p: ConvSpec.Problem,+offset: U32,+span: U32,+packed: PackingStorage.Packed,proof: LeafCertificate(p,offset,span,
    packed_output(packed))) -> PreparedCertificates(p,PackingStorage.leaf(offset,span,packed)):
  PackingStorage.Packed{input,output} = packed
  proof

def attach_certificates(+p: ConvSpec.Problem,+left: PackingStorage.Tiles(),+right: PackingStorage.Prepared,left_proof: AllCertificates(p,left),
  right_proof: PreparedCertificates(p,right)) -> PreparedCertificates(p,PackingStorage.attach(left,right)):
  PackingStorage.Prepared{input,tiles} = right
  Shape.Both{left_proof,right_proof}

def after_left_certificates(+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+depth: Nat,+offset: U32,+span: U32,+left: PackingStorage.Prepared,
  left_proof: PreparedCertificates(p,left),same_input: {PackingStorage.prepared_input(left) == original_input(p,storage_contract) : Tree.Tree<F32>},
  right_proof: PreparedCertificates(p,prepare(p,depth,(offset + TilePartitionBounds.half(span) : U32),(span - TilePartitionBounds.half(span) : U32),
      original_input(p,storage_contract)))) -> PreparedCertificates(p,PackingStorage.after_left(depth,offset,span,U32.from_nat(Guards.height(p)),
    U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),U32.from_nat(Guards.stride(p)),U32.from_nat(Guards.pad(p)),
    U32.from_nat(ConvSpec.out_width(p)),Coordinates.native_inner(p),left)):
  PackingStorage.Prepared{+input,+tiles} = left
  attach_certificates(p,tiles,prepare(p,depth,(offset + TilePartitionBounds.half(span) : U32),(span - TilePartitionBounds.half(span) : U32),input),left_proof,
    %Equal.sym(Tree.Tree<F32>,input,original_input(p,storage_contract),same_input) : PreparedCertificates(p,prepare(p,depth,
        (offset + TilePartitionBounds.half(span) : U32),(span - TilePartitionBounds.half(span) : U32),_))
    right_proof)

def prepare_certificates(+limits: Machine.Limits,+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+depth: Nat,+offset: U32,+span: U32,
  +valid: {ConvSpec.valid(p) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,p) == True{} : Bool},
  +alignment: TilePartitionBounds.BlockAligned<offset>,+span_alignment: TilePartitionBounds.BlockAligned<span>,+room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(p)) == True{} : Bool},
  +range_bound: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(p)) == True{} : Bool}) -> PreparedCertificates(p,prepare(p,depth,offset,span,original_input(p,storage_contract))):
  match depth:
    case 0n:
      leaf_certificate(p,offset,span,allocated_pack(p,storage_contract,offset,span),LeafEvidence{storage_contract,span_bound,alignment,span_alignment,range_bound,{==}})
    case 1n+ +rest:
      +half = TilePartitionBounds.half(span)
      +capacity = Machine.capacity(limits)
      +capacity_ok = Limits.capacity_fits(limits)
      +bounded = Order.transitive(U32.to_nat(span),Support.padded(p),U32.to_nat(capacity),span_bound,Padded.padded_bound(limits,p,
          Guards.valid(p,valid),Support.fast(limits,p,supported)))
      +left_bound = Order.transitive(U32.to_nat(half),U32.to_nat(span),Support.padded(p),TilePartitionBounds.half_le(span,capacity,capacity_ok,
          bounded),span_bound)
      after_left_certificates(p,storage_contract,rest,offset,span,prepare(p,rest,offset,half,original_input(p,storage_contract)),
        prepare_certificates(limits,p,storage_contract,rest,offset,half,valid,supported,alignment,TilePartitionBounds.half_alignment(span,capacity,capacity_ok,bounded),TilePartitionBounds.left_room(offset,span,capacity,capacity_ok,room),left_bound,
          TilePartitionBounds.left_interval_bound(offset,span,capacity,Support.padded(p),capacity_ok,room,range_bound)),
        PackingStorage.prepare_preserves_input(rest,offset,half,original_input(p,storage_contract),U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),
          U32.from_nat(Guards.kernel(p)),U32.from_nat(Guards.stride(p)),U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p)),
          Coordinates.native_inner(p)),
        prepare_certificates(limits,p,storage_contract,rest,(offset + half : U32),(span - half : U32),valid,supported,TilePartitionBounds.right_alignment(offset,span,capacity,capacity_ok,room,alignment),TilePartitionBounds.remainder_alignment(span,capacity,capacity_ok,bounded,span_alignment),TilePartitionBounds.right_room(offset,span,capacity,capacity_ok,room),
          %Equal.sym(Nat,U32.to_nat((span - half : U32)),Nat.sub(U32.to_nat(span),U32.to_nat(half)),TilePartitionBounds.remainder_exact(span,capacity,
              capacity_ok,bounded)) : {Nat.is_le(_,Support.padded(p)) == True{} : Bool}
          Order.transitive(Nat.sub(U32.to_nat(span),U32.to_nat(half)),U32.to_nat(span),Support.padded(p),Order.sub_le(U32.to_nat(span),
              U32.to_nat(half)),span_bound),
          TilePartitionBounds.right_interval_bound(offset,span,capacity,Support.padded(p),capacity_ok,room,range_bound)))

def certified_leaf_samples(+limits: Machine.Limits,+p: ConvSpec.Problem,+offset: U32,+span: U32,+data: Tree.Tree<Tile.Tile8>,
  valid: {ConvSpec.valid(p) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,p) == True{} : Bool},
  certificate: LeafCertificate(p,offset,span,
    data)) -> LeafSamples(p,offset,span,data):
  LeafEvidence{+storage_contract,span_bound,alignment,span_alignment,range_bound,exact_contents} = certificate
  %Equal.sym(Tree.Tree<Tile.Tile8>,data,packed_output(allocated_pack(p,storage_contract,offset,span)),exact_contents) : LeafSamples(p,offset,span,_)
  leaf_samples(limits,p,storage_contract,offset,span,valid,supported,span_bound,padded_range(p,offset,span,range_bound))

# A packed pair exposes the unchanged input and the written workspace.
def packed_pair(+pair: PackingStorage.Packed) ->
  {pair == PackingStorage.Packed{PackingStorage.packed_input(pair),packed_output(pair)} : PackingStorage.Packed}:
  PackingStorage.Packed{input,output} = pair
  {==}

def workspace_pair(+p: ConvSpec.Problem,+storage_contract: Inputs.Contract<p>,+offset: U32,+span: U32,+workspace: Tree.Tree<Tile.Tile8>) ->
  {workspace_pack(p,storage_contract,offset,span,workspace) ==
    PackingStorage.Packed{original_input(p,storage_contract),packed_output(workspace_pack(p,storage_contract,offset,span,workspace))} : PackingStorage.Packed}:
  %PackingStorage.packing_preserves_input(U32.to_nat(PackingBounds.count(span,p)),offset,original_input(p,storage_contract),workspace,
    U32.from_nat(Guards.height(p)),U32.from_nat(Guards.width(p)),U32.from_nat(Guards.kernel(p)),U32.from_nat(Guards.stride(p)),U32.from_nat(Guards.pad(p)),U32.from_nat(ConvSpec.out_width(p)),Coordinates.native_inner(p)) :
    {workspace_pack(p,storage_contract,offset,span,workspace) == PackingStorage.Packed{_,packed_output(workspace_pack(p,storage_contract,offset,span,workspace))} : PackingStorage.Packed}
  packed_pair(workspace_pack(p,storage_contract,offset,span,workspace))
