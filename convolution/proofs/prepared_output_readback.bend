import ../../lib/proofs/storage_shape.bend as Shape
# Instantiate gather's premises from the actual root prepare and allocation.
import Base
import ./input_storage.bend as Inputs
import ../../lib/traversal_partition.bend as Partition
import ../../lib/storage_buffer.bend as Storage
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../inference.bend as NativeConv
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/u32_comparison.bend as Comparison
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./prepared_packing_samples.bend as Prepared
import ./packing_storage_refinement.bend as PackingStorage
import ./tile_partition_bounds.bend as TilePartitionBounds
import ./tile_partition_coverage.bend as TilePartitionCoverage
import ./tile_layout.bend as TileLayout
import ./reduction_coordinate_bounds.bend as Coordinates
import ./packed_output_bounds.bend as Output
import ./output_values.bend as Values
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./packed_output_allocation.bend as PackedOutputAllocation
import ../../lib/proofs/array_allocation_bounds.bend as ArrayAllocation
import ./gather_storage_model.bend as GatherModel
import ./gather_tree_readback.bend as GatherTreeReadback
import ./gather_leaf_writes.bend as GatherLeafWrites
import ./gather_copy_coordinates.bend as GatherCopyCoordinates

def selected_depth(+p: ConvSpec.Problem,depth: U32) -> Nat:
  U32.to_nat(U32.min(depth,(U32.from_nat(Storage.bit_length(32n,(Padded.padded(p) / 8 : U32))) - 1 : U32)))

def prepared(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,depth: U32) -> PackingStorage.Prepared:
  Prepared.prepare(problem,selected_depth(problem,depth),0,Padded.padded(problem),Prepared.original_input(problem,storage_contract))

def tiles(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,depth: U32) -> PackingStorage.Tiles():
  PackingStorage.prepared_tiles(prepared(problem,storage_contract,depth))

def extract_certificates(+problem: ConvSpec.Problem,+ready: PackingStorage.Prepared,certificate: Prepared.PreparedCertificates(problem,ready)) ->
  Prepared.AllCertificates(problem,PackingStorage.prepared_tiles(ready)):
  PackingStorage.Prepared{input,packed} = ready
  certificate

def certificates(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+depth: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  Prepared.AllCertificates(problem,tiles(problem,storage_contract,depth)):
  +facts = Guards.valid(problem,valid)
  +fast = Support.fast(limits,problem,supported)
  +span = Padded.padded(problem)
  +span_exact = Padded.padded_exact(limits,problem,facts,fast)
  +span_bound = {%Equal.sym(Nat,U32.to_nat(span),Support.padded(problem),span_exact) : {Nat.is_le(_,Support.padded(problem)) == True{} : Bool}
    Order.reflexive(Support.padded(problem)) : {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool}}
  extract_certificates(problem,prepared(problem,storage_contract,depth),Prepared.prepare_certificates(limits,problem,storage_contract,selected_depth(problem,depth),0,span,valid,supported,
    TilePartitionBounds.BlockAligned{0n,{==}},TilePartitionBounds.BlockAligned{Nat.div(Nat.add(GatherCopyCoordinates.spatial(problem),7n),8n),span_exact},Padded.root_room(limits,problem,facts,fast),span_bound,span_bound))

def extracted_layout(+ready: PackingStorage.Prepared) -> {TileLayout.tiles(PackingStorage.prepared_tiles(ready)) == TileLayout.prepared(ready) : TileLayout.Layout()}:
  PackingStorage.Prepared{input,packed} = ready
  {==}

def layout_exact(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+depth: U32) ->
  {TileLayout.tiles(tiles(problem,storage_contract,depth)) == TileLayout.divide(selected_depth(problem,depth),0,Padded.padded(problem)) : TileLayout.Layout()}:
  %Equal.sym(TileLayout.Layout(),TileLayout.tiles(tiles(problem,storage_contract,depth)),TileLayout.prepared(prepared(problem,storage_contract,depth)),extracted_layout(prepared(problem,storage_contract,depth))) :
    {_ == TileLayout.divide(selected_depth(problem,depth),0,Padded.padded(problem)) : TileLayout.Layout()}
  TileLayout.prepare(selected_depth(problem,depth),0,Padded.padded(problem),Prepared.original_input(problem,storage_contract),U32.from_nat(Guards.height(problem)),U32.from_nat(Guards.width(problem)),
    U32.from_nat(Guards.kernel(problem)),U32.from_nat(Guards.stride(problem)),U32.from_nat(Guards.pad(problem)),U32.from_nat(ConvSpec.out_width(problem)),Coordinates.native_inner(problem))

def position_member(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+depth: U32,+output_row_index: Nat,+output_column_index: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  row_inside: {Nat.is_lt(output_row_index,ConvSpec.out_height(problem)) == True{} : Bool},column_inside: {Nat.is_lt(output_column_index,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  Members.Member(Nat,Coordinates.position(problem,output_row_index,output_column_index),TilePartitionCoverage.positions(TileLayout.tiles(tiles(problem,storage_contract,depth)))):
  +facts = Guards.valid(problem,valid)
  +fast = Support.fast(limits,problem,supported)
  %Equal.sym(TileLayout.Layout(),TileLayout.tiles(tiles(problem,storage_contract,depth)),TileLayout.divide(selected_depth(problem,depth),0,Padded.padded(problem)),layout_exact(problem,storage_contract,depth)) :
    Members.Member(Nat,Coordinates.position(problem,output_row_index,output_column_index),TilePartitionCoverage.positions(_))
  %Equal.sym(List<&2,Nat>,TilePartitionCoverage.positions(TileLayout.divide(selected_depth(problem,depth),0,Padded.padded(problem))),Ranges.range(U32.to_nat(Padded.padded(problem)),0n),
    TilePartitionCoverage.split_tiles_cover_range(selected_depth(problem,depth),0,Padded.padded(problem),Machine.capacity(limits),Limits.capacity_fits(limits),Padded.root_room(limits,problem,facts,fast))) :
    Members.Member(Nat,Coordinates.position(problem,output_row_index,output_column_index),_)
  %Equal.sym(Nat,U32.to_nat(Padded.padded(problem)),Support.padded(problem),Padded.padded_exact(limits,problem,facts,fast)) :
    Members.Member(Nat,Coordinates.position(problem,output_row_index,output_column_index),Ranges.range(_,0n))
  Members.range_member(Support.padded(problem),0n,Coordinates.position(problem,output_row_index,output_column_index),Bounds.nat_trans(Coordinates.position(problem,output_row_index,output_column_index),GatherCopyCoordinates.spatial(problem),Support.padded(problem),
    GatherLeafWrites.position_inside(problem,output_row_index,output_column_index,row_inside,column_inside),Padded.spatial_within_padded(limits,problem,facts,fast)))

def count(+problem: ConvSpec.Problem) -> U32:
  (U32.from_nat(Guards.output_channel_count(problem)) * Padded.spatial(problem) : U32)

def initial(+problem: ConvSpec.Problem) -> Tree.Tree<F32>:
  Tree.fresh(F32,Storage.allocation_depth(count(problem)),0.0)

def count_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) -> {U32.to_nat(count(problem)) == Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem)) : Nat}:
  +facts = Guards.valid(problem,valid)
  +fast = Support.fast(limits,problem,supported)
  %Padded.spatial_exact(limits,problem,facts,fast) : {U32.to_nat(count(problem)) == Nat.mul(Guards.output_channel_count(problem),_) : Nat}
  Output.count_exact(limits,problem,Padded.spatial(problem),fast,
    %Equal.sym(Nat,U32.to_nat(Padded.spatial(problem)),GatherCopyCoordinates.spatial(problem),Padded.spatial_exact(limits,problem,facts,fast)) : {Nat.is_le(_,Support.padded(problem)) == True{} : Bool}
    Padded.spatial_within_padded(limits,problem,facts,fast))

def query_inside_count(+problem: ConvSpec.Problem,+channel: Nat,+output_row_index: Nat,+output_column_index: Nat,
  channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},row_inside: {Nat.is_lt(output_row_index,ConvSpec.out_height(problem)) == True{} : Bool},column_inside: {Nat.is_lt(output_column_index,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Nat.is_lt(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index),Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem))) == True{} : Bool}:
  RowMajor.radix(channel,Guards.output_channel_count(problem),Coordinates.position(problem,output_row_index,output_column_index),GatherCopyCoordinates.spatial(problem),channel_inside,GatherLeafWrites.position_inside(problem,output_row_index,output_column_index,row_inside,column_inside))

def allocated(+limits: Machine.Limits,+problem: ConvSpec.Problem,+channel: Nat,+output_row_index: Nat,+output_column_index: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},row_inside: {Nat.is_lt(output_row_index,ConvSpec.out_height(problem)) == True{} : Bool},column_inside: {Nat.is_lt(output_column_index,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  PackedOutputAllocation.Ready<count(problem),Storage.allocation_depth(count(problem))>:
  +exact = count_exact(limits,problem,valid,supported)
  PackedOutputAllocation.from_allocation(count(problem),ArrayAllocation.allocated(count(problem),Machine.capacity(limits),Limits.capacity_fits(limits),Limits.capacity_allocatable(limits),
    %Equal.sym(Nat,U32.to_nat(count(problem)),Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem)),exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.le_lt_trans(0n,GatherLeafWrites.query(problem,channel,output_row_index,output_column_index),Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem)),Order.zero_le(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index)),query_inside_count(problem,channel,output_row_index,output_column_index,channel_inside,row_inside,column_inside)),
    %Equal.sym(Nat,U32.to_nat(count(problem)),Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem)),exact) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    Padded.scalar_output_bound(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)))))

def allocated_gather_readback(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+packed: PackingStorage.Tiles(),+channel: Nat,+output_row_index: Nat,+output_column_index: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+certificates: Prepared.AllCertificates(problem,packed),
  member: Members.Member(Nat,Coordinates.position(problem,output_row_index,output_column_index),TilePartitionCoverage.positions(TileLayout.tiles(packed))),
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+row_inside: {Nat.is_lt(output_row_index,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column_index,ConvSpec.out_width(problem)) == True{} : Bool},
  allocation: PackedOutputAllocation.Ready<count(problem),Storage.allocation_depth(count(problem))>) ->
  {Tree.read(F32,GatherModel.gather(GatherTreeReadback.result(problem,storage_contract,packed),initial(problem),Padded.spatial(problem)),U32.from_nat(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index))) == Values.matrix_value(problem,channel,output_row_index,output_column_index) : F32}:
  PackedOutputAllocation.Ready{depth_bound,room} = allocation
  +room = {%count_exact(limits,problem,valid,supported) : {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,initial(problem)))) == True{} : Bool}
    room : {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem)),U32.to_nat(Tree.capacity(F32,initial(problem)))) == True{} : Bool}}
  +inside = query_inside_count(problem,channel,output_row_index,output_column_index,channel_inside,row_inside,column_inside)
  +query_exact = Bounds.roundtrip(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index),Machine.capacity(limits),Limits.capacity_fits(limits),Bounds.strict_le(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index),U32.to_nat(Machine.capacity(limits)),
    Bounds.nat_trans(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index),Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem)),U32.to_nat(Machine.capacity(limits)),inside,Padded.scalar_output_bound(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported))))))
  GatherTreeReadback.readback_with_evidence(problem,storage_contract,packed,initial(problem),Storage.allocation_depth(count(problem)),U32.from_nat(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index)),channel,output_row_index,output_column_index,
    Shape.fresh(F32,Storage.allocation_depth(count(problem)),0.0),depth_bound,
    Bounds.native_bound(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index),Tree.capacity(F32,initial(problem)),query_exact,Bounds.nat_trans(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index),Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem)),U32.to_nat(Tree.capacity(F32,initial(problem))),inside,room)),query_exact,
    GatherTreeReadback.tree_evidence(limits,problem,storage_contract,packed,channel,output_row_index,output_column_index,Tree.capacity(F32,initial(problem)),valid,supported,certificates,room,row_inside,column_inside),
    GatherTreeReadback.tree_coverage(limits,problem,storage_contract,packed,channel,output_row_index,output_column_index,valid,supported,certificates,member,channel_inside,row_inside,column_inside))

def finish_leaf_readback(+identity: Bool,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+input: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+channel: Nat,+output_row_index: Nat,+output_column_index: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+certificate: Prepared.LeafCertificate(problem,offset,span,input),
  member: Members.Member(Nat,Coordinates.position(problem,output_row_index,output_column_index),Ranges.range(U32.to_nat(span),U32.to_nat(offset))),
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+row_inside: {Nat.is_lt(output_row_index,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column_index,ConvSpec.out_width(problem)) == True{} : Bool},
  +guard: {U32.is_zero(offset) && U32.is_eq(span,Padded.spatial(problem)) == identity : Bool}) ->
  {Tree.read(F32,GatherModel.finish_leaf(identity,GatherLeafWrites.output(problem,storage_contract,offset,span),offset,span,U32.from_nat(Guards.output_channel_count(problem)),Padded.spatial(problem)),U32.from_nat(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index))) == Values.matrix_value(problem,channel,output_row_index,output_column_index) : F32}:
  match identity:
    case False{}:
      allocated_gather_readback(limits,problem,storage_contract,Partition.Leaf{input,offset,span,Unit{}},channel,output_row_index,output_column_index,valid,supported,certificate,member,channel_inside,row_inside,column_inside,
        allocated(limits,problem,channel,output_row_index,output_column_index,valid,supported,channel_inside,row_inside,column_inside))
    case True{}:
      +offset_zero = Order.equal_of_is_eq(U32.to_nat(offset),0n,
        %Comparison.eq(offset,0) : {_ == True{} : Bool}
        Boolean.left(U32.is_zero(offset),U32.is_eq(span,Padded.spatial(problem)),guard))
      +span_exact = Equal.trans(Nat,U32.to_nat(span),U32.to_nat(Padded.spatial(problem)),GatherCopyCoordinates.spatial(problem),
        Order.equal_of_is_eq(U32.to_nat(span),U32.to_nat(Padded.spatial(problem)),
          %Comparison.eq(span,Padded.spatial(problem)) : {_ == True{} : Bool}
          Boolean.right(U32.is_zero(offset),U32.is_eq(span,Padded.spatial(problem)),guard)),Padded.spatial_exact(limits,problem,Guards.valid(problem,valid),Support.fast(limits,problem,supported)))
      %span_exact : {Tree.read(F32,GatherLeafWrites.output(problem,storage_contract,offset,span),U32.from_nat(Nat.add(Nat.mul(channel,_),Coordinates.position(problem,output_row_index,output_column_index)))) == Values.matrix_value(problem,channel,output_row_index,output_column_index) : F32}
      MicroLeaf.leaf_every_position(limits,problem,storage_contract,input,offset,span,channel,Coordinates.position(problem,output_row_index,output_column_index),output_row_index,output_column_index,valid,supported,certificate,channel_inside,
        %Equal.sym(Nat,U32.to_nat(span),GatherCopyCoordinates.spatial(problem),span_exact) : {Nat.is_lt(Coordinates.position(problem,output_row_index,output_column_index),_) == True{} : Bool}
        GatherLeafWrites.position_inside(problem,output_row_index,output_column_index,row_inside,column_inside),
        %Equal.sym(Nat,U32.to_nat(offset),0n,offset_zero) : {Nat.add(_,Coordinates.position(problem,output_row_index,output_column_index)) == Coordinates.position(problem,output_row_index,output_column_index) : Nat}
        {==},row_inside,column_inside)

def finish_readback(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+packed: PackingStorage.Tiles(),+channel: Nat,+output_row_index: Nat,+output_column_index: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+certificates: Prepared.AllCertificates(problem,packed),
  member: Members.Member(Nat,Coordinates.position(problem,output_row_index,output_column_index),TilePartitionCoverage.positions(TileLayout.tiles(packed))),
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+row_inside: {Nat.is_lt(output_row_index,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column_index,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Tree.read(F32,GatherModel.finish_result(GatherTreeReadback.result(problem,storage_contract,packed),U32.from_nat(Guards.output_channel_count(problem)),Padded.spatial(problem)),U32.from_nat(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index))) == Values.matrix_value(problem,channel,output_row_index,output_column_index) : F32}:
  match packed:
    case Partition.Leaf{+input,+offset,+span,Unit{}}:
      finish_leaf_readback(U32.is_zero(offset) && U32.is_eq(span,Padded.spatial(problem)),limits,problem,storage_contract,input,offset,span,channel,output_row_index,output_column_index,valid,supported,certificates,member,channel_inside,row_inside,column_inside,{==})
    case Partition.Fork{+left,+right}:
      allocated_gather_readback(limits,problem,storage_contract,Partition.Fork{left,right},channel,output_row_index,output_column_index,valid,supported,certificates,member,channel_inside,row_inside,column_inside,
        allocated(limits,problem,channel,output_row_index,output_column_index,valid,supported,channel_inside,row_inside,column_inside))

def output(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,depth: U32) -> Tree.Tree<F32>:
  GatherModel.finish_result(GatherTreeReadback.result(problem,storage_contract,tiles(problem,storage_contract,depth)),U32.from_nat(Guards.output_channel_count(problem)),Padded.spatial(problem))

def every_output_matches_matrix(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+depth: U32,+channel: Nat,+output_row_index: Nat,+output_column_index: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+row_inside: {Nat.is_lt(output_row_index,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column_index,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Tree.read(F32,output(problem,storage_contract,depth),U32.from_nat(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index))) == Values.matrix_value(problem,channel,output_row_index,output_column_index) : F32}:
  finish_readback(limits,problem,storage_contract,tiles(problem,storage_contract,depth),channel,output_row_index,output_column_index,valid,supported,certificates(limits,problem,storage_contract,depth,valid,supported),
    position_member(limits,problem,storage_contract,depth,output_row_index,output_column_index,valid,supported,row_inside,column_inside),channel_inside,row_inside,column_inside)
