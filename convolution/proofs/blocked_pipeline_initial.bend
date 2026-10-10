# Original validation guards establish the fixed blocked loop and allocation bounds.
import Base
import ../../lib/kernel_window.bend as Window
import ../../lib/proofs/kernel_pipeline_activation.bend as ActivationProofs
import ../../lib/proofs/nat_ceiling.bend as Ceiling
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/machine_limits.bend as Machine
import ../../lib/tensor.bend as Tensors
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/kernel_tile.bend as Tile
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/encoded_coordinates.bend as Encoded
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/u32_shift.bend as Shift
import ../../lib/proofs/checked_word_refinement.bend as Checked
import ../../lib/math_checked_word.bend as CheckedWord
import ../../lib/proofs/boolean_logic.bend as Boolean
import ./blocked_pipeline_bounds.bend as BlockBounds
import ./blocked_pipeline_model.bend as Model
import ./blocked_convolution_readback.bend as Blocked
import ./input_storage.bend as Inputs
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./reduction_coordinate_bounds.bend as Coordinates
import ./packing_tile_bounds.bend as PackingBounds
import ./packing_contents.bend as PackingContents
import ./prepared_packing_samples.bend as Prepared
import ./tile_partition_bounds.bend as Partition
import ./microkernel_inference_refinement.bend as Inference
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./gemm_dot_matches_matrix.bend as GemmDot
import ./prepared_output_readback.bend as PreparedOutput
import ./supported_coordinate_bounds.bend as SupportedCoordinates
import ./output_values.bend as Values

# These are internal arithmetic parameters; public callers supply no new guards.
type Configuration<-limits: Machine.Limits,-target: U32,-allocation_cap: U32,-loop_cap: U32> is Data:
  Configuration{target_positive: {Nat.is_lt(0n,U32.to_nat(target)) == True{} : Bool},
    target_room: {U32.is_le(target,allocation_cap) == True{} : Bool},
    target_span_room: {Nat.is_le(Nat.mul(U32.to_nat(target),8n),U32.to_nat(allocation_cap)) == True{} : Bool},
    allocation_fits: {U32.is_le(allocation_cap,2147483648) == True{} : Bool},allocation_bound: {U32.is_le(allocation_cap,1073741824) == True{} : Bool},
    capacity_room: {U32.is_le(Machine.capacity(limits),allocation_cap) == True{} : Bool},
    loop_fits: {U32.is_le(loop_cap,2147483648) == True{} : Bool},doubled: {U32.to_nat(loop_cap) == Nat.double(U32.to_nat(allocation_cap)) : Nat}}

def span(+target: U32,+problem: ConvSpec.Problem) -> U32:
  (8 * U32.max(1,(target / Coordinates.native_inner(problem) : U32)) : U32)

def tile_count(+target: U32,+problem: ConvSpec.Problem) -> U32:
  U32.max(target,Coordinates.native_inner(problem))

def workspace(+target: U32,+problem: ConvSpec.Problem) -> Tree.Tree<Tile.Tile8>:
  Model.allocated(~Tile.Tile8,Tile.filled(0.0),tile_count(target,problem))

def output_count(+problem: ConvSpec.Problem,positions: U32) -> U32:
  (MicroLeaf.rows(problem) * positions : U32)

type Region<-problem: ConvSpec.Problem,-origin: U32,-positions: U32> is Data:
  Region{within: {Nat.is_le(Nat.add(U32.to_nat(origin),U32.to_nat(positions)),PackingContents.grid(problem)) == True{} : Bool},
    positive: {Nat.is_lt(0n,U32.to_nat(positions)) == True{} : Bool}}

def positions_bound(+problem: ConvSpec.Problem,+origin: U32,+positions: U32,region: Region<problem,origin,positions>) ->
  {Nat.is_le(U32.to_nat(positions),PackingContents.grid(problem)) == True{} : Bool}:
  Region{within,positive} = region
  Order.transitive(U32.to_nat(positions),Nat.add(U32.to_nat(origin),U32.to_nat(positions)),PackingContents.grid(problem),(%Order.commutative(U32.to_nat(positions),U32.to_nat(origin)) : {Nat.is_le(U32.to_nat(positions),_) == True{} : Bool} Order.le_add(U32.to_nat(positions),U32.to_nat(origin))),within)

def previous(+problem: ConvSpec.Problem,+positions: U32) -> Tree.Tree<F32>:
  Model.allocated(~F32,0.0,output_count(problem,positions))

def inner_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+fast: Support.FastSupported<limits,problem>) ->
  {Nat.is_le(U32.to_nat(Coordinates.native_inner(problem)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(Coordinates.native_inner(problem)),Coordinates.inner(problem),Coordinates.inner_exact(limits,problem,valid,Padded.scalar(limits,problem,fast))) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  Coordinates.inner_bound(limits,problem,valid,Padded.scalar(limits,problem,fast))

def inner_positive(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+fast: Support.FastSupported<limits,problem>) ->
  {Nat.is_lt(0n,U32.to_nat(Coordinates.native_inner(problem))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(Coordinates.native_inner(problem)),Coordinates.inner(problem),Coordinates.inner_exact(limits,problem,valid,Padded.scalar(limits,problem,fast))) : {Nat.is_lt(0n,_) == True{} : Bool}
  Coordinates.inner_positive(problem,valid)

def sum_room(+left: Nat,+right: Nat,+allocation_cap: U32,+loop_cap: U32,+left_room: {Nat.is_le(left,U32.to_nat(allocation_cap)) == True{} : Bool},
  +right_room: {Nat.is_le(right,U32.to_nat(allocation_cap)) == True{} : Bool},+doubled: {U32.to_nat(loop_cap) == Nat.double(U32.to_nat(allocation_cap)) : Nat}) ->
  {Nat.is_le(Nat.add(left,right),U32.to_nat(loop_cap)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(loop_cap),Nat.double(U32.to_nat(allocation_cap)),doubled) : {Nat.is_le(Nat.add(left,right),_) == True{} : Bool}
  %Addition.double(U32.to_nat(allocation_cap)) : {Nat.is_le(Nat.add(left,right),_) == True{} : Bool}
  Encoded.sum_bounds(left,right,U32.to_nat(allocation_cap),U32.to_nat(allocation_cap),left_room,right_room)

type LoopReady<-limits: Machine.Limits,-problem: ConvSpec.Problem,-origin: U32,-positions: U32,-target: U32,-loop_cap: U32> is Data:
  LoopReady{geometry: Blocked.Geometry<limits,problem,origin,positions,Window.padded(positions),span(target,problem),loop_cap>,
    schedule: BlockBounds.Schedule<U32.to_nat(((Window.padded(positions) + span(target,problem) - 1) / span(target,problem) : U32)),0n,U32.to_nat(Window.padded(positions)),U32.to_nat(span(target,problem))>,
    prefix_room: {Nat.is_le(Nat.mul(PackingBounds.groups(span(target,problem)),Coordinates.inner(problem)),U32.to_nat(tile_count(target,problem))) == True{} : Bool},
    tiles: Depth.Sized<tile_count(target,problem)>,output: Depth.Sized<output_count(problem,positions)>}

def output_product_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+positions: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+region: Region<problem,origin,positions>) ->
  {Nat.is_le(Nat.mul(U32.to_nat(MicroLeaf.rows(problem)),U32.to_nat(positions)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  +fast = Support.fast(limits,problem,supported)
  %Equal.sym(Nat,U32.to_nat(MicroLeaf.rows(problem)),Guards.output_channel_count(problem),SupportedCoordinates.co_exact(limits,problem,Padded.scalar(limits,problem,fast))) : {Nat.is_le(Nat.mul(_,U32.to_nat(positions)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  Order.transitive(Nat.mul(Guards.output_channel_count(problem),U32.to_nat(positions)),Nat.mul(Guards.output_channel_count(problem),PackingContents.grid(problem)),U32.to_nat(Machine.capacity(limits)),
    (%RowMajor.multiply_commutative(U32.to_nat(positions),Guards.output_channel_count(problem)) : {Nat.is_le(_,Nat.mul(Guards.output_channel_count(problem),PackingContents.grid(problem))) == True{} : Bool}
    %RowMajor.multiply_commutative(PackingContents.grid(problem),Guards.output_channel_count(problem)) : {Nat.is_le(Nat.mul(U32.to_nat(positions),Guards.output_channel_count(problem)),_) == True{} : Bool}
    RowMajor.mul_le(U32.to_nat(positions),PackingContents.grid(problem),Guards.output_channel_count(problem),positions_bound(problem,origin,positions,region))),Padded.scalar_output_bound(limits,problem,Padded.scalar(limits,problem,fast)))

def output_ready(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+positions: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+region: Region<problem,origin,positions>) -> Depth.Sized<output_count(problem,positions)>:
  ActivationProofs.product_ready(limits,MicroLeaf.rows(problem),positions,output_product_room(limits,problem,origin,positions,valid,supported,region))

def prepare(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+target: U32,+allocation_cap: U32,+loop_cap: U32,
  +configuration: Configuration<limits,target,allocation_cap,loop_cap>,+valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+region: Region<problem,origin,positions>) -> LoopReady<limits,problem,origin,positions,target,loop_cap>:
  Configuration{+target_positive,+target_room,+target_span_room,+allocation_fits,+allocation_bound,+capacity_room,+loop_fits,+doubled} = configuration
  Region{+within,+positions_positive} = region
  +facts = Guards.valid(problem,valid)
  +fast = Support.fast(limits,problem,supported)
  +inner = Coordinates.native_inner(problem)
  +inner_bound = inner_room(limits,problem,facts,fast)
  +inner_positive = inner_positive(limits,problem,facts,fast)
  +inner_exact = Coordinates.inner_exact(limits,problem,facts,Padded.scalar(limits,problem,fast))
  +groups = BlockBounds.groups(target,inner)
  +span = span(target,problem)
  +span_exact = BlockBounds.span_exact(target,target_positive,limits,inner,allocation_cap,allocation_fits,target_span_room,inner_positive,inner_bound)
  +span_room = {
    %Equal.sym(Nat,U32.to_nat(span),Nat.mul(groups,8n),span_exact) : {Nat.is_le(_,U32.to_nat(allocation_cap)) == True{} : Bool}
    Order.transitive(Nat.mul(groups,8n),Nat.mul(U32.to_nat(target),8n),U32.to_nat(allocation_cap),BlockBounds.span_room(target,target_positive,limits,inner,inner_bound,inner_positive),target_span_room) : {Nat.is_le(U32.to_nat(span),U32.to_nat(allocation_cap)) == True{} : Bool}}
  +span_positive = {
    %Equal.sym(Nat,U32.to_nat(span),Nat.mul(groups,8n),span_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Support.positive_mul(groups,8n,BlockBounds.groups_positive(target,inner),{==}) : {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool}}
  +region = {Region{within,positions_positive} : Region<problem,origin,positions>}
  +grid_room = Padded.spatial_bound_facts(limits,problem,facts,Padded.scalar(limits,problem,fast))
  +positions_grid = positions_bound(problem,origin,positions,region)
  +positions_room = Order.transitive(U32.to_nat(positions),PackingContents.grid(problem),U32.to_nat(Machine.capacity(limits)),positions_grid,grid_room)
  +padded = Window.padded(positions)
  +padded_exact = BlockBounds.padded_exact(limits,positions,positions_room)
  +padded_bound = {%Equal.sym(Nat,U32.to_nat(padded),BlockBounds.ceiling_eight(U32.to_nat(positions)),padded_exact) : {Nat.is_le(_,Support.padded(problem)) == True{} : Bool}
    BlockBounds.padded_le(U32.to_nat(positions),PackingContents.grid(problem),positions_grid) : {Nat.is_le(U32.to_nat(padded),Support.padded(problem)) == True{} : Bool}}
  +padded_end = {%Equal.sym(Nat,U32.to_nat(padded),BlockBounds.ceiling_eight(U32.to_nat(positions)),padded_exact) : {Nat.is_le(_,Nat.add(U32.to_nat(positions),7n)) == True{} : Bool}
    DivisionOrder.product_below(Nat.add(U32.to_nat(positions),7n),8n,{==}) : {Nat.is_le(U32.to_nat(padded),Nat.add(U32.to_nat(positions),7n)) == True{} : Bool}}
  +origin_end = {Order.transitive(Nat.add(U32.to_nat(origin),U32.to_nat(padded)),Nat.add(U32.to_nat(origin),Nat.add(U32.to_nat(positions),7n)),Nat.add(PackingContents.grid(problem),7n),
    Order.offset_le(U32.to_nat(origin),U32.to_nat(padded),Nat.add(U32.to_nat(positions),7n),padded_end),
    %Addition.associative(U32.to_nat(origin),U32.to_nat(positions),7n) : {Nat.is_le(_,Nat.add(PackingContents.grid(problem),7n)) == True{} : Bool}
    Order.add_le(Nat.add(U32.to_nat(origin),U32.to_nat(positions)),PackingContents.grid(problem),7n,within)) : {Nat.is_le(Nat.add(U32.to_nat(origin),U32.to_nat(padded)),Nat.add(PackingContents.grid(problem),7n)) == True{} : Bool}}
  +pack_room = Order.transitive(Nat.add(U32.to_nat(origin),U32.to_nat(padded)),Nat.add(PackingContents.grid(problem),7n),U32.to_nat(loop_cap),origin_end,
    Padded.add_in_double(PackingContents.grid(problem),allocation_cap,loop_cap,doubled,
      Order.transitive(PackingContents.grid(problem),U32.to_nat(Machine.capacity(limits)),U32.to_nat(allocation_cap),grid_room,Bounds.as_nat_le(Machine.capacity(limits),allocation_cap,capacity_room)),
      Order.transitive(7n,U32.to_nat(Machine.capacity(limits)),U32.to_nat(allocation_cap),Bounds.as_nat_le(7,Machine.capacity(limits),Limits.seven_within_capacity(limits)),Bounds.as_nat_le(Machine.capacity(limits),allocation_cap,capacity_room))))
  +padded_room = Order.transitive(U32.to_nat(padded),Support.padded(problem),U32.to_nat(allocation_cap),padded_bound,
    Order.transitive(Support.padded(problem),U32.to_nat(Machine.capacity(limits)),U32.to_nat(allocation_cap),Padded.padded_bound(limits,problem,facts,fast),Bounds.as_nat_le(Machine.capacity(limits),allocation_cap,capacity_room)))
  +sum_room = sum_room(U32.to_nat(padded),U32.to_nat(span),allocation_cap,loop_cap,padded_room,span_room,doubled)
  +divisor_room = {
    %Equal.sym(Nat,U32.to_nat(loop_cap),Nat.double(U32.to_nat(allocation_cap)),doubled) : {Nat.is_le(Nat.double(U32.to_nat(span)),_) == True{} : Bool}
    Order.double_le(U32.to_nat(span),U32.to_nat(allocation_cap),span_room) : {Nat.is_le(Nat.double(U32.to_nat(span)),U32.to_nat(loop_cap)) == True{} : Bool}}
  +padded_positive = {%Equal.sym(Nat,U32.to_nat(padded),BlockBounds.ceiling_eight(U32.to_nat(positions)),padded_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Bounds.nat_trans(0n,U32.to_nat(positions),BlockBounds.ceiling_eight(U32.to_nat(positions)),positions_positive,BlockBounds.within_ceiling(U32.to_nat(positions))) : {Nat.is_lt(0n,U32.to_nat(padded)) == True{} : Bool}}
  +remaining = U32.to_nat(((padded + span - 1) / span : U32))
  +remaining_exact = BlockBounds.remaining_exact(padded,span,loop_cap,loop_fits,padded_positive,span_positive,sum_room,divisor_room)
  +tile_count = tile_count(target,problem)
  +tile_room = BlockBounds.tile_count_room(target,inner,allocation_cap,target_room,Order.transitive(U32.to_nat(inner),U32.to_nat(Machine.capacity(limits)),U32.to_nat(allocation_cap),inner_bound,Bounds.as_nat_le(Machine.capacity(limits),allocation_cap,capacity_room)))
  LoopReady{
    Blocked.Geometry{origin_end,pack_room,valid,supported,loop_fits,padded_bound,padded_end,
      Order.transitive(U32.to_nat(positions),U32.to_nat(Machine.capacity(limits)),U32.to_nat(Machine.half_word(limits)),positions_room,Bounds.as_nat_le(Machine.capacity(limits),Machine.half_word(limits),Limits.capacity_within_half(limits))),
      sum_room,span_positive,Partition.BlockAligned{Nat.div(Nat.add(U32.to_nat(positions),7n),8n),padded_exact},Partition.BlockAligned{groups,span_exact}},
    %Equal.sym(Nat,remaining,Ceiling.ceiling(U32.to_nat(padded),U32.to_nat(span)),remaining_exact) : BlockBounds.Schedule<_,0n,U32.to_nat(padded),U32.to_nat(span)>
    BlockBounds.initial_schedule(U32.to_nat(padded),U32.to_nat(span),padded_positive,span_positive),
    %Equal.sym(Nat,PackingBounds.groups(span),groups,MicroLeaf.groups_exact(span,groups,span_exact)) : {Nat.is_le(Nat.mul(_,Coordinates.inner(problem)),U32.to_nat(tile_count)) == True{} : Bool}
    %inner_exact : {Nat.is_le(Nat.mul(groups,_),U32.to_nat(tile_count)) == True{} : Bool}
    BlockBounds.tile_prefix_room(target,limits,inner,inner_bound,inner_positive),
    Depth.bounded(tile_count,Arithmetic.count_within(tile_count,U32.to_nat(tile_count),allocation_cap,1073741824,{==},tile_room,allocation_bound)),
    output_ready(limits,problem,origin,positions,valid,supported,region)}

def eight_product(+target: U32,+product: U32,accepted: {CheckedWord.multiply(target,8) == Some{product} : Maybe<&2,U32>}) ->
  {Nat.mul(U32.to_nat(target),8n) == U32.to_nat(product) : Nat}:
  Checked.multiply_sound(target,8,product,accepted)

def eight_room(+target: U32,+product: U32,+cap: U32,+accepted: {CheckedWord.multiply(target,8) == Some{product} : Maybe<&2,U32>},room: {U32.is_le(product,cap) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(target),8n),U32.to_nat(cap)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(U32.to_nat(target),8n),U32.to_nat(product),eight_product(target,product,accepted)) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
  Bounds.as_nat_le(product,cap,room)

def doubled_cap(+small: U32,+large: U32,+exact: {U32.shl(small) == large : U32},bounded: {U32.is_lt(small,2147483648) == True{} : Bool}) ->
  {U32.to_nat(large) == Nat.double(U32.to_nat(small)) : Nat}:
  %exact : {U32.to_nat(_) == Nat.double(U32.to_nat(small)) : Nat}
  Shift.double(small,bounded)

def configuration(+limits: Machine.Limits) -> Configuration<limits,8192,1073741824,2147483648>:
  Configuration{{==},{==},eight_room(8192,65536,1073741824,{==},{==}),{==},{==},Limits.capacity_allocatable(limits),{==},doubled_cap(1073741824,2147483648,{==},{==})}

def output_count_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+region: Region<problem,origin,positions>) ->
  {U32.to_nat(output_count(problem,positions)) == Nat.mul(U32.to_nat(MicroLeaf.rows(problem)),U32.to_nat(positions)) : Nat}:
  ActivationProofs.product_exact(limits,MicroLeaf.rows(problem),positions,output_product_room(limits,problem,origin,positions,valid,supported,region))

def memory(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+target: U32,+valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+region: Region<problem,origin,positions>,
  +prefix_room: {Nat.is_le(Nat.mul(PackingBounds.groups(span(target,problem)),Coordinates.inner(problem)),U32.to_nat(tile_count(target,problem))) == True{} : Bool},
  tile_ready: Depth.Sized<tile_count(target,problem)>,output_ready: Depth.Sized<output_count(problem,positions)>) ->
  Blocked.Memory<problem,origin,positions,span(target,problem),workspace(target,problem),previous(problem,positions),Tensors.depth_for(tile_count(target,problem)),Tensors.depth_for(output_count(problem,positions))>:
  Depth.Sized{+tile_depth_ok,+tile_room} = tile_ready
  Depth.Sized{+output_depth_ok,+output_room} = output_ready
  +tile_depth = Tensors.depth_for(tile_count(target,problem))
  +output_depth = Tensors.depth_for(output_count(problem,positions))
  +tile_balance = Shape.fresh(Tile.Tile8,tile_depth,Tile.filled(0.0))
  +output_balance = Shape.fresh(F32,output_depth,0.0)
  Blocked.Memory{tile_balance,tile_depth_ok,
    %Equal.sym(U32,Tree.capacity(Tile.Tile8,workspace(target,problem)),U32.shln(1,tile_depth),Shape.capacity(Tile.Tile8,tile_depth,workspace(target,problem),tile_balance)) : {Nat.is_le(Nat.mul(PackingBounds.groups(span(target,problem)),Coordinates.inner(problem)),U32.to_nat(_)) == True{} : Bool}
    Order.transitive(Nat.mul(PackingBounds.groups(span(target,problem)),Coordinates.inner(problem)),U32.to_nat(tile_count(target,problem)),U32.to_nat(U32.shln(1,tile_depth)),prefix_room,tile_room),
    output_balance,output_depth_ok,
    %Equal.sym(U32,Tree.capacity(F32,previous(problem,positions)),U32.shln(1,output_depth),Shape.capacity(F32,output_depth,previous(problem,positions),output_balance)) : {Nat.is_le(Nat.mul(U32.to_nat(MicroLeaf.rows(problem)),U32.to_nat(positions)),U32.to_nat(_)) == True{} : Bool}
    %output_count_exact(limits,problem,origin,positions,valid,supported,region) : {Nat.is_le(_,U32.to_nat(U32.shln(1,output_depth))) == True{} : Bool}
    output_room}

def point(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+region: Region<problem,origin,positions>,
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool},
  +lower: {Nat.is_le(U32.to_nat(origin),Coordinates.position(problem,row,column)) == True{} : Bool},
  +inside: {Nat.is_lt(Blocked.local_position(problem,origin,row,column),U32.to_nat(positions)) == True{} : Bool}) ->
  Blocked.Point<problem,origin,positions,channel,row,column,Window.padded(positions)>:
  +room = Order.transitive(U32.to_nat(positions),PackingContents.grid(problem),U32.to_nat(Machine.capacity(limits)),positions_bound(problem,origin,positions,region),
    Padded.spatial_bound_facts(limits,problem,Guards.valid(problem,valid),Padded.scalar(limits,problem,Support.fast(limits,problem,supported))))
  Blocked.Point{Order.restore_shift(U32.to_nat(origin),Coordinates.position(problem,row,column),lower),channel_inside,row_inside,column_inside,inside,
    %Equal.sym(Nat,U32.to_nat(Window.padded(positions)),BlockBounds.ceiling_eight(U32.to_nat(positions)),BlockBounds.padded_exact(limits,positions,room)) : {Nat.is_lt(Blocked.local_position(problem,origin,row,column),_) == True{} : Bool}
    Bounds.nat_trans(Blocked.local_position(problem,origin,row,column),U32.to_nat(positions),BlockBounds.ceiling_eight(U32.to_nat(positions)),inside,BlockBounds.within_ceiling(U32.to_nat(positions)))}

def finished(~custom: F32 -> F32,+activation: Activations.Activation,+count: U32,+state: Model.State) ->
  {Model.finish(~custom,activation,count,state) == ActivationProofs.activation_tree(~custom,activation,count,Blocked.result(state)) : Tree.Tree<F32>}:
  Model.State{input,tiles,weights,bias,output} = state
  {==}

def completed(+target: U32,+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+storage: Inputs.Contract<problem>) -> Model.State:
  Model.blocks(U32.to_nat(((Window.padded(positions) + span(target,problem) - 1) / span(target,problem) : U32)),Blocked.work(problem,positions),origin,0,Window.padded(positions),span(target,problem),Blocked.state(problem,storage,workspace(target,problem),previous(problem,positions)))

def run_expanded(~custom: F32 -> F32,+activation: Activations.Activation,+target: U32,+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+storage: Inputs.Contract<problem>) ->
  {Model.run_with_target(~custom,target,activation,Blocked.work(problem,positions),origin,Inputs.input_tree(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)) ==
    ActivationProofs.activation_tree(~custom,activation,output_count(problem,positions),Blocked.result(completed(target,problem,origin,positions,storage))) : Tree.Tree<F32>}:
  ConvSpec.Problem{shape,+input_values,+weight_values,+bias_values} = problem
  ConvSpec.Shape{+input_channels,+output_channels,+height,+width,+kernel,+stride,+padding} = shape
  +problem = {ConvSpec.Problem{ConvSpec.Shape{input_channels,output_channels,height,width,kernel,stride,padding},input_values,weight_values,bias_values} : ConvSpec.Problem}
  finished(~custom,activation,output_count(problem,positions),completed(target,problem,origin,positions,storage))

def query(+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+channel: Nat,+row: Nat,+column: Nat) -> Nat:
  Nat.add(Nat.mul(channel,U32.to_nat(positions)),Blocked.local_position(problem,origin,row,column))

def activated(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+target: U32,+storage: Inputs.Contract<problem>,+channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+region: Region<problem,origin,positions>,
  +point: Blocked.Point<problem,origin,positions,channel,row,column,Window.padded(positions)>,
  output_ready: Depth.Sized<output_count(problem,positions)>,evidence: Blocked.LoopEvidence<problem,origin,positions,channel,row,column,Tensors.depth_for(output_count(problem,positions)),Blocked.result(completed(target,problem,origin,positions,storage))>) ->
  {Tree.read(F32,ActivationProofs.activation_tree(~custom,activation,output_count(problem,positions),Blocked.result(completed(target,problem,origin,positions,storage))),Blocked.address(problem,origin,positions,channel,row,column)) == ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}:
  Blocked.Point{restored,+channel_inside,row_inside,column_inside,+inside,padded_inside} = point
  Depth.Sized{+depth_ok,+room} = output_ready
  Blocked.LoopEvidence{+balanced,+value} = evidence
  +output = Blocked.result(completed(target,problem,origin,positions,storage))
  +count = output_count(problem,positions)
  +depth = Tensors.depth_for(count)
  Equal.trans(F32,Tree.read(F32,ActivationProofs.activation_tree(~custom,activation,count,output),Blocked.address(problem,origin,positions,channel,row,column)),ActivationProofs.activation_value(~custom,activation,Tree.read(F32,output,Blocked.address(problem,origin,positions,channel,row,column))),ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)),
    ActivationProofs.activated_cases(~custom,activation,count,output,depth,Shape.capacity(F32,depth,output,balanced),balanced,depth_ok,ActivationProofs.power_fits(depth,depth_ok),room,query(problem,origin,positions,channel,row,column),
      %Equal.sym(Nat,U32.to_nat(count),Nat.mul(U32.to_nat(MicroLeaf.rows(problem)),U32.to_nat(positions)),output_count_exact(limits,problem,origin,positions,valid,supported,region)) : {Nat.is_lt(query(problem,origin,positions,channel,row,column),_) == True{} : Bool}
      RowMajor.radix(channel,U32.to_nat(MicroLeaf.rows(problem)),Blocked.local_position(problem,origin,row,column),U32.to_nat(positions),
        %Equal.sym(Nat,U32.to_nat(MicroLeaf.rows(problem)),Guards.output_channel_count(problem),SupportedCoordinates.co_exact(limits,problem,Padded.scalar(limits,problem,Support.fast(limits,problem,supported)))) : {Nat.is_lt(channel,_) == True{} : Bool} channel_inside,inside)),
    Equal.cong(F32,F32,value => ActivationProofs.activation_value(~custom,activation,value),Tree.read(F32,output,Blocked.address(problem,origin,positions,channel,row,column)),Values.matrix_value(problem,channel,row,column),value))

def ready_readback(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+target: U32,+loop_cap: U32,+storage: Inputs.Contract<problem>,+channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+region: Region<problem,origin,positions>,
  +point: Blocked.Point<problem,origin,positions,channel,row,column,Window.padded(positions)>,
  ready: LoopReady<limits,problem,origin,positions,target,loop_cap>) ->
  {Tree.read(F32,Model.run_with_target(~custom,target,activation,Blocked.work(problem,positions),origin,Inputs.input_tree(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)),Blocked.address(problem,origin,positions,channel,row,column)) == ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}:
  LoopReady{+geometry,+schedule,+prefix_room,+tile_ready,+output_ready} = ready
  %Equal.sym(Tree.Tree<F32>,Model.run_with_target(~custom,target,activation,Blocked.work(problem,positions),origin,Inputs.input_tree(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)),ActivationProofs.activation_tree(~custom,activation,output_count(problem,positions),Blocked.result(completed(target,problem,origin,positions,storage))),run_expanded(~custom,activation,target,problem,origin,positions,storage)) :
    {Tree.read(F32,_,Blocked.address(problem,origin,positions,channel,row,column)) == ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}
  activated(~custom,activation,limits,problem,origin,positions,target,storage,channel,row,column,valid,supported,region,point,output_ready,
    %Arithmetic.add_zero(origin) : Blocked.LoopEvidence<problem,origin,positions,channel,row,column,Tensors.depth_for(output_count(problem,positions)),Blocked.result(Model.blocks(U32.to_nat(((Window.padded(positions) + span(target,problem) - 1) / span(target,problem) : U32)),Blocked.work(problem,positions),_,0,Window.padded(positions),span(target,problem),Blocked.state(problem,storage,workspace(target,problem),previous(problem,positions))))>
    Blocked.loop_readback(U32.to_nat(((Window.padded(positions) + span(target,problem) - 1) / span(target,problem) : U32)),limits,problem,origin,positions,storage,0,Window.padded(positions),span(target,problem),loop_cap,workspace(target,problem),previous(problem,positions),Tensors.depth_for(tile_count(target,problem)),Tensors.depth_for(output_count(problem,positions)),channel,row,column,geometry,
      memory(limits,problem,origin,positions,target,valid,supported,region,prefix_room,tile_ready,output_ready),point,schedule,Partition.BlockAligned{0n,{==}},
      %Equal.sym(Bool,Nat.is_lt(Blocked.local_position(problem,origin,row,column),0n),False{},Division.lt_zero(Blocked.local_position(problem,origin,row,column))) : Boolean.When(_,{Tree.read(F32,previous(problem,positions),Blocked.address(problem,origin,positions,channel,row,column)) == Values.matrix_value(problem,channel,row,column) : F32}) Unit{}))

# The runtime uses the fixed 256 KiB target and one activation pass.
def run_readback(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+positions: U32,+storage: Inputs.Contract<problem>,+channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+region: Region<problem,origin,positions>,
  +point: Blocked.Point<problem,origin,positions,channel,row,column,Window.padded(positions)>) ->
  {Tree.read(F32,Model.run(~custom,activation,Blocked.work(problem,positions),origin,Inputs.input_tree(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)),Blocked.address(problem,origin,positions,channel,row,column)) == ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}:
  ready_readback(~custom,activation,limits,problem,origin,positions,8192,2147483648,storage,channel,row,column,valid,supported,region,point,
    prepare(limits,problem,origin,positions,8192,1073741824,2147483648,configuration(limits),valid,supported,region))

# The full dense image is the region with packing origin zero.
def full_readback(~custom: F32 -> F32,+activation: Activations.Activation,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Tree.read(F32,Model.run(~custom,activation,Inference.work(problem),0,Inputs.input_tree(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)),
    U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(Padded.spatial(problem))),Coordinates.position(problem,row,column)))) == ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}:
  ConvSpec.Problem{shape,+input_values,+weight_values,+bias_values} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input_values,weight_values,bias_values} : ConvSpec.Problem}
  +spatial = Padded.spatial(problem)
  +spatial_exact = Padded.spatial_exact(limits,problem,Guards.valid(problem,valid),Support.fast(limits,problem,supported))
  +region = {Region{
    %Equal.sym(Nat,U32.to_nat(spatial),PackingContents.grid(problem),spatial_exact) : {Nat.is_le(_,PackingContents.grid(problem)) == True{} : Bool} Order.reflexive(PackingContents.grid(problem)),
    %Equal.sym(Nat,U32.to_nat(spatial),PackingContents.grid(problem),spatial_exact) : {Nat.is_lt(0n,_) == True{} : Bool} Inference.spatial_positive(problem)} : Region<problem,0,spatial>}
  +point = point(limits,problem,0,spatial,channel,row,column,valid,supported,region,channel_inside,row_inside,column_inside,Order.zero_le(Coordinates.position(problem,row,column)),
    %Equal.sym(Nat,Blocked.local_position(problem,0,row,column),Coordinates.position(problem,row,column),Order.subtract_zero(Coordinates.position(problem,row,column))) : {Nat.is_lt(_,U32.to_nat(spatial)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(spatial),PackingContents.grid(problem),spatial_exact) : {Nat.is_lt(Coordinates.position(problem,row,column),_) == True{} : Bool}
    RowMajor.radix(row,ConvSpec.out_height(problem),column,ConvSpec.out_width(problem),row_inside,column_inside))
  %Order.subtract_zero(Coordinates.position(problem,row,column)) :
    {Tree.read(F32,Model.run(~custom,activation,Inference.work(problem),0,Inputs.input_tree(problem,storage),GemmDot.reordered(problem,storage),Inputs.bias_tree(problem,storage)),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(spatial)),_))) == ActivationProofs.activation_value(~custom,activation,Values.matrix_value(problem,channel,row,column)) : F32}
  run_readback(~custom,activation,limits,problem,0,spatial,storage,channel,row,column,valid,supported,region,point)
