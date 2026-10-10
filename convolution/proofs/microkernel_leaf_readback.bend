# One packed leaf of the four-row-unit microkernel reads back the frozen matrix
# value at every channel/column; this replaces the paired-kernel leaf theorem.
import Base
import ./gemm_output_relocation.bend as Relocation
import ../../lib/tensor.bend as Tensors
import ../../lib/storage_buffer.bend as Storage
import ../../lib/tensor_f32.bend as F
import ../../lib/kernel_tile.bend as Tile
import ../../lib/kernel_rows.bend as Rows
import ../../lib/math_sequence.bend as Sequence
import ../../lib/traversal_reduction.bend as Reduction
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/kernel_gemm_refinement.bend as GemmLaws
import ../../lib/proofs/kernel_accumulator.bend as Accumulator
import ../../lib/proofs/kernel_reduction_model.bend as ReductionModel
import ../../lib/proofs/kernel_coordinate_refinement.bend as CoordinateLaws
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ../specification.bend as ConvSpec
import ../matrix_specification.bend as MatrixSpec
import ../inference.bend as NativeConv
import ./input_storage.bend as Inputs
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./reduction_coordinate_bounds.bend as Coordinates
import ./prepared_packing_samples.bend as Prepared
import ./convolution_leaf_contract.bend as LeafContract
import ./packing_tile_bounds.bend as PackingBounds
import ./packing_readback.bend as PackingReadback
import ./packed_output_bounds.bend as Output
import ./packed_output_allocation.bend as PackedOutputAllocation
import ./output_values.bend as Values
import ./tile_partition_bounds.bend as TilePartitionBounds
import ./packed_leaf_bounds.bend as LeafBounds
import ./packing_storage_refinement.bend as PackingStorage
import ./tile_layout.bend as TileLayout
import ./gather_storage_model.bend as GatherModel
import ../../lib/traversal_partition.bend as Partition
import ./weight_reorder_bounds.bend as WeightBounds
import ./gemm_output_readback.bend as GemmReadback
import ./gemm_dot_matches_matrix.bend as GemmDot

def rows(+problem: ConvSpec.Problem) -> U32:
  U32.from_nat(Guards.output_channel_count(problem))

def allocated(+rows: U32,+span: U32) -> Tree.Tree<F32>:
  Tree.fresh(F32,Tensors.depth_for((rows * span : U32)),0.0)

# The kernel stores accumulator plus bias; activation is a separate pass.
def output(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,tiles: Tree.Tree<Tile.Tile8>,+span: U32) -> Tree.Tree<F32>:
  GemmLaws.run(rows(problem),tiles,GemmDot.reordered(problem,storage_contract),Inputs.bias_tree(problem,storage_contract),
    allocated(rows(problem),span),span,Coordinates.native_inner(problem))

def row_number_of(+row: Nat,small: {Nat.is_lt(row,4n) == True{} : Bool}) -> {CoordinateLaws.row_number(GemmDot.row_of(row)) == row : Nat}:
  match row:
    case 0n: {==}
    case 1n+a:
      match a:
        case 0n: {==}
        case 1n+b:
          match b:
            case 0n: {==}
            case 1n+c:
              match c:
                case 0n: {==}
                case 1n+d:
                  Empty.absurd({CoordinateLaws.row_number(GemmDot.row_of(4n+d)) == 4n+d : Nat},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(d,0n),True{},Equal.sym(Bool,Nat.is_lt(d,0n),False{},Division.lt_zero(d)),small)))

def lane_value_of(+lane: Nat,small: {Nat.is_lt(lane,8n) == True{} : Bool}) -> {Sequence.index_value(8n,GemmDot.lane_of(lane)) == lane : Nat}:
  match lane:
    case 0n: {==}
    case 1n+a:
      match a:
        case 0n: {==}
        case 1n+b:
          match b:
            case 0n: {==}
            case 1n+c:
              match c:
                case 0n: {==}
                case 1n+d:
                  match d:
                    case 0n: {==}
                    case 1n+e:
                      match e:
                        case 0n: {==}
                        case 1n+f:
                          match f:
                            case 0n: {==}
                            case 1n+g:
                              match g:
                                case 0n: {==}
                                case 1n+h:
                                  Empty.absurd({Sequence.index_value(8n,GemmDot.lane_of(8n+h)) == 8n+h : Nat},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(h,0n),True{},Equal.sym(Bool,Nat.is_lt(h,0n),False{},Division.lt_zero(h)),small)))

def lane_position(+offset: U32,+group: Nat,+lane: Nat,+position: Nat,
  +lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +exact: {Nat.add(Nat.add(U32.to_nat(offset),Nat.mul(group,8n)),lane) == position : Nat}) ->
  {PackingReadback.logical_position(offset,group,GemmDot.lane_of(lane)) == position : Nat}:
  %Equal.sym(Nat,Sequence.index_value(8n,GemmDot.lane_of(lane)),lane,lane_value_of(lane,lane_small)) :
    {Nat.add(Nat.add(U32.to_nat(offset),Nat.mul(group,8n)),_) == position : Nat}
  exact

# The unit parity selects the lower or upper four rows of the weight block.
def parity_dot(~Evidence: LeafContract.Evidence(),~samples: LeafContract.Sampling(~Evidence),+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +tiles: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+base: U32,+group: Nat,+block: Nat,+row: Accumulator.Row,+lane: Nat,
  +channel: Nat,+output_row: Nat,+output_column: Nat,+upper: Bool,+parity: Nat,
  +upper_exact: {upper == Nat.is_eq(parity,1n) : Bool},+parity_small: {Nat.is_lt(parity,2n) == True{} : Bool},
  +certificate: Evidence(problem,offset,span,tiles,output_row,output_column),
  +group_inside: {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool},
  +lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +position: {Nat.add(Nat.add(U32.to_nat(offset),Nat.mul(group,8n)),lane) == Coordinates.position(problem,output_row,output_column) : Nat},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool},
  +base_exact: {U32.to_nat(base) == Nat.mul(block,Coordinates.inner(problem)) : Nat},
  +channel_exact: {channel == Nat.add(Nat.mul(block,8n),Nat.add(Nat.mul(parity,4n),CoordinateLaws.row_number(row))) : Nat},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {GemmDot.selected_dot(upper,ReductionModel.RowLane{row,GemmDot.lane_of(lane)},Coordinates.inner(problem),
      ReductionModel.initial(tiles,GemmDot.reordered(problem,storage_contract),U32.from_nat(Nat.mul(group,Coordinates.inner(problem))),U32.from_nat(0n),base)) ==
    Values.matrix_dot(MatrixSpec.kchannels(Guards.input_channel_count(problem),0n,Guards.kernel(problem)),problem,channel,output_row,output_column,0.0) : F32}:
  match parity:
    case 0n:
      %Equal.sym(Bool,upper,False{},upper_exact) :
        {GemmDot.selected_dot(_,ReductionModel.RowLane{row,GemmDot.lane_of(lane)},Coordinates.inner(problem),
          ReductionModel.initial(tiles,GemmDot.reordered(problem,storage_contract),U32.from_nat(Nat.mul(group,Coordinates.inner(problem))),U32.from_nat(0n),base)) ==
          Values.matrix_dot(MatrixSpec.kchannels(Guards.input_channel_count(problem),0n,Guards.kernel(problem)),problem,channel,output_row,output_column,0.0) : F32}
      GemmDot.job_dot(~Evidence,~samples,~False{},limits,problem,storage_contract,valid,supported,tiles,offset,span,base,group,block,row,GemmDot.lane_of(lane),channel,output_row,output_column,
        certificate,group_inside,lane_position(offset,group,lane,Coordinates.position(problem,output_row,output_column),lane_small,position),row_inside,column_inside,base_exact,
        %Equal.sym(Nat,Sequence.index_value(8n,CoordinateLaws.weight_lane(False{},row)),Nat.add(Bool.pick(Nat,False{},4n,0n),CoordinateLaws.row_number(row)),
          CoordinateLaws.weight_coordinate(False{},row)) : {channel == Nat.add(Nat.mul(block,8n),_) : Nat}
        channel_exact,channel_inside)
    case 1n+rest:
      match rest:
        case 0n:
          %Equal.sym(Bool,upper,True{},upper_exact) :
            {GemmDot.selected_dot(_,ReductionModel.RowLane{row,GemmDot.lane_of(lane)},Coordinates.inner(problem),
              ReductionModel.initial(tiles,GemmDot.reordered(problem,storage_contract),U32.from_nat(Nat.mul(group,Coordinates.inner(problem))),U32.from_nat(0n),base)) ==
              Values.matrix_dot(MatrixSpec.kchannels(Guards.input_channel_count(problem),0n,Guards.kernel(problem)),problem,channel,output_row,output_column,0.0) : F32}
          GemmDot.job_dot(~Evidence,~samples,~True{},limits,problem,storage_contract,valid,supported,tiles,offset,span,base,group,block,row,GemmDot.lane_of(lane),channel,output_row,output_column,
            certificate,group_inside,lane_position(offset,group,lane,Coordinates.position(problem,output_row,output_column),lane_small,position),row_inside,column_inside,base_exact,
            %Equal.sym(Nat,Sequence.index_value(8n,CoordinateLaws.weight_lane(True{},row)),Nat.add(Bool.pick(Nat,True{},4n,0n),CoordinateLaws.row_number(row)),
              CoordinateLaws.weight_coordinate(True{},row)) : {channel == Nat.add(Nat.mul(block,8n),_) : Nat}
            channel_exact,channel_inside)
        case 1n+more:
          Empty.absurd({GemmDot.selected_dot(upper,ReductionModel.RowLane{row,GemmDot.lane_of(lane)},Coordinates.inner(problem),
              ReductionModel.initial(tiles,GemmDot.reordered(problem,storage_contract),U32.from_nat(Nat.mul(group,Coordinates.inner(problem))),U32.from_nat(0n),base)) ==
            Values.matrix_dot(MatrixSpec.kchannels(Guards.input_channel_count(problem),0n,Guards.kernel(problem)),problem,channel,output_row,output_column,0.0) : F32},
            WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(more,0n),True{},Equal.sym(Bool,Nat.is_lt(more,0n),False{},Division.lt_zero(more)),parity_small)))

def leaf_query(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+tiles: Tree.Tree<Tile.Tile8>,+span: U32,
  +blocks: Nat,+channel: Nat,+column: Nat,+span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},+fast: Support.FastSupported<limits,problem>,
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool}) ->
  GemmReadback.Query:
  +channels = Guards.output_channel_count(problem)
  +rows_exact = SupportedCoordinateBounds.co_exact(limits,problem,Padded.scalar(limits,problem,fast))
  +half = Machine.half_word(limits)
  GemmReadback.Query{rows(problem),span,Coordinates.native_inner(problem),blocks,half,
    tiles,GemmDot.reordered(problem,storage_contract),Inputs.bias_tree(problem,storage_contract),channel,column,
    Limits.half_word_fits(limits),span_exact,
    {%Equal.sym(Nat,U32.to_nat(rows(problem)),channels,rows_exact) : {Nat.is_le(Nat.add(_,3n),U32.to_nat(half)) == True{} : Bool}
      Order.transitive(Nat.add(channels,3n),Nat.add(channels,7n),U32.to_nat(half),GemmReadback.add_small(channels,3n,7n,{==}),
        WeightBounds.padded_rows_bound(limits,problem,fast)) : {Nat.is_le(Nat.add(U32.to_nat(rows(problem)),3n),U32.to_nat(half)) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(rows(problem)),channels,rows_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(span)),U32.to_nat(half)) == True{} : Bool}
      Order.transitive(Nat.mul(channels,U32.to_nat(span)),U32.to_nat(Machine.capacity(limits)),U32.to_nat(half),
        Output.elements_bound(limits,problem,span,fast,span_bound),Bounds.as_nat_le(Machine.capacity(limits),half,Limits.capacity_within_half(limits))) :
        {Nat.is_le(Nat.mul(U32.to_nat(rows(problem)),U32.to_nat(span)),U32.to_nat(half)) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat(rows(problem)),channels,rows_exact) : {Nat.is_lt(channel,_) == True{} : Bool}
      channel_inside : {Nat.is_lt(channel,U32.to_nat(rows(problem))) == True{} : Bool}},
    column_inside}

# channel = unit*4 + row and unit = block*2 + parity: row block*8 + parity*4 + row.
def eight_split(+channel: Nat,+unit: Nat,+row: Nat,+block: Nat,+parity: Nat,
  +channel_split: {channel == Nat.add(Nat.mul(unit,4n),row) : Nat},+unit_split: {unit == Nat.add(Nat.mul(block,2n),parity) : Nat}) ->
  {channel == Nat.add(Nat.mul(block,8n),Nat.add(Nat.mul(parity,4n),row)) : Nat}:
  %Addition.associative(Nat.mul(block,8n),Nat.mul(parity,4n),row) : {channel == _ : Nat}
  %RowMajor.assoc(block,2n,4n) : {channel == Nat.add(Nat.add(_,Nat.mul(parity,4n)),row) : Nat}
  %RowMajor.distrib(Nat.mul(block,2n),parity,4n) : {channel == Nat.add(_,row) : Nat}
  %unit_split : {channel == Nat.add(Nat.mul(_,4n),row) : Nat}
  channel_split

def groups_exact(+span: U32,+blocks: Nat,+span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat}) -> {PackingBounds.groups(span) == blocks : Nat}:
  %Equal.sym(Nat,U32.to_nat(span),Nat.mul(blocks,8n),span_exact) : {Nat.div(_,8n) == blocks : Nat}
  %Addition.zero_right(Nat.mul(blocks,8n)) : {Nat.div(_,8n) == blocks : Nat}
  Radix.quotient(blocks,0n,8n,{==},{==})

def input_base_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+fast: Support.FastSupported<limits,problem>,
  +span: U32,+group_word: U32,+group: Nat,
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +group_exact: {U32.to_nat(group_word) == group : Nat},
  +group_inside: {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool}) ->
  {(group_word * Coordinates.native_inner(problem) : U32) == U32.from_nat(Nat.mul(group,Coordinates.inner(problem))) : U32}:
  +inner = Coordinates.inner(problem)
  +native = Coordinates.native_inner(problem)
  +cap = Machine.capacity(limits)
  +inner_exact = Coordinates.inner_exact(limits,problem,valid,Padded.scalar(limits,problem,fast))
  GemmDot.word_from_exact((group_word * native : U32),Nat.mul(group,inner),
    %group_exact : {U32.to_nat((group_word * native : U32)) == Nat.mul(_,inner) : Nat}
    %inner_exact : {U32.to_nat((group_word * native : U32)) == Nat.mul(U32.to_nat(group_word),_) : Nat}
    Arithmetic.multiply(group_word,native,cap,Limits.capacity_fits(limits),
      %Equal.sym(Nat,U32.to_nat(group_word),group,group_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(native)),U32.to_nat(cap)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(native),inner,inner_exact) : {Nat.is_le(Nat.mul(group,_),U32.to_nat(cap)) == True{} : Bool}
      Order.transitive(Nat.mul(group,inner),Nat.mul(PackingBounds.groups(span),inner),U32.to_nat(cap),
        RowMajor.mul_le(group,PackingBounds.groups(span),inner,Bounds.strict_le(group,PackingBounds.groups(span),group_inside)),
        PackingBounds.elements_bound(limits,problem,span,fast,span_bound))))

def weight_base_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+fast: Support.FastSupported<limits,problem>,
  +unit_word: U32,+unit: Nat,+block: Nat,+unit_exact: {U32.to_nat(unit_word) == unit : Nat},+block_exact: {block == Nat.div(unit,2n) : Nat},
  +block_inside: {Nat.is_lt(Nat.mul(block,8n),Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {U32.to_nat((unit_word / 2 * Coordinates.native_inner(problem) : U32)) == Nat.mul(block,Coordinates.inner(problem)) : Nat}:
  +inner = Coordinates.inner(problem)
  +native = Coordinates.native_inner(problem)
  +cap = Machine.capacity(limits)
  +inner_exact = Coordinates.inner_exact(limits,problem,valid,Padded.scalar(limits,problem,fast))
  +half_exact = {%Equal.sym(Nat,block,Nat.div(unit,2n),block_exact) : {U32.to_nat((unit_word / 2 : U32)) == _ : Nat}
    %unit_exact : {U32.to_nat((unit_word / 2 : U32)) == Nat.div(_,2n) : Nat}
    U32Division.divide(unit_word,2,2147483648,{==},{==},{==}) : {U32.to_nat((unit_word / 2 : U32)) == block : Nat}}
  %half_exact : {U32.to_nat((unit_word / 2 * native : U32)) == Nat.mul(_,inner) : Nat}
  %inner_exact : {U32.to_nat((unit_word / 2 * native : U32)) == Nat.mul(U32.to_nat((unit_word / 2 : U32)),_) : Nat}
  Arithmetic.multiply((unit_word / 2 : U32),native,cap,Limits.capacity_fits(limits),
    %Equal.sym(Nat,U32.to_nat((unit_word / 2 : U32)),block,half_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(native)),U32.to_nat(cap)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(native),inner,inner_exact) : {Nat.is_le(Nat.mul(block,_),U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(Nat.mul(block,inner),Nat.mul(Guards.output_channel_count(problem),inner),U32.to_nat(cap),
      RowMajor.mul_le(block,Guards.output_channel_count(problem),inner,Bounds.strict_le(block,Guards.output_channel_count(problem),
        Order.le_lt_trans(block,Nat.mul(block,8n),Guards.output_channel_count(problem),RowMajor.product_covers(block,8n,{==}),block_inside))),
      GemmDot.weights_capacity(limits,problem,valid,fast)))

def parity_exact(+unit_word: U32,+unit: Nat,+unit_exact: {U32.to_nat(unit_word) == unit : Nat}) ->
  {U32.is_eq((unit_word % 2 : U32),1) == Nat.is_eq(Nat.mod(unit,2n),1n) : Bool}:
  Equal.trans(Bool,U32.is_eq((unit_word % 2 : U32),1),Nat.is_eq(U32.to_nat((unit_word % 2 : U32)),1n),Nat.is_eq(Nat.mod(unit,2n),1n),
    Comparison.eq((unit_word % 2 : U32),1),
    Equal.cong(Nat,Bool,value => Nat.is_eq(value,1n),U32.to_nat((unit_word % 2 : U32)),Nat.mod(unit,2n),
      Equal.trans(Nat,U32.to_nat((unit_word % 2 : U32)),Nat.mod(U32.to_nat(unit_word),2n),Nat.mod(unit,2n),
        U32Division.modulo(unit_word,2,2147483648,{==},{==},{==}),
        Equal.cong(Nat,Nat,value => Nat.mod(value,2n),U32.to_nat(unit_word),unit,unit_exact))))

def normalized_dot(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+tiles: Tree.Tree<Tile.Tile8>,+upper: Bool,+key: ReductionModel.RowLane,
  +input_word: U32,+group: Nat,+base: U32,
  +inner_exact: {U32.to_nat(Coordinates.native_inner(problem)) == Coordinates.inner(problem) : Nat},
  +input_exact: {input_word == U32.from_nat(Nat.mul(group,Coordinates.inner(problem))) : U32}) ->
  {GemmDot.selected_dot(upper,key,U32.to_nat(Coordinates.native_inner(problem)),ReductionModel.initial(tiles,GemmDot.reordered(problem,storage_contract),input_word,0,base)) ==
    GemmDot.selected_dot(upper,key,Coordinates.inner(problem),
      ReductionModel.initial(tiles,GemmDot.reordered(problem,storage_contract),U32.from_nat(Nat.mul(group,Coordinates.inner(problem))),U32.from_nat(0n),base)) : F32}:
  +target = GemmDot.selected_dot(upper,key,Coordinates.inner(problem),
    ReductionModel.initial(tiles,GemmDot.reordered(problem,storage_contract),U32.from_nat(Nat.mul(group,Coordinates.inner(problem))),U32.from_nat(0n),base))
  %Equal.sym(Nat,U32.to_nat(Coordinates.native_inner(problem)),Coordinates.inner(problem),inner_exact) : {GemmDot.selected_dot(upper,key,_,ReductionModel.initial(tiles,GemmDot.reordered(problem,storage_contract),input_word,0,base)) == target : F32}
  %Equal.sym(U32,input_word,U32.from_nat(Nat.mul(group,Coordinates.inner(problem))),input_exact) : {GemmDot.selected_dot(upper,key,Coordinates.inner(problem),ReductionModel.initial(tiles,GemmDot.reordered(problem,storage_contract),_,0,base)) == target : F32}
  {==}

def channel_row_exact(+channel: Nat,+block: Nat,+parity: Nat,+row: Nat,+row_small: {Nat.is_lt(row,4n) == True{} : Bool},
  +split: {channel == Nat.add(Nat.mul(block,8n),Nat.add(Nat.mul(parity,4n),row)) : Nat}) ->
  {channel == Nat.add(Nat.mul(block,8n),Nat.add(Nat.mul(parity,4n),CoordinateLaws.row_number(GemmDot.row_of(row)))) : Nat}:
  %Equal.sym(Nat,CoordinateLaws.row_number(GemmDot.row_of(row)),row,row_number_of(row,row_small)) : {channel == Nat.add(Nat.mul(block,8n),Nat.add(Nat.mul(parity,4n),_)) : Nat}
  split

def eight_position(+offset: U32,+column: Nat,+group: Nat,+lane: Nat,+position: Nat,
  +column_eight: {column == Nat.add(Nat.mul(group,8n),lane) : Nat},+exact: {Nat.add(U32.to_nat(offset),column) == position : Nat}) ->
  {Nat.add(Nat.add(U32.to_nat(offset),Nat.mul(group,8n)),lane) == position : Nat}:
  %Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(offset),Nat.mul(group,8n)),lane),Nat.add(U32.to_nat(offset),Nat.add(Nat.mul(group,8n),lane)),
    Addition.associative(U32.to_nat(offset),Nat.mul(group,8n),lane)) : {_ == position : Nat}
  %column_eight : {Nat.add(U32.to_nat(offset),_) == position : Nat}
  exact

def group_inside(+span: U32,+blocks: Nat,+column: Nat,+group: Nat,+lane: Nat,
  +span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},+column_eight: {column == Nat.add(Nat.mul(group,8n),lane) : Nat},
  +column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool}) -> {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool}:
  %Equal.sym(Nat,PackingBounds.groups(span),blocks,groups_exact(span,blocks,span_exact)) : {Nat.is_lt(group,_) == True{} : Bool}
  RowMajor.row_inside(group,blocks,8n,lane,
    %column_eight : {Nat.is_lt(_,Nat.mul(blocks,8n)) == True{} : Bool}
    %span_exact : {Nat.is_lt(column,_) == True{} : Bool}
    column_inside)

# The selected accumulator of the actual job is the frozen ordered matrix dot.
def selection_matches(~Evidence: LeafContract.Evidence(),~samples: LeafContract.Sampling(~Evidence),+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +tiles: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+blocks: Nat,+channel: Nat,+column: Nat,+output_row: Nat,+output_column: Nat,
  +certificate: Evidence(problem,offset,span,tiles,output_row,output_column),
  +span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},+span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},
  +position: {Nat.add(U32.to_nat(offset),column) == Coordinates.position(problem,output_row,output_column) : Nat},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},+column_in_width: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {GemmReadback.select_lane(Nat.mod(column,8n),GemmReadback.select_row(Nat.mod(channel,4n),
      GemmReadback.unit_accumulators(leaf_query(limits,problem,storage_contract,tiles,span,blocks,channel,column,span_exact,Support.fast(limits,problem,supported),span_bound,channel_inside,column_inside),
        Nat.div(channel,4n),Nat.div(column,8n)))) ==
    Values.matrix_dot(MatrixSpec.kchannels(Guards.input_channel_count(problem),0n,Guards.kernel(problem)),problem,channel,output_row,output_column,0.0) : F32}:
  +fast = Support.fast(limits,problem,supported)
  +facts = Guards.valid(problem,valid)
  +query = leaf_query(limits,problem,storage_contract,tiles,span,blocks,channel,column,span_exact,fast,span_bound,channel_inside,column_inside)
  +half = Machine.half_word(limits)
  +fits = Limits.half_word_fits(limits)
  +native = Coordinates.native_inner(problem)
  +inner = Coordinates.inner(problem)
  +inner_exact = Coordinates.inner_exact(limits,problem,facts,Padded.scalar(limits,problem,fast))
  +channel_small = GemmDot.half_room(limits,problem,fast,channel,channel_inside)
  +column_small = Order.transitive(column,U32.to_nat(span),U32.to_nat(half),Bounds.strict_le(column,U32.to_nat(span),column_inside),GemmReadback.columns_within(query))
  +unit = Nat.div(channel,4n)
  +row = Nat.mod(channel,4n)
  +group = Nat.div(column,8n)
  +lane = Nat.mod(column,8n)
  +channel_split = GemmReadback.decomposed(channel,4n,half,fits,channel_small,{==})
  +column_split = GemmReadback.decomposed(column,8n,half,fits,column_small,{==})
  +row_small = GemmReadback.remainder_small(channel,4n,half,fits,channel_small,{==})
  +lane_small = GemmReadback.remainder_small(column,8n,half,fits,column_small,{==})
  +unit_rows = GemmReadback.split_unit_inside(query,unit,row,channel_split)
  +unit_inside = GemmReadback.unit_counted(query,unit,unit_rows)
  +query_group_inside = GemmReadback.split_group_inside(query,group,lane,column_split)
  +unit_word = GemmReadback.unit_word(query,unit,group)
  +group_word = GemmReadback.group_word(query,unit,group)
  +unit_exact = GemmReadback.unit_exact(query,unit,group,unit_inside,query_group_inside)
  +group_exact = GemmReadback.group_exact(query,unit,group,unit_inside,query_group_inside)
  +unit_small = Order.transitive(unit,channel,U32.to_nat(half),
    Order.transitive(unit,Nat.mul(unit,4n),channel,RowMajor.product_covers(unit,4n,{==}),
      %Equal.sym(Nat,channel,Nat.add(Nat.mul(unit,4n),row),channel_split) : {Nat.is_le(Nat.mul(unit,4n),_) == True{} : Bool}
      Order.le_add(Nat.mul(unit,4n),row)),channel_small)
  +parity = Nat.mod(unit,2n)
  +block = Nat.div(unit,2n)
  +unit_split = GemmReadback.decomposed(unit,2n,half,fits,unit_small,{==})
  +parity_small = GemmReadback.remainder_small(unit,2n,half,fits,unit_small,{==})
  +channel_eight = eight_split(channel,unit,row,block,parity,channel_split,unit_split)
  +block_inside = Order.le_lt_trans(Nat.mul(block,8n),channel,Guards.output_channel_count(problem),
    %Equal.sym(Nat,channel,Nat.add(Nat.mul(block,8n),Nat.add(Nat.mul(parity,4n),row)),channel_eight) : {Nat.is_le(Nat.mul(block,8n),_) == True{} : Bool}
    Order.le_add(Nat.mul(block,8n),Nat.add(Nat.mul(parity,4n),row)),channel_inside)
  +packing_group = group_inside(span,blocks,column,group,lane,span_exact,column_split,column_inside)
  +base = (unit_word / 2 * native : U32)
  +base_exact = weight_base_exact(limits,problem,facts,fast,unit_word,unit,block,unit_exact,{==},block_inside)
  +input_word = (group_word * native : U32)
  +input_exact = input_base_exact(limits,problem,facts,fast,span,group_word,group,span_bound,group_exact,packing_group)
  +upper = U32.is_eq((unit_word % 2 : U32),1)
  +key = {ReductionModel.RowLane{GemmDot.row_of(row),GemmDot.lane_of(lane)} : ReductionModel.RowLane}
  +accumulators = GemmReadback.unit_accumulators(query,unit,group)
  +state = ReductionModel.initial(tiles,GemmDot.reordered(problem,storage_contract),input_word,0,base)
  Equal.trans(F32,GemmReadback.select_lane(lane,GemmReadback.select_row(row,accumulators)),ReductionModel.accumulator_at(key,accumulators),
    Values.matrix_dot(MatrixSpec.kchannels(Guards.input_channel_count(problem),0n,Guards.kernel(problem)),problem,channel,output_row,output_column,0.0),
    GemmDot.selected_key(row,lane,accumulators),
    Equal.trans(F32,ReductionModel.accumulator_at(key,accumulators),GemmDot.selected_dot(upper,key,U32.to_nat(native),state),
      Values.matrix_dot(MatrixSpec.kchannels(Guards.input_channel_count(problem),0n,Guards.kernel(problem)),problem,channel,output_row,output_column,0.0),
      GemmDot.accumulator_dot(upper,key,U32.to_nat(native),state),
      Equal.trans(F32,GemmDot.selected_dot(upper,key,U32.to_nat(native),state),
        GemmDot.selected_dot(upper,key,inner,ReductionModel.initial(tiles,GemmDot.reordered(problem,storage_contract),U32.from_nat(Nat.mul(group,inner)),U32.from_nat(0n),base)),
        Values.matrix_dot(MatrixSpec.kchannels(Guards.input_channel_count(problem),0n,Guards.kernel(problem)),problem,channel,output_row,output_column,0.0),
        normalized_dot(problem,storage_contract,tiles,upper,key,input_word,group,base,inner_exact,input_exact),
        parity_dot(~Evidence,~samples,limits,problem,storage_contract,valid,supported,tiles,offset,span,base,group,block,GemmDot.row_of(row),lane,channel,output_row,output_column,
          upper,parity,parity_exact(unit_word,unit,unit_exact),parity_small,certificate,packing_group,lane_small,
          eight_position(offset,column,group,lane,Coordinates.position(problem,output_row,output_column),column_split,position),
          row_inside,column_in_width,base_exact,channel_row_exact(channel,block,parity,row,row_small,channel_eight),channel_inside))))

def wanted_matches(~Evidence: LeafContract.Evidence(),~samples: LeafContract.Sampling(~Evidence),+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +tiles: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+blocks: Nat,+channel: Nat,+column: Nat,+output_row: Nat,+output_column: Nat,
  +certificate: Evidence(problem,offset,span,tiles,output_row,output_column),
  +span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},+span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},
  +position: {Nat.add(U32.to_nat(offset),column) == Coordinates.position(problem,output_row,output_column) : Nat},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},+column_in_width: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {GemmReadback.wanted(leaf_query(limits,problem,storage_contract,tiles,span,blocks,channel,column,span_exact,Support.fast(limits,problem,supported),span_bound,channel_inside,column_inside)) ==
    Values.matrix_value(problem,channel,output_row,output_column) : F32}:
  +query = leaf_query(limits,problem,storage_contract,tiles,span,blocks,channel,column,span_exact,Support.fast(limits,problem,supported),span_bound,channel_inside,column_inside)
  +selection = GemmReadback.select_lane(Nat.mod(column,8n),GemmReadback.select_row(Nat.mod(channel,4n),GemmReadback.unit_accumulators(query,Nat.div(channel,4n),Nat.div(column,8n))))
  +matrix = Values.matrix_dot(MatrixSpec.kchannels(Guards.input_channel_count(problem),0n,Guards.kernel(problem)),problem,channel,output_row,output_column,0.0)
  +bias_read = Tree.read(F32,Inputs.bias_tree(problem,storage_contract),U32.from_nat(channel))
  %Equal.sym(F32,selection,matrix,selection_matches(~Evidence,~samples,limits,problem,storage_contract,valid,supported,tiles,offset,span,blocks,channel,column,output_row,output_column,
    certificate,span_exact,span_bound,channel_inside,column_inside,position,row_inside,column_in_width)) :
    {(_ + bias_read : F32) == Values.matrix_value(problem,channel,output_row,output_column) : F32}
  %Equal.sym(F32,bias_read,ConvSpec.bias_at(problem,channel),Values.bias_matches_spec(problem,storage_contract,channel,valid,channel_inside)) :
    {(matrix + _ : F32) == Values.matrix_value(problem,channel,output_row,output_column) : F32}
  {==}

def relocated_readback(~Evidence: LeafContract.Evidence(),~samples: LeafContract.Sampling(~Evidence),+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +tiles: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+blocks: Nat,+channel: Nat,+column: Nat,+output_row: Nat,+output_column: Nat,
  +certificate: Evidence(problem,offset,span,tiles,output_row,output_column),
  +span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},+span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},
  +position: {Nat.add(U32.to_nat(offset),column) == Coordinates.position(problem,output_row,output_column) : Nat},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},+column_in_width: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool},
  +output_start: U32,+columns: U32,+buffer: Tree.Tree<F32>,+depth: Nat,+balanced: Shape.Balanced(F32,depth,buffer),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(U32.to_nat(rows(problem)),U32.to_nat(columns)),U32.to_nat(Tree.capacity(F32,buffer))) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(output_start),U32.to_nat(span)),Nat.add(U32.to_nat(columns),7n)) == True{} : Bool},
  +columns_room: {Nat.is_le(U32.to_nat(columns),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +in_output: {Nat.is_lt(Nat.add(U32.to_nat(output_start),column),U32.to_nat(columns)) == True{} : Bool}) ->
  {Tree.read(F32,GemmLaws.run_at(rows(problem),tiles,GemmDot.reordered(problem,storage_contract),Inputs.bias_tree(problem,storage_contract),buffer,span,Coordinates.native_inner(problem),output_start,columns),
      U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(columns)),Nat.add(U32.to_nat(output_start),column)))) == Values.matrix_value(problem,channel,output_row,output_column) : F32}:
  +query = leaf_query(limits,problem,storage_contract,tiles,span,blocks,channel,column,span_exact,Support.fast(limits,problem,supported),span_bound,channel_inside,column_inside)
  Equal.trans(F32,Tree.read(F32,GemmLaws.run_at(rows(problem),tiles,GemmDot.reordered(problem,storage_contract),Inputs.bias_tree(problem,storage_contract),buffer,span,Coordinates.native_inner(problem),output_start,columns),
      U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(columns)),Nat.add(U32.to_nat(output_start),column)))),GemmReadback.wanted(query),Values.matrix_value(problem,channel,output_row,output_column),
    Relocation.run_readback(query,output_start,columns,buffer,depth,balanced,depth_ok,output_room,end_room,columns_room,in_output),
    wanted_matches(~Evidence,~samples,limits,problem,storage_contract,valid,supported,tiles,offset,span,blocks,channel,column,output_row,output_column,
      certificate,span_exact,span_bound,channel_inside,column_inside,position,row_inside,column_in_width))

def allocation_readback(~Evidence: LeafContract.Evidence(),~samples: LeafContract.Sampling(~Evidence),+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +tiles: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+blocks: Nat,+channel: Nat,+column: Nat,+output_row: Nat,+output_column: Nat,
  +certificate: Evidence(problem,offset,span,tiles,output_row,output_column),
  +span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},+span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},
  +position: {Nat.add(U32.to_nat(offset),column) == Coordinates.position(problem,output_row,output_column) : Nat},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},+column_in_width: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool},
  +ready: PackedOutputAllocation.Sized<PackedOutputAllocation.count(problem,span),PackedOutputAllocation.depth(problem,span)>) ->
  {Tree.read(F32,output(problem,storage_contract,tiles,span),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(span)),column))) ==
    Values.matrix_value(problem,channel,output_row,output_column) : F32}:
  PackedOutputAllocation.Sized{+depth_bound,+room} = ready
  +fast = Support.fast(limits,problem,supported)
  +facts = Guards.valid(problem,valid)
  +query = leaf_query(limits,problem,storage_contract,tiles,span,blocks,channel,column,span_exact,fast,span_bound,channel_inside,column_inside)
  +count = PackedOutputAllocation.count(problem,span)
  +channels = Guards.output_channel_count(problem)
  +exact = Output.count_exact(limits,problem,span,fast,span_bound)
  +rows_exact = SupportedCoordinateBounds.co_exact(limits,problem,Padded.scalar(limits,problem,fast))
  +span_positive = Order.le_lt_trans(0n,column,U32.to_nat(span),Order.zero_le(column),column_inside)
  +positive = {%Equal.sym(Nat,U32.to_nat(count),Nat.mul(channels,U32.to_nat(span)),exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Support.positive_mul(channels,U32.to_nat(span),PackedOutputAllocation.channel_positive(problem,facts),span_positive) : {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}}
  +depth_same = Depth.positive_depth(count,positive)
  +depth_ok = {%Equal.sym(Nat,Tensors.depth_for(count),Storage.allocation_depth(count),depth_same) : {Nat.is_le(_,30n) == True{} : Bool}
    depth_bound : {Nat.is_le(Tensors.depth_for(count),30n) == True{} : Bool}}
  +output_room = {%Equal.sym(Nat,Tensors.depth_for(count),Storage.allocation_depth(count),depth_same) :
      {Nat.is_le(Nat.mul(U32.to_nat(rows(problem)),U32.to_nat(span)),U32.to_nat(Tree.capacity(F32,Tree.fresh(F32,_,0.0)))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(rows(problem)),channels,rows_exact) :
      {Nat.is_le(Nat.mul(_,U32.to_nat(span)),U32.to_nat(Tree.capacity(F32,Tree.fresh(F32,Storage.allocation_depth(count),0.0)))) == True{} : Bool}
    %exact : {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,Tree.fresh(F32,Storage.allocation_depth(count),0.0)))) == True{} : Bool}
    room : {Nat.is_le(Nat.mul(U32.to_nat(rows(problem)),U32.to_nat(span)),U32.to_nat(Tree.capacity(F32,allocated(rows(problem),span)))) == True{} : Bool}}
  Equal.trans(F32,Tree.read(F32,output(problem,storage_contract,tiles,span),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(span)),column))),
    GemmReadback.wanted(query),Values.matrix_value(problem,channel,output_row,output_column),
    GemmReadback.run_readback(query,allocated(rows(problem),span),Tensors.depth_for(count),Shape.fresh(F32,Tensors.depth_for(count),0.0),depth_ok,output_room),
    wanted_matches(~Evidence,~samples,limits,problem,storage_contract,valid,supported,tiles,offset,span,blocks,channel,column,output_row,output_column,
      certificate,span_exact,span_bound,channel_inside,column_inside,position,row_inside,column_in_width))

def first_group(+span: U32,+blocks: Nat,+span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool}) -> {Nat.is_lt(0n,PackingBounds.groups(span)) == True{} : Bool}:
  %Equal.sym(Nat,PackingBounds.groups(span),blocks,groups_exact(span,blocks,span_exact)) : {Nat.is_lt(0n,_) == True{} : Bool}
  RowMajor.left_factor_positive(blocks,8n,
    %span_exact : {Nat.is_lt(0n,_) == True{} : Bool}
    span_positive)

# The leaf boundary accepts any packing provenance that proves its samples.
# Band callers need neither a global eight-aligned offset nor dense packing.
def every_position_from_samples(~Evidence: LeafContract.Evidence(),~samples: LeafContract.Sampling(~Evidence),+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +input: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+channel: Nat,+column: Nat,+output_row_index: Nat,+output_column_index: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  certificate: Evidence(problem,offset,span,input,output_row_index,output_column_index),+span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},span_alignment: TilePartitionBounds.BlockAligned<span>,
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},
  +position: {Nat.add(U32.to_nat(offset),column) == Coordinates.position(problem,output_row_index,output_column_index) : Nat},
  +row_inside: {Nat.is_lt(output_row_index,ConvSpec.out_height(problem)) == True{} : Bool},+x_inside: {Nat.is_lt(output_column_index,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Tree.read(F32,output(problem,storage_contract,input,span),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(span)),column))) ==
    Values.matrix_value(problem,channel,output_row_index,output_column_index) : F32}:
  TilePartitionBounds.BlockAligned{+blocks,+span_exact} = span_alignment
  +fast = Support.fast(limits,problem,supported)
  +geometry = {LeafBounds.Geometry{Guards.valid(problem,valid),fast,span_bound,TilePartitionBounds.BlockAligned{blocks,span_exact}} : LeafBounds.Geometry<limits,problem,span>}
  +span_positive = Order.le_lt_trans(0n,column,U32.to_nat(span),Order.zero_le(column),column_inside)
  allocation_readback(~Evidence,~samples,limits,problem,storage_contract,valid,supported,input,offset,span,blocks,channel,column,output_row_index,output_column_index,
    certificate,span_exact,span_bound,channel_inside,column_inside,position,row_inside,x_inside,
    PackedOutputAllocation.allocated(limits,problem,span,0n,geometry,first_group(span,blocks,span_exact,span_positive)))

# Preserve the original dense/root contract by supplying its existing evidence.
def every_position_matches_matrix(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +input: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+channel: Nat,+column: Nat,+output_row_index: Nat,+output_column_index: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  certificate: Prepared.LeafCertificate(problem,offset,span,input),
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},
  +position: {Nat.add(U32.to_nat(offset),column) == Coordinates.position(problem,output_row_index,output_column_index) : Nat},
  +row_inside: {Nat.is_lt(output_row_index,ConvSpec.out_height(problem)) == True{} : Bool},+x_inside: {Nat.is_lt(output_column_index,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Tree.read(F32,output(problem,storage_contract,input,span),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(span)),column))) ==
    Values.matrix_value(problem,channel,output_row_index,output_column_index) : F32}:
  Prepared.LeafEvidence{+packing_contract,+span_bound,+offset_alignment,+span_alignment,+range_bound,+contents} = certificate
  +certificate = {Prepared.LeafEvidence{packing_contract,span_bound,offset_alignment,span_alignment,range_bound,contents} :
    Prepared.LeafCertificate(problem,offset,span,input)}
  every_position_from_samples(~LeafContract.dense_evidence,~LeafContract.dense_samples,limits,problem,storage_contract,input,offset,span,channel,column,output_row_index,output_column_index,
    valid,supported,certificate,span_bound,span_alignment,channel_inside,column_inside,position,row_inside,x_inside)

# The actual leaf allocates its packed buffer with the tail-recursive depth.
# Readback converts it to the certified buffer when the leaf has samples.
def packed_tiles(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+offset: U32,+span: U32) -> Tree.Tree<Tile.Tile8>:
  Prepared.packed_output(PackingStorage.packing(U32.to_nat(PackingBounds.count(span,problem)),offset,
    Inputs.input_tree(problem,storage_contract),Tree.fresh(Tile.Tile8,Tensors.depth_for(PackingBounds.count(span,problem)),Tile.filled(0.0)),
    U32.from_nat(Guards.height(problem)),U32.from_nat(Guards.width(problem)),U32.from_nat(Guards.kernel(problem)),U32.from_nat(Guards.stride(problem)),
    U32.from_nat(Guards.pad(problem)),U32.from_nat(ConvSpec.out_width(problem)),Coordinates.native_inner(problem)))

def leaf_output(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+offset: U32,+span: U32) -> Tree.Tree<F32>:
  output(problem,storage_contract,packed_tiles(problem,storage_contract,offset,span),span)

def layout_result(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,layout: TileLayout.Layout()) -> GatherModel.ConvResult():
  match layout:
    case Partition.Leaf{unit,+offset,+span,details}:
      Partition.Leaf{leaf_output(problem,storage_contract,offset,span),offset,span,rows(problem)}
    case Partition.Fork{left,right}:
      Partition.Fork{layout_result(problem,storage_contract,left),layout_result(problem,storage_contract,right)}

def packed_matches(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+offset: U32,+span: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(PackingBounds.count(span,problem))) == True{} : Bool}) ->
  {packed_tiles(problem,storage_contract,offset,span) == Prepared.packed_output(Prepared.allocated_pack(problem,storage_contract,offset,span)) : Tree.Tree<Tile.Tile8>}:
  %Equal.sym(Nat,Tensors.depth_for(PackingBounds.count(span,problem)),Storage.allocation_depth(PackingBounds.count(span,problem)),
    Depth.positive_depth(PackingBounds.count(span,problem),positive)) :
    {Prepared.packed_output(PackingStorage.packing(U32.to_nat(PackingBounds.count(span,problem)),offset,
      Inputs.input_tree(problem,storage_contract),Tree.fresh(Tile.Tile8,_,Tile.filled(0.0)),
      U32.from_nat(Guards.height(problem)),U32.from_nat(Guards.width(problem)),U32.from_nat(Guards.kernel(problem)),U32.from_nat(Guards.stride(problem)),
      U32.from_nat(Guards.pad(problem)),U32.from_nat(ConvSpec.out_width(problem)),Coordinates.native_inner(problem))) ==
      Prepared.packed_output(Prepared.allocated_pack(problem,storage_contract,offset,span)) : Tree.Tree<Tile.Tile8>}
  {==}

def leaf_positive(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+fast: Support.FastSupported<limits,problem>,+span: U32,+blocks: Nat,
  +span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},+span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(PackingBounds.count(span,problem))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(PackingBounds.count(span,problem)),Nat.mul(PackingBounds.groups(span),Coordinates.inner(problem)),
    PackingBounds.count_exact(limits,problem,span,valid,fast,span_bound)) : {Nat.is_lt(0n,_) == True{} : Bool}
  Support.positive_mul(PackingBounds.groups(span),Coordinates.inner(problem),first_group(span,blocks,span_exact,span_positive),Coordinates.inner_positive(problem,valid))

# Gather proofs use this form: the leaf is identified by its layout.
def leaf_every_position(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +input: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+channel: Nat,+column: Nat,+output_row_index: Nat,+output_column_index: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  certificate: Prepared.LeafCertificate(problem,offset,span,input),
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},
  +position: {Nat.add(U32.to_nat(offset),column) == Coordinates.position(problem,output_row_index,output_column_index) : Nat},
  +row_inside: {Nat.is_lt(output_row_index,ConvSpec.out_height(problem)) == True{} : Bool},+x_inside: {Nat.is_lt(output_column_index,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Tree.read(F32,leaf_output(problem,storage_contract,offset,span),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(span)),column))) ==
    Values.matrix_value(problem,channel,output_row_index,output_column_index) : F32}:
  Prepared.LeafEvidence{+packing_contract,+span_bound,+offset_alignment,+span_alignment,+range_bound,+contents} = certificate
  TilePartitionBounds.BlockAligned{+blocks,+span_exact} = span_alignment
  +span_positive = Order.le_lt_trans(0n,column,U32.to_nat(span),Order.zero_le(column),column_inside)
  +positive = leaf_positive(limits,problem,Guards.valid(problem,valid),Support.fast(limits,problem,supported),span,blocks,span_exact,span_bound,span_positive)
  +fresh_certificate = {Prepared.LeafEvidence{storage_contract,span_bound,offset_alignment,TilePartitionBounds.BlockAligned{blocks,span_exact},range_bound,
    packed_matches(problem,storage_contract,offset,span,positive)} : Prepared.LeafCertificate(problem,offset,span,packed_tiles(problem,storage_contract,offset,span))}
  every_position_matches_matrix(limits,problem,storage_contract,packed_tiles(problem,storage_contract,offset,span),offset,span,channel,column,output_row_index,output_column_index,
    valid,supported,fresh_certificate,channel_inside,column_inside,position,row_inside,x_inside)
