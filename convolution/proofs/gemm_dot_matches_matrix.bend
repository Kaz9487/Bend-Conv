# A four-row GEMM job computes the frozen ordered matrix dot for every valid
# row and lane: packed input samples and reordered weights are the spec values.
import Base
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/kernel_rows.bend as Rows
import ../../lib/kernel_tile.bend as Tile
import ../../lib/kernel_weights.bend as Weights
import ../../lib/math_sequence.bend as Sequence
import ../../lib/ordered_matrix.bend as OrderedMatrix
import ../../lib/traversal_reduction.bend as Reduction
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/kernel_accumulator.bend as Accumulator
import ../../lib/proofs/kernel_reduction_model.bend as ReductionModel
import ../../lib/proofs/kernel_coordinate_refinement.bend as CoordinateLaws
import ../../lib/proofs/traversal_reduction_refinement.bend as ReductionLaws
import ../../lib/proofs/ordered_reduction_split.bend as Ordered
import ../matrix_specification.bend as MatrixSpec
import ./matrix_reduction_order.bend as MatrixOrder
import ./gemm_output_readback.bend as GemmReadback
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/kernel_weights_refinement.bend as WeightsLaws
import ../../lib/proofs/kernel_weights_readback.bend as WeightsReadback
import ../../lib/proofs/kernel_block_geometry.bend as Geometry
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/u32_comparison.bend as Comparison
import ../../lib/proofs/natural_addition.bend as Addition
import ./input_storage.bend as Inputs
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./native_weight_refinement.bend as NativeWeightRefinement
import ./reduction_coordinate_bounds.bend as Coordinates
import ./weight_reorder_bounds.bend as WeightBounds
import ./weight_reorder_readback.bend as WeightReadback
import ./packed_matrix_accessors.bend as Accessors
import ./prepared_packing_samples.bend as Prepared
import ./convolution_leaf_contract.bend as LeafContract
import ./packing_tile_bounds.bend as PackingBounds
import ./packing_readback.bend as PackingReadback
import ./matrix_reduction_bounds.bend as ReductionBounds
import ./output_values.bend as Values
import ../inference.bend as NativeConv
import ../../lib/kernel_gemm.bend as Gemm

def row_of(row: Nat) -> Accumulator.Row:
  match row:
    case 0n: Accumulator.R0{}
    case 1n+a:
      match a:
        case 0n: Accumulator.R1{}
        case 1n+b:
          match b:
            case 0n: Accumulator.R2{}
            case 1n+c: Accumulator.R3{}

def lane_of(lane: Nat) -> Sequence.Index(8n):
  TileLaws.lane_of(lane)

def selected_row(+row: Nat,+values: Rows.Rows<Tile.Tile8>) ->
  {GemmReadback.select_row(row,values) == Accumulator.row(~Tile.Tile8,row_of(row),values) : Tile.Tile8}:
  match row:
    case 0n:
      Rows.Rows{a,b,c,d} = values
      {==}
    case 1n+ +a:
      match a:
        case 0n:
          Rows.Rows{a,b,c,d} = values
          {==}
        case 1n+ +b:
          match b:
            case 0n:
              Rows.Rows{a,b,c,d} = values
              {==}
            case 1n+ +c:
              Rows.Rows{a,b,c,d} = values
              {==}

def selected_lane(+lane: Nat,+values: Tile.Tile8) ->
  {GemmReadback.select_lane(lane,values) == Tile.element_at(lane_of(lane),values) : F32}:
  TileLaws.selected_lane(lane,values)

def selected_key(+row: Nat,+lane: Nat,+values: Rows.Rows<Tile.Tile8>) ->
  {GemmReadback.select_lane(lane,GemmReadback.select_row(row,values)) == ReductionModel.accumulator_at(ReductionModel.RowLane{row_of(row),lane_of(lane)},values) : F32}:
  %Equal.sym(Tile.Tile8,GemmReadback.select_row(row,values),Accumulator.row(~Tile.Tile8,row_of(row),values),selected_row(row,values)) :
    {GemmReadback.select_lane(lane,_) == ReductionModel.accumulator_at(ReductionModel.RowLane{row_of(row),lane_of(lane)},values) : F32}
  selected_lane(lane,Accumulator.row(~Tile.Tile8,row_of(row),values))

def selected_dot(upper: Bool,key: ReductionModel.RowLane,count: Nat,state: ReductionModel.State()) -> F32:
  match upper:
    case False{}: ReductionModel.ordered_dot(~False{},key,count,state)
    case True{}: ReductionModel.ordered_dot(~True{},key,count,state)

def accumulator_dot(+upper: Bool,+key: ReductionModel.RowLane,+count: Nat,+state: ReductionModel.State()) ->
  {ReductionModel.accumulator_at(key,Reduction.accumulator(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~U32,~Rows.Rows<Tile.Tile8>,ReductionModel.selected(upper,count,state))) ==
    selected_dot(upper,key,count,state) : F32}:
  match upper:
    case False{}: ReductionModel.run_matches_ordered_dot(~False{},key,count,state)
    case True{}: ReductionModel.run_matches_ordered_dot(~True{},key,count,state)

def initial_zero(+key: ReductionModel.RowLane) -> {ReductionModel.accumulator_at(key,Rows.filled(~Tile.Tile8,Tile.filled(0.0))) == 0.0 : F32}:
  ReductionModel.RowLane{+row,+lane} = key
  match row:
    case Accumulator.R0{}: Accumulator.initial_at(lane,0.0)
    case Accumulator.R1{}: Accumulator.initial_at(lane,0.0)
    case Accumulator.R2{}: Accumulator.initial_at(lane,0.0)
    case Accumulator.R3{}: Accumulator.initial_at(lane,0.0)

def Source() -> Data:
  ReductionLaws.SampleSource<Tile.Tile8,Tile.Tile8,U32>

def input_sample() -> Source() -> ReductionModel.RowLane -> ReductionLaws.AddressPair -> F32:
  ReductionLaws.input_element(~Tile.Tile8,~Tile.Tile8,~U32,~Tile.Tile8,~ReductionModel.RowLane,~ReductionModel.input_tile,~ReductionModel.input_at)

def weight_sample(~upper: Bool) -> Source() -> ReductionModel.RowLane -> ReductionLaws.AddressPair -> F32:
  ReductionLaws.weight_element(~Tile.Tile8,~Tile.Tile8,~U32,~Rows.Rows<F32>,~ReductionModel.RowLane,~ReductionModel.weights_half(~upper),~ReductionModel.weight_at)

def Mapped() -> Data:
  Shape.Both<MatrixOrder.TapLayout,Source()>

def mapped_input() -> Mapped() -> ReductionModel.RowLane -> MatrixSpec.Reduction -> F32:
  Ordered.mapped_sample(~MatrixSpec.Reduction,~ReductionLaws.AddressPair,~ReductionModel.RowLane,~Source(),~MatrixOrder.TapLayout,~MatrixOrder.mapped_tap,~input_sample())

def mapped_weight(~upper: Bool) -> Mapped() -> ReductionModel.RowLane -> MatrixSpec.Reduction -> F32:
  Ordered.mapped_sample(~MatrixSpec.Reduction,~ReductionLaws.AddressPair,~ReductionModel.RowLane,~Source(),~MatrixOrder.TapLayout,~MatrixOrder.mapped_tap,~weight_sample(~upper))

def matrix_keyed_dot(~upper: Bool,+key: ReductionModel.RowLane,+channels: Nat,+kernel: Nat,+input_base: Nat,+source: Source()) -> F32:
  OrderedMatrix.dot(~ReductionModel.RowLane,~MatrixSpec.Reduction,~ReductionModel.RowLane,~Mapped(),~mapped_input(),~mapped_weight(~upper),
    MatrixSpec.kchannels(channels,0n,kernel),key,key,Shape.Both{MatrixOrder.TapLayout{kernel,input_base,0n},source},0.0)

# The job reads consecutive K addresses; they are exactly the frozen K axis.
def reindexed(~upper: Bool,+key: ReductionModel.RowLane,+input: Tree.Tree<Tile.Tile8>,+weights: Tree.Tree<Tile.Tile8>,+base: U32,
  +channels: Nat,+kernel: Nat,+input_base: Nat) ->
  {ReductionModel.ordered_dot(~upper,key,Nat.mul(channels,Nat.mul(kernel,kernel)),ReductionModel.initial(input,weights,U32.from_nat(input_base),U32.from_nat(0n),base)) ==
    matrix_keyed_dot(~upper,key,channels,kernel,input_base,ReductionLaws.SampleSource{input,weights,base}) : F32}:
  +source = {ReductionLaws.SampleSource{input,weights,base} : Source()}
  %Equal.sym(F32,ReductionModel.accumulator_at(key,Rows.filled(~Tile.Tile8,Tile.filled(0.0))),0.0,initial_zero(key)) :
    {OrderedMatrix.dot(~ReductionModel.RowLane,~ReductionLaws.AddressPair,~ReductionModel.RowLane,~Source(),~input_sample(),~weight_sample(~upper),
      ReductionLaws.addresses(Nat.mul(channels,Nat.mul(kernel,kernel)),U32.from_nat(input_base),U32.from_nat(0n)),key,key,source,_) ==
      matrix_keyed_dot(~upper,key,channels,kernel,input_base,source) : F32}
  %Equal.sym(List<&2,ReductionLaws.AddressPair>,ReductionLaws.addresses(Nat.mul(channels,Nat.mul(kernel,kernel)),U32.from_nat(input_base),U32.from_nat(0n)),
    MatrixOrder.matrix_taps(MatrixSpec.kchannels(channels,0n,kernel),kernel,input_base,0n),
    MatrixOrder.kernel_taps_follow_matrix_reduction(channels,kernel,input_base,0n)) :
    {OrderedMatrix.dot(~ReductionModel.RowLane,~ReductionLaws.AddressPair,~ReductionModel.RowLane,~Source(),~input_sample(),~weight_sample(~upper),_,key,key,source,0.0) ==
      matrix_keyed_dot(~upper,key,channels,kernel,input_base,source) : F32}
  Ordered.dot_reindexed(~MatrixSpec.Reduction,~ReductionLaws.AddressPair,~ReductionModel.RowLane,~ReductionModel.RowLane,~Source(),~MatrixOrder.TapLayout,~MatrixOrder.mapped_tap,
    ~input_sample(),~weight_sample(~upper),MatrixSpec.kchannels(channels,0n,kernel),MatrixOrder.TapLayout{kernel,input_base,0n},key,key,source,0.0)

def reordered(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>) -> Tree.Tree<Tile.Tile8>:
  WeightsLaws.run(U32.from_nat(Guards.output_channel_count(problem)),Coordinates.native_inner(problem),Inputs.weight_tree(problem,storage_contract))

def weights_capacity(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+fast: Support.FastSupported<limits,problem>) ->
  {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),Coordinates.inner(problem)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Guards.Valid{v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10} = valid
  %Guards.reflect(ConvSpec.length(Guards.weights(problem)),Nat.mul(Guards.output_channel_count(problem),Coordinates.inner(problem)),v9) :
    {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  NativeWeightRefinement.weights_bound(limits,problem,Padded.scalar(limits,problem,fast))

def lane_word(first: U32,lane: Sequence.Index(8n)) -> U32:
  match lane:
    case Sequence.First{}: first
    case Sequence.Next{Sequence.First{}}: (first + 1 : U32)
    case Sequence.Next{Sequence.Next{Sequence.First{}}}: (first + 2 : U32)
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}: (first + 3 : U32)
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}: (first + 4 : U32)
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}}: (first + 5 : U32)
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}}}: (first + 6 : U32)
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}}}}: (first + 7 : U32)
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{empty}}}}}}}}:
      match empty:

def sample_lane(+tree: Tree.Tree<F32>,+rows: U32,+inner: U32,+first: U32,+destination: U32,+index: U32,+lane: Sequence.Index(8n)) ->
  {Tile.element_at(lane,WeightsReadback.sample(tree,Weights.Context{rows,inner,first,destination},index)) ==
    WeightsReadback.value(tree,rows,inner,lane_word(first,lane),index) : F32}:
  match lane:
    case Sequence.First{}: {==}
    case Sequence.Next{Sequence.First{}}: {==}
    case Sequence.Next{Sequence.Next{Sequence.First{}}}: {==}
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}: {==}
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}: {==}
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}}: {==}
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}}}: {==}
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}}}}: {==}
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{empty}}}}}}}}:
      match empty:

def half_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+fast: Support.FastSupported<limits,problem>,+value: Nat,
  inside: {Nat.is_lt(value,Guards.output_channel_count(problem)) == True{} : Bool}) -> {Nat.is_le(value,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  Order.transitive(value,Nat.add(Guards.output_channel_count(problem),7n),U32.to_nat(Machine.half_word(limits)),
    Order.transitive(value,Guards.output_channel_count(problem),Nat.add(Guards.output_channel_count(problem),7n),Bounds.strict_le(value,Guards.output_channel_count(problem),inside),
      Order.le_add(Guards.output_channel_count(problem),7n)),
    WeightBounds.padded_rows_bound(limits,problem,fast))

def offset_row(+limits: Machine.Limits,+first: U32,+base: Nat,+offset: Nat,+channel: Nat,
  +first_exact: {U32.to_nat(first) == base : Nat},+channel_exact: {channel == Nat.add(base,offset) : Nat},
  +channel_small: {Nat.is_le(channel,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {U32.to_nat((first + U32.from_nat(offset) : U32)) == channel : Nat}:
  +half = Machine.half_word(limits)
  +sum_small = {%channel_exact : {Nat.is_le(_,U32.to_nat(half)) == True{} : Bool}
    channel_small : {Nat.is_le(Nat.add(base,offset),U32.to_nat(half)) == True{} : Bool}}
  +offset_exact = Bounds.roundtrip(offset,half,Limits.half_word_fits(limits),
    Order.transitive(offset,Nat.add(base,offset),U32.to_nat(half),
      %Order.commutative(offset,base) : {Nat.is_le(offset,_) == True{} : Bool}
      Order.le_add(offset,base),sum_small))
  %Equal.sym(Nat,channel,Nat.add(base,offset),channel_exact) : {U32.to_nat((first + U32.from_nat(offset) : U32)) == _ : Nat}
  %first_exact : {U32.to_nat((first + U32.from_nat(offset) : U32)) == Nat.add(_,offset) : Nat}
  %offset_exact : {U32.to_nat((first + U32.from_nat(offset) : U32)) == Nat.add(U32.to_nat(first),_) : Nat}
  Arithmetic.add(first,U32.from_nat(offset),half,Limits.half_word_fits(limits),
    %Equal.sym(Nat,U32.to_nat(first),base,first_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(U32.from_nat(offset))),U32.to_nat(half)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(offset)),offset,offset_exact) : {Nat.is_le(Nat.add(base,_),U32.to_nat(half)) == True{} : Bool}
    sum_small)

def first_word_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+fast: Support.FastSupported<limits,problem>,+block: Nat,+lane: Sequence.Index(8n),+channel: Nat,
  +channel_exact: {channel == Nat.add(Nat.mul(block,8n),Sequence.index_value(8n,lane)) : Nat},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {U32.to_nat((U32.from_nat(block) * 8 : U32)) == Nat.mul(block,8n) : Nat}:
  +half = Machine.half_word(limits)
  +fits = Limits.half_word_fits(limits)
  +channel_small = half_room(limits,problem,fast,channel,channel_inside)
  +first_small = Order.transitive(Nat.mul(block,8n),channel,U32.to_nat(half),
    %Equal.sym(Nat,channel,Nat.add(Nat.mul(block,8n),Sequence.index_value(8n,lane)),channel_exact) : {Nat.is_le(Nat.mul(block,8n),_) == True{} : Bool}
    Order.le_add(Nat.mul(block,8n),Sequence.index_value(8n,lane)),channel_small)
  +block_exact = Bounds.roundtrip(block,half,fits,Order.transitive(block,Nat.mul(block,8n),U32.to_nat(half),RowMajor.product_covers(block,8n,{==}),first_small))
  %block_exact : {U32.to_nat((U32.from_nat(block) * 8 : U32)) == Nat.mul(_,8n) : Nat}
  Arithmetic.multiply(U32.from_nat(block),8,half,fits,
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(block)),block,block_exact) : {Nat.is_le(Nat.mul(_,8n),U32.to_nat(half)) == True{} : Bool}
    first_small)

def lane_word_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+fast: Support.FastSupported<limits,problem>,+block: Nat,+lane: Sequence.Index(8n),+channel: Nat,
  +channel_exact: {channel == Nat.add(Nat.mul(block,8n),Sequence.index_value(8n,lane)) : Nat},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {U32.to_nat(lane_word((U32.from_nat(block) * 8 : U32),lane)) == channel : Nat}:
  match lane:
    case Sequence.First{}:
      %Equal.sym(Nat,channel,Nat.add(Nat.mul(block,8n),0n),channel_exact) : {U32.to_nat((U32.from_nat(block) * 8 : U32)) == _ : Nat}
      %Equal.sym(Nat,Nat.add(Nat.mul(block,8n),0n),Nat.mul(block,8n),Addition.zero_right(Nat.mul(block,8n))) : {U32.to_nat((U32.from_nat(block) * 8 : U32)) == _ : Nat}
      first_word_exact(limits,problem,fast,block,Sequence.First{},channel,channel_exact,channel_inside)
    case Sequence.Next{Sequence.First{}}:
      offset_row(limits,(U32.from_nat(block) * 8 : U32),Nat.mul(block,8n),1n,channel,first_word_exact(limits,problem,fast,block,Sequence.Next{Sequence.First{}},channel,channel_exact,channel_inside),
        channel_exact,half_room(limits,problem,fast,channel,channel_inside))
    case Sequence.Next{Sequence.Next{Sequence.First{}}}:
      offset_row(limits,(U32.from_nat(block) * 8 : U32),Nat.mul(block,8n),2n,channel,first_word_exact(limits,problem,fast,block,Sequence.Next{Sequence.Next{Sequence.First{}}},channel,channel_exact,channel_inside),
        channel_exact,half_room(limits,problem,fast,channel,channel_inside))
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}:
      offset_row(limits,(U32.from_nat(block) * 8 : U32),Nat.mul(block,8n),3n,channel,first_word_exact(limits,problem,fast,block,Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}},channel,channel_exact,channel_inside),
        channel_exact,half_room(limits,problem,fast,channel,channel_inside))
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}:
      offset_row(limits,(U32.from_nat(block) * 8 : U32),Nat.mul(block,8n),4n,channel,first_word_exact(limits,problem,fast,block,Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}},channel,channel_exact,channel_inside),
        channel_exact,half_room(limits,problem,fast,channel,channel_inside))
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}}:
      offset_row(limits,(U32.from_nat(block) * 8 : U32),Nat.mul(block,8n),5n,channel,first_word_exact(limits,problem,fast,block,Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}},channel,channel_exact,channel_inside),
        channel_exact,half_room(limits,problem,fast,channel,channel_inside))
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}}}:
      offset_row(limits,(U32.from_nat(block) * 8 : U32),Nat.mul(block,8n),6n,channel,first_word_exact(limits,problem,fast,block,Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}}},channel,channel_exact,channel_inside),
        channel_exact,half_room(limits,problem,fast,channel,channel_inside))
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}}}}:
      offset_row(limits,(U32.from_nat(block) * 8 : U32),Nat.mul(block,8n),7n,channel,first_word_exact(limits,problem,fast,block,Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.First{}}}}}}}},channel,channel_exact,channel_inside),
        channel_exact,half_room(limits,problem,fast,channel,channel_inside))
    case Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{Sequence.Next{empty}}}}}}}}:
      match empty:

def word_from_exact(+word: U32,+value: Nat,exact: {U32.to_nat(word) == value : Nat}) -> {word == U32.from_nat(value) : U32}:
  %exact : {word == U32.from_nat(_) : U32}
  Equal.sym(U32,U32.from_nat(U32.to_nat(word)),word,Arithmetic.native_roundtrip(word))

def weight_address(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+fast: Support.FastSupported<limits,problem>,
  +block: Nat,+channel: Nat,+base: U32,+tap: Nat,
  +base_exact: {U32.to_nat(base) == Nat.mul(block,Coordinates.inner(problem)) : Nat},
  +block_le: {Nat.is_le(block,channel) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,Coordinates.inner(problem)) == True{} : Bool}) ->
  {(base + U32.from_nat(tap) : U32) == U32.from_nat(Nat.add(Nat.mul(block,Coordinates.inner(problem)),tap)) : U32}:
  +inner = Coordinates.inner(problem)
  +cap = Machine.capacity(limits)
  +fits = Limits.capacity_fits(limits)
  +target = Nat.add(Nat.mul(block,inner),tap)
  +small = Order.transitive(target,Nat.mul(Guards.output_channel_count(problem),inner),U32.to_nat(cap),
    Bounds.strict_le(target,Nat.mul(Guards.output_channel_count(problem),inner),
      Order.le_lt_trans(target,Nat.add(Nat.mul(channel,inner),tap),Nat.mul(Guards.output_channel_count(problem),inner),
        Order.add_le(Nat.mul(block,inner),Nat.mul(channel,inner),tap,RowMajor.mul_le(block,channel,inner,block_le)),
        RowMajor.radix(channel,Guards.output_channel_count(problem),tap,inner,channel_inside,tap_inside))),
    weights_capacity(limits,problem,valid,fast))
  +tap_exact = Bounds.roundtrip(tap,cap,fits,Order.transitive(tap,target,U32.to_nat(cap),
    %Order.commutative(tap,Nat.mul(block,inner)) : {Nat.is_le(tap,_) == True{} : Bool}
    Order.le_add(tap,Nat.mul(block,inner)),small))
  word_from_exact((base + U32.from_nat(tap) : U32),target,
    %base_exact : {U32.to_nat((base + U32.from_nat(tap) : U32)) == Nat.add(_,tap) : Nat}
    %tap_exact : {U32.to_nat((base + U32.from_nat(tap) : U32)) == Nat.add(U32.to_nat(base),_) : Nat}
    Arithmetic.add(base,U32.from_nat(tap),cap,fits,
      %Equal.sym(Nat,U32.to_nat(base),Nat.mul(block,inner),base_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(U32.from_nat(tap))),U32.to_nat(cap)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(tap)),tap,tap_exact) : {Nat.is_le(Nat.add(Nat.mul(block,inner),_),U32.to_nat(cap)) == True{} : Bool}
      small))

def value_read(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+fast: Support.FastSupported<limits,problem>,
  +storage_contract: Inputs.Contract<problem>,+channel: Nat,+tap: Nat,+word: U32,
  +word_exact: {U32.to_nat(word) == channel : Nat},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +tap_inside: {Nat.is_lt(tap,Coordinates.inner(problem)) == True{} : Bool}) ->
  {WeightsReadback.value(Inputs.weight_tree(problem,storage_contract),U32.from_nat(Guards.output_channel_count(problem)),Coordinates.native_inner(problem),word,U32.from_nat(tap)) ==
    Tree.read(F32,Inputs.weight_tree(problem,storage_contract),U32.from_nat(Nat.add(Nat.mul(channel,Coordinates.inner(problem)),tap))) : F32}:
  +inner = Coordinates.inner(problem)
  +native = Coordinates.native_inner(problem)
  +rows = U32.from_nat(Guards.output_channel_count(problem))
  +cap = Machine.capacity(limits)
  +fits = Limits.capacity_fits(limits)
  +target = Nat.add(Nat.mul(channel,inner),tap)
  +inner_exact = Coordinates.inner_exact(limits,problem,valid,Padded.scalar(limits,problem,fast))
  +small = Order.transitive(target,Nat.mul(Guards.output_channel_count(problem),inner),U32.to_nat(cap),
    Bounds.strict_le(target,Nat.mul(Guards.output_channel_count(problem),inner),RowMajor.radix(channel,Guards.output_channel_count(problem),tap,inner,channel_inside,tap_inside)),
    weights_capacity(limits,problem,valid,fast))
  +product_small = Order.transitive(Nat.mul(channel,inner),target,U32.to_nat(cap),Order.le_add(Nat.mul(channel,inner),tap),small)
  +tap_exact = Bounds.roundtrip(tap,cap,fits,Order.transitive(tap,target,U32.to_nat(cap),
    %Order.commutative(tap,Nat.mul(channel,inner)) : {Nat.is_le(tap,_) == True{} : Bool}
    Order.le_add(tap,Nat.mul(channel,inner)),small))
  +product = (word * native : U32)
  +product_exact = {%word_exact : {U32.to_nat(product) == Nat.mul(_,inner) : Nat}
    %inner_exact : {U32.to_nat(product) == Nat.mul(U32.to_nat(word),_) : Nat}
    Arithmetic.multiply(word,native,cap,fits,
      %Equal.sym(Nat,U32.to_nat(word),channel,word_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(native)),U32.to_nat(cap)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(native),inner,inner_exact) : {Nat.is_le(Nat.mul(channel,_),U32.to_nat(cap)) == True{} : Bool}
      product_small) : {U32.to_nat(product) == Nat.mul(channel,inner) : Nat}}
  +address = word_from_exact((product + U32.from_nat(tap) : U32),target,
    %product_exact : {U32.to_nat((product + U32.from_nat(tap) : U32)) == Nat.add(_,tap) : Nat}
    %tap_exact : {U32.to_nat((product + U32.from_nat(tap) : U32)) == Nat.add(U32.to_nat(product),_) : Nat}
    Arithmetic.add(product,U32.from_nat(tap),cap,fits,
      %Equal.sym(Nat,U32.to_nat(product),Nat.mul(channel,inner),product_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(U32.from_nat(tap))),U32.to_nat(cap)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(tap)),tap,tap_exact) : {Nat.is_le(Nat.add(Nat.mul(channel,inner),_),U32.to_nat(cap)) == True{} : Bool}
      small))
  +accepted = {%Equal.sym(Bool,U32.is_lt(word,rows),Nat.is_lt(U32.to_nat(word),U32.to_nat(rows)),Comparison.lt(word,rows)) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(word),channel,word_exact) : {Nat.is_lt(_,U32.to_nat(rows)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(rows),Guards.output_channel_count(problem),SupportedCoordinateBounds.co_exact(limits,problem,Padded.scalar(limits,problem,fast))) :
      {Nat.is_lt(channel,_) == True{} : Bool}
    channel_inside : {U32.is_lt(word,rows) == True{} : Bool}}
  %Equal.sym(Bool,U32.is_lt(word,rows),True{},accepted) :
    {Bool.pick(F32,_,Tree.read(F32,Inputs.weight_tree(problem,storage_contract),(word * native + U32.from_nat(tap) : U32)),0.0) ==
      Tree.read(F32,Inputs.weight_tree(problem,storage_contract),U32.from_nat(target)) : F32}
  %Equal.sym(U32,(product + U32.from_nat(tap) : U32),U32.from_nat(target),address) :
    {Tree.read(F32,Inputs.weight_tree(problem,storage_contract),_) == Tree.read(F32,Inputs.weight_tree(problem,storage_contract),U32.from_nat(target)) : F32}
  {==}

def mapped_source(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+tiles: Tree.Tree<Tile.Tile8>,+group: Nat,+base: U32) -> Mapped():
  Shape.Both{MatrixOrder.TapLayout{Guards.kernel(problem),Nat.mul(group,Coordinates.inner(problem)),0n},ReductionLaws.SampleSource{tiles,reordered(problem,storage_contract),base}}

def input_agrees(~Evidence: LeafContract.Evidence(),~samples: LeafContract.Sampling(~Evidence),+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+tiles: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,
  +base: U32,+group: Nat,+row: Accumulator.Row,+lane: Sequence.Index(8n),+output_row: Nat,+output_column: Nat,+index: MatrixSpec.Reduction,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +certificate: Evidence(problem,offset,span,tiles,output_row,output_column),
  +group_inside: {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool},
  +position: {PackingReadback.logical_position(offset,group,lane) == Coordinates.position(problem,output_row,output_column) : Nat},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool},
  +bounds: ReductionBounds.IndexBounds(problem,index)) ->
  {mapped_input()(mapped_source(problem,storage_contract,tiles,group,base),ReductionModel.RowLane{row,lane},index) ==
    MatrixSpec.patch(problem,MatrixSpec.Position{output_row,output_column},index) : F32}:
  MatrixSpec.KIndex{+channel,+kernel_row,+kernel_column} = index
  Shape.Both{+channel_inside,+spatial} = bounds
  Shape.Both{+kernel_row_inside,+kernel_column_inside} = spatial
  samples(limits,problem,offset,span,tiles,output_row,output_column,valid,supported,certificate)(lane,group,channel,kernel_row,kernel_column,
    group_inside,position,channel_inside,row_inside,column_inside,kernel_row_inside,kernel_column_inside)

def block_le_channel(+block: Nat,+lane: Sequence.Index(8n),+channel: Nat,+channel_exact: {channel == Nat.add(Nat.mul(block,8n),Sequence.index_value(8n,lane)) : Nat}) ->
  {Nat.is_le(Nat.mul(block,8n),channel) == True{} : Bool}:
  %Equal.sym(Nat,channel,Nat.add(Nat.mul(block,8n),Sequence.index_value(8n,lane)),channel_exact) : {Nat.is_le(Nat.mul(block,8n),_) == True{} : Bool}
  Order.le_add(Nat.mul(block,8n),Sequence.index_value(8n,lane))

def weight_agrees(~upper: Bool,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +tiles: Tree.Tree<Tile.Tile8>,+base: U32,+group: Nat,+block: Nat,+row: Accumulator.Row,+lane: Sequence.Index(8n),+channel: Nat,+index: MatrixSpec.Reduction,
  +base_exact: {U32.to_nat(base) == Nat.mul(block,Coordinates.inner(problem)) : Nat},
  +channel_exact: {channel == Nat.add(Nat.mul(block,8n),Sequence.index_value(8n,CoordinateLaws.weight_lane(upper,row))) : Nat},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +bounds: ReductionBounds.IndexBounds(problem,index)) ->
  {mapped_weight(~upper)(mapped_source(problem,storage_contract,tiles,group,base),ReductionModel.RowLane{row,lane},index) == MatrixSpec.weight(problem,channel,index) : F32}:
  MatrixSpec.KIndex{+input_channel,+kernel_row,+kernel_column} = index
  Shape.Both{+input_inside,+spatial} = bounds
  Shape.Both{+kernel_row_inside,+kernel_column_inside} = spatial
  +facts = Guards.valid(problem,valid)
  +fast = Support.fast(limits,problem,supported)
  +tap = Coordinates.reduction(input_channel,kernel_row,kernel_column,Guards.kernel(problem))
  +tap_inside = Coordinates.reduction_inside(problem,input_channel,kernel_row,kernel_column,input_inside,kernel_row_inside,kernel_column_inside)
  +weights = Inputs.weight_tree(problem,storage_contract)
  +rows = U32.from_nat(Guards.output_channel_count(problem))
  +native = Coordinates.native_inner(problem)
  +address = (base + U32.from_nat(tap) : U32)
  +target = U32.from_nat(Nat.add(Nat.mul(block,Coordinates.inner(problem)),tap))
  +first = (U32.from_nat(block) * 8 : U32)
  +destination = (U32.from_nat(block) * native : U32)
  +lane8 = CoordinateLaws.weight_lane(upper,row)
  +eight_le = block_le_channel(block,lane8,channel,channel_exact)
  +block_le = Order.transitive(block,Nat.mul(block,8n),channel,RowMajor.product_covers(block,8n,{==}),eight_le)
  +block_inside = Geometry.contains_row(Guards.output_channel_count(problem),block,Order.le_lt_trans(Nat.mul(block,8n),channel,Guards.output_channel_count(problem),eight_le,channel_inside))
  %Equal.sym(F32,Accumulator.row(~F32,row,Rows.half(upper,Tree.read(Tile.Tile8,reordered(problem,storage_contract),address))),
    Tile.element_at(lane8,Tree.read(Tile.Tile8,reordered(problem,storage_contract),address)),
    CoordinateLaws.weight_sample(upper,row,Tree.read(Tile.Tile8,reordered(problem,storage_contract),address))) :
    {_ == MatrixSpec.weight(problem,channel,MatrixSpec.KIndex{input_channel,kernel_row,kernel_column}) : F32}
  %Equal.sym(U32,address,target,weight_address(limits,problem,facts,fast,block,channel,base,tap,base_exact,block_le,channel_inside,tap_inside)) :
    {Tile.element_at(lane8,Tree.read(Tile.Tile8,reordered(problem,storage_contract),_)) == MatrixSpec.weight(problem,channel,MatrixSpec.KIndex{input_channel,kernel_row,kernel_column}) : F32}
  %Equal.sym(F32,Tile.element_at(lane8,Tree.read(Tile.Tile8,reordered(problem,storage_contract),target)),
    Tile.element_at(lane8,WeightsReadback.sample(weights,Weights.Context{rows,native,first,destination},U32.from_nat(tap))),
    WeightReadback.run_readback(limits,problem,facts,fast,weights,lane8,block,tap,block_inside,tap_inside)) :
    {_ == MatrixSpec.weight(problem,channel,MatrixSpec.KIndex{input_channel,kernel_row,kernel_column}) : F32}
  %Equal.sym(F32,Tile.element_at(lane8,WeightsReadback.sample(weights,Weights.Context{rows,native,first,destination},U32.from_nat(tap))),
    WeightsReadback.value(weights,rows,native,lane_word(first,lane8),U32.from_nat(tap)),sample_lane(weights,rows,native,first,destination,U32.from_nat(tap),lane8)) :
    {_ == MatrixSpec.weight(problem,channel,MatrixSpec.KIndex{input_channel,kernel_row,kernel_column}) : F32}
  %Equal.sym(F32,WeightsReadback.value(weights,rows,native,lane_word(first,lane8),U32.from_nat(tap)),
    Tree.read(F32,weights,U32.from_nat(Nat.add(Nat.mul(channel,Coordinates.inner(problem)),tap))),
    value_read(limits,problem,facts,fast,storage_contract,channel,tap,lane_word(first,lane8),
      lane_word_exact(limits,problem,fast,block,lane8,channel,channel_exact,channel_inside),channel_inside,tap_inside)) :
    {_ == MatrixSpec.weight(problem,channel,MatrixSpec.KIndex{input_channel,kernel_row,kernel_column}) : F32}
  Accessors.first_weight_matches_matrix(problem,storage_contract,tiles,0n,channel,input_channel,kernel_row,kernel_column,valid,
    channel_inside,input_inside,kernel_row_inside,kernel_column_inside)

def Agreements(~upper: Bool,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+tiles: Tree.Tree<Tile.Tile8>,+group: Nat,+base: U32,
  +row: Accumulator.Row,+lane: Sequence.Index(8n),+channel: Nat,+output_row: Nat,+output_column: Nat,+indices: List<&2,MatrixSpec.Reduction>) -> Data:
  Ordered.MatchingSamples(~MatrixSpec.Reduction,~ReductionModel.RowLane,~ReductionModel.RowLane,~Mapped(),~Nat,~MatrixSpec.Pixel,~ConvSpec.Problem,
    ~mapped_input(),~mapped_weight(~upper),~MatrixSpec.patch,~MatrixSpec.weight,
    indices,ReductionModel.RowLane{row,lane},ReductionModel.RowLane{row,lane},mapped_source(problem,storage_contract,tiles,group,base),channel,MatrixSpec.Position{output_row,output_column},problem)

def agreements(~Evidence: LeafContract.Evidence(),~samples: LeafContract.Sampling(~Evidence),~upper: Bool,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +tiles: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+base: U32,+group: Nat,+block: Nat,+row: Accumulator.Row,+lane: Sequence.Index(8n),
  +channel: Nat,+output_row: Nat,+output_column: Nat,+indices: List<&2,MatrixSpec.Reduction>,
  +certificate: Evidence(problem,offset,span,tiles,output_row,output_column),
  +group_inside: {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool},
  +position: {PackingReadback.logical_position(offset,group,lane) == Coordinates.position(problem,output_row,output_column) : Nat},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool},
  +base_exact: {U32.to_nat(base) == Nat.mul(block,Coordinates.inner(problem)) : Nat},
  +channel_exact: {channel == Nat.add(Nat.mul(block,8n),Sequence.index_value(8n,CoordinateLaws.weight_lane(upper,row))) : Nat},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  bounds: ReductionBounds.AllBounds(problem,indices)) ->
  Agreements(~upper,problem,storage_contract,tiles,group,base,row,lane,channel,output_row,output_column,indices):
  match indices:
    case Nil{}: Unit{}
    case +index <> +rest:
      Shape.Both{+head,tail} = bounds
      Shape.Both{
        Shape.Both{
          input_agrees(~Evidence,~samples,limits,problem,storage_contract,tiles,offset,span,base,group,row,lane,output_row,output_column,index,valid,supported,certificate,group_inside,position,row_inside,column_inside,head),
          weight_agrees(~upper,limits,problem,storage_contract,valid,supported,tiles,base,group,block,row,lane,channel,index,base_exact,channel_exact,channel_inside,head)},
        agreements(~Evidence,~samples,~upper,limits,problem,storage_contract,valid,supported,tiles,offset,span,base,group,block,row,lane,channel,output_row,output_column,rest,
          certificate,group_inside,position,row_inside,column_inside,base_exact,channel_exact,channel_inside,tail)}

# One job row/lane equals the frozen ordered matrix dot for its coordinate.
def job_dot(~Evidence: LeafContract.Evidence(),~samples: LeafContract.Sampling(~Evidence),~upper: Bool,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +tiles: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+base: U32,+group: Nat,+block: Nat,+row: Accumulator.Row,+lane: Sequence.Index(8n),
  +channel: Nat,+output_row: Nat,+output_column: Nat,
  +certificate: Evidence(problem,offset,span,tiles,output_row,output_column),
  +group_inside: {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool},
  +position: {PackingReadback.logical_position(offset,group,lane) == Coordinates.position(problem,output_row,output_column) : Nat},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool},
  +base_exact: {U32.to_nat(base) == Nat.mul(block,Coordinates.inner(problem)) : Nat},
  +channel_exact: {channel == Nat.add(Nat.mul(block,8n),Sequence.index_value(8n,CoordinateLaws.weight_lane(upper,row))) : Nat},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {ReductionModel.ordered_dot(~upper,ReductionModel.RowLane{row,lane},Coordinates.inner(problem),
      ReductionModel.initial(tiles,reordered(problem,storage_contract),U32.from_nat(Nat.mul(group,Coordinates.inner(problem))),U32.from_nat(0n),base)) ==
    Values.matrix_dot(MatrixSpec.kchannels(Guards.input_channel_count(problem),0n,Guards.kernel(problem)),problem,channel,output_row,output_column,0.0) : F32}:
  +key = {ReductionModel.RowLane{row,lane} : ReductionModel.RowLane}
  +channels = Guards.input_channel_count(problem)
  +kernel = Guards.kernel(problem)
  +input_base = Nat.mul(group,Coordinates.inner(problem))
  +source = {ReductionLaws.SampleSource{tiles,reordered(problem,storage_contract),base} : Source()}
  Equal.trans(F32,
    ReductionModel.ordered_dot(~upper,key,Coordinates.inner(problem),ReductionModel.initial(tiles,reordered(problem,storage_contract),U32.from_nat(input_base),U32.from_nat(0n),base)),
    matrix_keyed_dot(~upper,key,channels,kernel,input_base,source),
    Values.matrix_dot(MatrixSpec.kchannels(channels,0n,kernel),problem,channel,output_row,output_column,0.0),
    reindexed(~upper,key,tiles,reordered(problem,storage_contract),base,channels,kernel,input_base),
    Ordered.dot_congruence(~MatrixSpec.Reduction,~ReductionModel.RowLane,~ReductionModel.RowLane,~Mapped(),~Nat,~MatrixSpec.Pixel,~ConvSpec.Problem,
      ~mapped_input(),~mapped_weight(~upper),~MatrixSpec.patch,~MatrixSpec.weight,
      MatrixSpec.kchannels(channels,0n,kernel),key,key,mapped_source(problem,storage_contract,tiles,group,base),channel,MatrixSpec.Position{output_row,output_column},problem,0.0,
      agreements(~Evidence,~samples,~upper,limits,problem,storage_contract,valid,supported,tiles,offset,span,base,group,block,row,lane,channel,output_row,output_column,
        MatrixSpec.kchannels(channels,0n,kernel),certificate,group_inside,position,row_inside,column_inside,base_exact,channel_exact,channel_inside,
        ReductionBounds.matrix_reduction_in_bounds(problem))))
