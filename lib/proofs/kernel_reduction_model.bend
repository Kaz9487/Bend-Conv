# Shared ordered-reduction model for the four-row native accumulator. The
# native loop refinement is separate; this module never substitutes a model for it.
# Model indices: input at the absolute tile address, weights relative to base.
import Base
import ../kernel_gemm.bend as Gemm
import ../kernel_rows.bend as Rows
import ../kernel_tile.bend as Tile
import ../math_sequence.bend as Sequence
import ../math_fp32.bend as Fp32
import ../traversal_reduction.bend as Reduction
import ./storage_tree_model.bend as Tree
import ./kernel_accumulator.bend as Accumulator
import ./traversal_reduction_refinement.bend as ReductionLaws

def State() -> Data:
  Reduction.State<&2,Tree.Tree<Tile.Tile8>,Tree.Tree<Tile.Tile8>,U32,Rows.Rows<Tile.Tile8>>

def input_tile(tree: Tree.Tree<Tile.Tile8>,base: U32,index: U32) -> Tile.Tile8:
  Tree.read(Tile.Tile8,tree,index)

def weights_half(~upper: Bool,tree: Tree.Tree<Tile.Tile8>,base: U32,index: U32) -> Rows.Rows<F32>:
  Rows.half(upper,Tree.read(Tile.Tile8,tree,(base + index : U32)))

def update(base: U32,accumulators: Rows.Rows<Tile.Tile8>,input: Tile.Tile8,weights: Rows.Rows<F32>) -> Rows.Rows<Tile.Tile8>:
  Gemm.update(accumulators,input,weights)

type RowLane is Data:
  RowLane{row: Accumulator.Row,lane: Sequence.Index(8n)}

def input_at(key: RowLane,input: Tile.Tile8) -> F32:
  RowLane{row,lane} = key
  Tile.element_at(lane,input)

def weight_at(key: RowLane,weights: Rows.Rows<F32>) -> F32:
  RowLane{row,lane} = key
  Accumulator.row(~F32,row,weights)

def accumulator_at(key: RowLane,accumulator: Rows.Rows<Tile.Tile8>) -> F32:
  RowLane{row,lane} = key
  Tile.element_at(lane,Accumulator.row(~Tile.Tile8,row,accumulator))

def update_matches(+context: U32,+key: RowLane,+accumulator: Rows.Rows<Tile.Tile8>,+input: Tile.Tile8,+weights: Rows.Rows<F32>) ->
  {accumulator_at(key,update(context,accumulator,input,weights)) ==
    Fp32.multiply_add(accumulator_at(key,accumulator),input_at(key,input),weight_at(key,weights)) : F32}:
  RowLane{+row,+lane} = key
  %Equal.sym(Tile.Tile8,
    Accumulator.row(~Tile.Tile8,row,Gemm.update(accumulator,input,weights)),
    Tile.update(Accumulator.row(~Tile.Tile8,row,accumulator),input,Accumulator.row(~F32,row,weights)),
    Accumulator.row_update(row,accumulator,input,weights)) :
    {Tile.element_at(lane,_) == Fp32.multiply_add(Tile.element_at(lane,Accumulator.row(~Tile.Tile8,row,accumulator)),
      Tile.element_at(lane,input),Accumulator.row(~F32,row,weights)) : F32}
  Accumulator.update_at(lane,Accumulator.row(~Tile.Tile8,row,accumulator),input,Accumulator.row(~F32,row,weights))

def run(~upper: Bool,count: Nat,state: State()) -> State():
  Reduction.run(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~Rows.Rows<F32>,~U32,~Rows.Rows<Tile.Tile8>,
    ~ReductionLaws.observe_read(~Tree.Tree<Tile.Tile8>,~Tile.Tile8,~U32,~input_tile),
    ~ReductionLaws.observe_read(~Tree.Tree<Tile.Tile8>,~Rows.Rows<F32>,~U32,~weights_half(~upper)),~update,count,state)

def ordered_dot(~upper: Bool,key: RowLane,count: Nat,state: State()) -> F32:
  ReductionLaws.ordered_dot(~Tile.Tile8,~Tile.Tile8,~Tile.Tile8,~Rows.Rows<F32>,~U32,~Rows.Rows<Tile.Tile8>,~RowLane,
    ~input_tile,~weights_half(~upper),~input_at,~weight_at,~accumulator_at,key,count,state)

def run_matches_ordered_dot(~upper: Bool,+key: RowLane,+count: Nat,+state: State()) ->
  {accumulator_at(key,Reduction.accumulator(~&2,~Tree.Tree<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~U32,~Rows.Rows<Tile.Tile8>,run(~upper,count,state))) ==
    ordered_dot(~upper,key,count,state) : F32}:
  ReductionLaws.model_loop_matches_ordered_dot(~Tile.Tile8,~Tile.Tile8,~Tile.Tile8,~Rows.Rows<F32>,~U32,~Rows.Rows<Tile.Tile8>,~RowLane,
    ~input_tile,~weights_half(~upper),~input_at,~weight_at,~accumulator_at,~update,~update_matches,key,count,state)

def initial(input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<Tile.Tile8>,input_index: U32,weight_index: U32,base: U32) -> State():
  Reduction.State{input,weights,input_index,weight_index,base,Rows.filled(~Tile.Tile8,Tile.filled(0.0))}

def selected(upper: Bool,count: Nat,state: State()) -> State():
  match upper:
    case False{}: run(~False{},count,state)
    case True{}: run(~True{},count,state)
