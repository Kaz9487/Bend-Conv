# Coverage of the real fresh allocation and every block of Weights.run.
# The caller supplies only existing guards and an in-range logical coordinate.
import Base
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../../lib/tensor.bend as Tensors
import ../../lib/kernel_weights.bend as Weights
import ../../lib/kernel_tile.bend as Tile
import ../../lib/math_sequence.bend as Sequence
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/kernel_block_geometry.bend as Geometry
import ../../lib/proofs/kernel_weights_refinement.bend as WeightsLaws
import ../../lib/proofs/kernel_weights_readback.bend as WeightsReadback
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./reduction_coordinate_bounds.bend as Coordinates
import ./weight_reorder_bounds.bend as WeightBounds
import ./weight_reorder_coverage.bend as ReorderCoverage
import ./tile_array_projection.bend as ArrayProjection

def Goal(+problem: ConvSpec.Problem,+input: Tree.Tree<F32>,+lane: Sequence.Index(8n),+block: Nat,+tap: Nat) -> Data:
  +rows = U32.from_nat(Guards.output_channel_count(problem))
  +inner = Coordinates.native_inner(problem)
  {Tile.element_at(lane,Tree.read(Tile.Tile8,WeightsLaws.run(rows,inner,input),U32.from_nat(Nat.add(Nat.mul(block,Coordinates.inner(problem)),tap)))) ==
    Tile.element_at(lane,WeightsReadback.sample(input,Weights.Context{rows,inner,(U32.from_nat(block) * 8 : U32),(U32.from_nat(block) * inner : U32)},U32.from_nat(tap))) : F32}

def allocated_readback(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+supported: Support.FastSupported<limits,problem>,
  +input: Tree.Tree<F32>,+lane: Sequence.Index(8n),+block: Nat,+tap: Nat,
  block_inside: {Nat.is_lt(block,Geometry.blocks(Guards.output_channel_count(problem))) == True{} : Bool},
  tap_inside: {Nat.is_lt(tap,Coordinates.inner(problem)) == True{} : Bool},ready: WeightBounds.Ready(problem)) -> Goal(problem,input,lane,block,tap):
  Shape.Both{+depth_ok,+room} = ready
  +rows = U32.from_nat(Guards.output_channel_count(problem))
  +inner = Coordinates.native_inner(problem)
  +depth = Tensors.depth_for(WeightBounds.count(problem))
  +blocks_exact = WeightBounds.blocks_exact(limits,problem,supported)
  +inner_exact = Coordinates.inner_exact(limits,problem,valid,Padded.scalar(limits,problem,supported))
  +query = {ReorderCoverage.Query{rows,inner,depth,input,lane,block,tap,depth_ok,
    %Equal.sym(Nat,U32.to_nat(WeightBounds.blocks(problem)),Geometry.blocks(Guards.output_channel_count(problem)),blocks_exact) :
      {Nat.is_le(Nat.mul(_,U32.to_nat(inner)),U32.to_nat(U32.shln(1,depth))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(inner),Coordinates.inner(problem),inner_exact) :
      {Nat.is_le(Nat.mul(Geometry.blocks(Guards.output_channel_count(problem)),_),U32.to_nat(U32.shln(1,depth))) == True{} : Bool}
    %WeightBounds.count_exact(limits,problem,valid,supported) : {Nat.is_le(_,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}
    room,
    %Equal.sym(Nat,U32.to_nat(WeightBounds.blocks(problem)),Geometry.blocks(Guards.output_channel_count(problem)),blocks_exact) : {Nat.is_lt(block,_) == True{} : Bool}
    block_inside,
    %Equal.sym(Nat,U32.to_nat(inner),Coordinates.inner(problem),inner_exact) : {Nat.is_lt(tap,_) == True{} : Bool}
    tap_inside} : ReorderCoverage.Query}
  %inner_exact :
    {Tile.element_at(lane,Tree.read(Tile.Tile8,WeightsLaws.run(rows,inner,input),U32.from_nat(Nat.add(Nat.mul(block,_),tap)))) == ReorderCoverage.wanted(query) : F32}
  ReorderCoverage.complete(query,Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0)),ArrayProjection.fresh_balanced(depth,Tile.filled(0.0),lane))

def run_readback(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: Guards.Valid<problem>,+supported: Support.FastSupported<limits,problem>,
  +input: Tree.Tree<F32>,+lane: Sequence.Index(8n),+block: Nat,+tap: Nat,
  block_inside: {Nat.is_lt(block,Geometry.blocks(Guards.output_channel_count(problem))) == True{} : Bool},
  tap_inside: {Nat.is_lt(tap,Coordinates.inner(problem)) == True{} : Bool}) -> Goal(problem,input,lane,block,tap):
  allocated_readback(limits,problem,valid,supported,input,lane,block,tap,block_inside,tap_inside,WeightBounds.allocated(limits,problem,valid,supported))
