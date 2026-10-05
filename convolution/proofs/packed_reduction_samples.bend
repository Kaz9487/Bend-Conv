import Base
import ../../lib/math_sequence.bend as Sequence
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/traversal_reduction_refinement.bend as ReductionLaws

type Context is Data:
  Context{input: Tree.Tree<Tile.Tile8>,weights: Tree.Tree<F32>,reduction_length: U32}

def select(lane: Sequence.Index(8n),values: Tile.Tile8) -> F32:
  Tile.element_at(lane,values)

def input(context: Context,lane: Sequence.Index(8n),tap: ReductionLaws.AddressPair) -> F32:
  Context{input,weights,reduction_length} = context
  ReductionLaws.AddressPair{input_index,weight_index} = tap
  select(lane,Tree.read(Tile.Tile8,input,input_index))

def weight(context: Context,row: Bool,tap: ReductionLaws.AddressPair) -> F32:
  Context{input,+weights,reduction_length} = context
  ReductionLaws.AddressPair{input_index,+weight_index} = tap
  Bool.pick(F32,row,Tree.read(F32,weights,(weight_index + 1 * reduction_length : U32)),Tree.read(F32,weights,weight_index))

