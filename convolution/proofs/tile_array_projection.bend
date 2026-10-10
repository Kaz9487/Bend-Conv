import ../../lib/proofs/storage_shape.bend as Shape
# Concrete Fp32Tile adapter; all tree induction belongs to the generic mapping law.
import Base
import ../../lib/math_sequence.bend as Sequence
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/tree_mapping.bend as Mapping

def project(tree: Tree.Tree<Tile.Tile8>,lane: Sequence.Index(8n)) -> Tree.Tree<F32>:
  Mapping.map(~Tile.Tile8,~F32,~Sequence.Index(8n),~Tile.element_at,lane,tree)

def project_fresh(+depth: Nat,+value: Tile.Tile8,+lane: Sequence.Index(8n)) ->
  {project(Tree.fresh(Tile.Tile8,depth,value),lane) == Tree.fresh(F32,depth,Tile.element_at(lane,value)) : Tree.Tree<F32>}:
  Mapping.map_fresh(~Tile.Tile8,~F32,~Sequence.Index(8n),~Tile.element_at,lane,depth,value)

def read_projected(+tree: Tree.Tree<Tile.Tile8>,+lane: Sequence.Index(8n),+index: U32) ->
  {Tree.read(F32,project(tree,lane),index) == Tile.element_at(lane,Tree.read(Tile.Tile8,tree,index)) : F32}:
  Mapping.map_read(~Tile.Tile8,~F32,~Sequence.Index(8n),~Tile.element_at,lane,tree,index)

def write_projected(+tree: Tree.Tree<Tile.Tile8>,+lane: Sequence.Index(8n),+index: U32,+value: Tile.Tile8) ->
  {project(Tree.write(Tile.Tile8,tree,index,value),lane) == Tree.write(F32,project(tree,lane),index,Tile.element_at(lane,value)) : Tree.Tree<F32>}:
  Mapping.map_write(~Tile.Tile8,~F32,~Sequence.Index(8n),~Tile.element_at,lane,tree,index,value)

def balanced(+depth: Nat,+tree: Tree.Tree<Tile.Tile8>,+lane: Sequence.Index(8n),evidence: Shape.Balanced(Tile.Tile8,depth,tree)) -> Shape.Balanced(F32,depth,project(tree,lane)):
  Mapping.map_preserves_balance(~Tile.Tile8,~F32,~Sequence.Index(8n),~Tile.element_at,lane,depth,tree,evidence)

def capacity(+tree: Tree.Tree<Tile.Tile8>,+lane: Sequence.Index(8n)) -> {Tree.capacity(F32,project(tree,lane)) == Tree.capacity(Tile.Tile8,tree) : U32}:
  Mapping.map_preserves_capacity(~Tile.Tile8,~F32,~Sequence.Index(8n),~Tile.element_at,lane,tree)

def fresh_balanced(+depth: Nat,+value: Tile.Tile8,+lane: Sequence.Index(8n)) -> Shape.Balanced(F32,depth,project(Tree.fresh(Tile.Tile8,depth,value),lane)):
  balanced(depth,Tree.fresh(Tile.Tile8,depth,value),lane,Shape.fresh(Tile.Tile8,depth,value))
