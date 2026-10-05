import ../../lib/proofs/storage_shape.bend as Shape
# The generic bit-width theorem owns masking; this module connects tree capacity.
import Base
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/word_power_of_two.bend as Power
import ./array_tree_structure.bend as ArrayTreeStructure
import ./tree_indexing.bend as TreeIndex
import ./native_array_model.bend as NativeArrayModel

def Bound(+capacity: U32,+index: U32) -> Type:
  {U32.is_lt(index,capacity) == True{} : Bool} -> {U32.and(index,U32.sub(capacity,1)) == index : U32}

law by_depth:
  for +depth: Nat
  for +index: U32
  for limit: {Nat.is_le(depth,30n) == True{} : Bool}
  Bound(U32.shln(1,depth),index)

def by_depth(depth,index,limit):
  Power.native_mask_preserves_bounded(depth,index)

law no_wrap:
  for +depth: Nat
  for +tree: Tree.Tree<F32>
  for +index: U32
  for balanced: Shape.Balanced(F32,depth,tree)
  for limit: {Nat.is_le(depth,30n) == True{} : Bool}
  Bound(Tree.capacity(F32,tree),index)

def no_wrap(depth,tree,index,balanced,limit):
  %Equal.sym(U32,Tree.capacity(F32,tree),U32.shln(1,depth),Shape.capacity(F32,depth,tree,balanced)) : Bound(_,index)
  by_depth(depth,index,limit)
