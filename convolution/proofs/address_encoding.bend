import ../../lib/proofs/storage_shape.bend as Shape
import Base
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./tree_indexing.bend as TreeIndex
import ./native_array_model.bend as NativeArrayModel
import ./address_index.bend as AddressIndex
import ./array_tree_structure.bend as ArrayTreeStructure
import ./word_capacity_bounds.bend as WordCapacityBounds

# Native addresses equal natural indices for balanced capacities <= 2^30.

law natural_address:
  for +depth: Nat
  for +tree: Tree.Tree<F32>
  for +i: U32
  for balanced: Shape.Balanced(F32,depth,tree)
  for depth_ok: {Nat.is_le(depth,30n) == True{} : Bool}
  for in_bounds: {U32.is_lt(i,Tree.capacity(F32,tree)) == True{} : Bool}
  {TreeIndex.index(tree,NativeArrayModel.wrapped_route(tree,i)) == U32.to_nat(i) : Nat}

# Connect the generic depth induction to the actual wrapped native address.

def NativeGoal(+cap: U32,+tree: Tree.Tree<F32>,+i: U32) -> Type:
  {U32.is_lt(i,cap) == True{} : Bool} -> {TreeIndex.index(tree,NativeArrayModel.route(tree,cap,i)) == U32.to_nat(i) : Nat}
law raw:
  for +depth: Nat
  for +tree: Tree.Tree<F32>
  for +i: U32
  for balanced: Shape.Balanced(F32,depth,tree)
  for limit: {Nat.is_le(depth,30n) == True{} : Bool}
  NativeGoal(Tree.capacity(F32,tree),tree,i)
def raw(depth,tree,i,balanced,limit):
  +balanced = balanced
  %Equal.sym(U32,Tree.capacity(F32,tree),U32.shln(1,depth),Shape.capacity(F32,depth,tree,balanced)) : NativeGoal(_,tree,i)
  bound => AddressIndex.by_depth(depth,tree,i,balanced,limit,bound)

def natural_address(depth,tree,i,balanced,depth_ok,in_bounds):
  +balanced = balanced
  +depth_ok = depth_ok
  +in_bounds = in_bounds
  %Equal.sym(U32,U32.and(i,U32.sub(Tree.capacity(F32,tree),1)),i,WordCapacityBounds.no_wrap(depth,tree,i,balanced,
      depth_ok)(in_bounds)) : {TreeIndex.index(tree,NativeArrayModel.route(tree,Tree.capacity(F32,tree),_)) == U32.to_nat(i) : Nat}
  raw(depth,tree,i,balanced,depth_ok)(in_bounds)
