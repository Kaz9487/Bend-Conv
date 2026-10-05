import ../../lib/proofs/storage_shape.bend as Shape
import Base
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./tree_indexing.bend as TreeIndex
import ../specification.bend as ConvSpec
import ./address_encoding.bend as AddressEncoding
import ./guarded_read_refinement.bend as GuardedReadLaws
import ./native_array_model.bend as NativeArrayModel

# Physical capacity, not logical tensor length: callers prove their own logical bounds.

law read:
  for +depth: Nat
  for +tree: Tree.Tree<F32>
  for +i: U32
  for balanced: Shape.Balanced(F32,depth,tree)
  for depth_ok: {Nat.is_le(depth,30n) == True{} : Bool}
  for in_bounds: {U32.is_lt(i,Tree.capacity(F32,tree)) == True{} : Bool}
  {Array.get(F32,Tree.pack(F32,tree),i) == (Tree.pack(F32,tree),ConvSpec.at(U32.to_nat(i),TreeIndex.flatten(tree))) : Array<F32> & F32}

def read(depth,tree,i,balanced,depth_ok,in_bounds):
  %AddressEncoding.natural_address(depth,tree,i,balanced,depth_ok,in_bounds) : {Array.get(F32,Tree.pack(F32,tree),
      i) == (Tree.pack(F32,tree),ConvSpec.at(_,TreeIndex.flatten(tree))) : Array<F32> & F32}
  GuardedReadLaws.native_dense(tree,i)

# The physical and dense observations now share the same parameterized tree.
# Only the address semantics differ; no representation conversion is needed.
law dense:
  for +depth: Nat
  for +tree: Tree.Tree<F32>
  for +index: U32
  for balanced: Shape.Balanced(F32,depth,tree)
  for depth_ok: {Nat.is_le(depth,30n) == True{} : Bool}
  for in_bounds: {U32.is_lt(index,Tree.capacity(F32,tree)) == True{} : Bool}
  {Tree.read(F32,tree,index) == ConvSpec.at(U32.to_nat(index),TreeIndex.flatten(tree)) : F32}

def dense(depth,tree,index,balanced,depth_ok,in_bounds):
  +actual = Tree.read(F32,tree,index)
  +expected = ConvSpec.at(U32.to_nat(index),TreeIndex.flatten(tree))
  Equal.cong(Array<F32> & F32,F32,pair => Pair.snd(Array<F32>,F32,pair),(Tree.pack(F32,tree),actual),(Tree.pack(F32,tree),expected),
    Equal.trans(Array<F32> & F32,(Tree.pack(F32,tree),actual),Array.get(F32,Tree.pack(F32,tree),index),(Tree.pack(F32,tree),expected),
      Equal.sym(Array<F32> & F32,Array.get(F32,Tree.pack(F32,tree),index),(Tree.pack(F32,tree),actual),StorageLaws.read(F32,tree,index)),
      read(depth,tree,index,balanced,depth_ok,in_bounds)))
