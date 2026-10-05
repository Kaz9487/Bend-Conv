import ../../lib/proofs/storage_shape.bend as Shape
import Base
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ./array_tree_structure.bend as ArrayTreeStructure
import ./native_array_model.bend as NativeArrayModel
import ./tree_indexing.bend as TreeIndex
import ../specification.bend as ConvSpec

law right:
  for +depth: Nat
  for +l: Tree.Tree<F32>
  for +r: Tree.Tree<F32>
  for +cap: U32
  for +i: U32
  for +j: U32
  for balanced: Shape.Balanced(F32,depth,l)
  for value: {U32.to_nat(i) == Nat.add(Shape.extent(depth),U32.to_nat(j)) : Nat}
  for child: {TreeIndex.index(r,NativeArrayModel.route(r,cap,j)) == U32.to_nat(j) : Nat}
  {Nat.add(ConvSpec.length(TreeIndex.flatten(l)),TreeIndex.index(r,NativeArrayModel.route(r,cap,j))) == U32.to_nat(i) : Nat}
def right(depth,l,r,cap,i,j,balanced,value,child):
  %Equal.sym(Nat,U32.to_nat(i),Nat.add(Shape.extent(depth),U32.to_nat(j)),value) : {Nat.add(ConvSpec.length(TreeIndex.flatten(l)),
      TreeIndex.index(r,NativeArrayModel.route(r,cap,j))) == _ : Nat}
  %ArrayTreeStructure.length(depth,l,balanced) : {Nat.add(ConvSpec.length(TreeIndex.flatten(l)),TreeIndex.index(r,NativeArrayModel.route(r,
          cap,j))) == Nat.add(_,U32.to_nat(j)) : Nat}
  %child : {Nat.add(ConvSpec.length(TreeIndex.flatten(l)),TreeIndex.index(r,NativeArrayModel.route(r,cap,
          j))) == Nat.add(ConvSpec.length(TreeIndex.flatten(l)),_) : Nat}
  {==}
