import ../../lib/proofs/storage_shape.bend as Shape
import Base
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/natural_addition.bend as Addition
import ../specification.bend as ConvSpec
import ./tree_indexing.bend as TreeIndex
import ./native_array_model.bend as NativeArrayModel

law length_cat:
  for +xs: List<&2,F32>
  for +ys: List<&2,F32>
  {ConvSpec.length(TreeIndex.cat(xs,ys)) == Nat.add(ConvSpec.length(xs),ConvSpec.length(ys)) : Nat}

def length_cat(xs,ys):
  match xs:
    case Nil{}:
      {==}
    case h <> +tail:
      %length_cat(tail,ys) : {1n+ConvSpec.length(TreeIndex.cat(tail,ys)) == 1n+_ : Nat}
      {==}

law length:
  for +depth: Nat
  for +tree: Tree.Tree<F32>
  for balanced: Shape.Balanced(F32,depth,tree)
  {ConvSpec.length(TreeIndex.flatten(tree)) == Shape.extent(depth) : Nat}

def length(depth,tree,balanced):
  match depth tree:
    case 0n Tree.Leaf{x}:
      {==}
    case 0n Tree.Node{l,r}:
      match balanced:
    case 1n+d Tree.Leaf{x}:
      match balanced:
    case 1n+ +d Tree.Node{+l,+r}:
      Shape.Both{left,right} = balanced
      %Addition.double(Shape.extent(d)) : {ConvSpec.length(TreeIndex.cat(TreeIndex.flatten(l),TreeIndex.flatten(r))) == _ : Nat}
      %length(d,l,left) : {ConvSpec.length(TreeIndex.cat(TreeIndex.flatten(l),TreeIndex.flatten(r))) == Nat.add(_,
          Shape.extent(d)) : Nat}
      %length(d,r,right) : {ConvSpec.length(TreeIndex.cat(TreeIndex.flatten(l),
            TreeIndex.flatten(r))) == Nat.add(ConvSpec.length(TreeIndex.flatten(l)),_) : Nat}
      length_cat(TreeIndex.flatten(l),TreeIndex.flatten(r))
