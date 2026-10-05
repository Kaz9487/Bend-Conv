import Base
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./tree_indexing.bend as TreeIndex

# Proof bridge, not a hot-loop implementation. Native packing requires balanced
# power-of-two trees; structural Array matching copies blocks in this runtime.
# The frozen equations describe Bend semantics, not arbitrary native tree shapes.

law read:
  for path: TreeIndex.Path
  for a: Array<F32>
  Array<F32> & F32

def restore_left(r: Array<F32>, pair: Array<F32> & F32) -> Array<F32> & F32:
  (l,x) = pair
  (ANode{l,r},x)

def restore_right(l: Array<F32>, pair: Array<F32> & F32) -> Array<F32> & F32:
  (r,x) = pair
  (ANode{l,r},x)

def read(path,a):
  match path a:
    case TreeIndex.Here{} ALeaf{+x}:
      (ALeaf{x},x)
    case TreeIndex.Here{} ANode{l,r}:
      (ANode{l,r},0.0)
    case TreeIndex.Left{p} ALeaf{x}:
      (ALeaf{x},0.0)
    case TreeIndex.Right{p} ALeaf{x}:
      (ALeaf{x},0.0)
    case TreeIndex.Left{p} ANode{l,r}:
      restore_left(r,read(p,l))
    case TreeIndex.Right{p} ANode{l,r}:
      restore_right(l,read(p,r))

# Physical address semantics share this proof-only representation boundary.
law route:
  for +tree: Tree.Tree<F32>
  for +n: U32
  for +i: U32
  TreeIndex.Routes(tree)

def route(tree,n,i):
  match tree:
    case Tree.Leaf{value}: Unit{}
    case Tree.Node{+left,+right}:
      Bool.pick(Either<&2,&2,TreeIndex.Routes(left),TreeIndex.Routes(right)>,U32.is_lt(i,U32.shr(n)),
        Inl{route(left,U32.shr(n),i)},Inr{route(right,U32.shr(n),U32.sub(i,U32.shr(n)))})

def wrapped_route(+tree: Tree.Tree<F32>,i: U32) -> TreeIndex.Routes(tree):
  route(tree,Tree.capacity(F32,tree),U32.and(i,U32.sub(Tree.capacity(F32,tree),1)))
