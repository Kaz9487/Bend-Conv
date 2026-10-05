import Base
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../specification.bend as ConvSpec

# Logical paths and dense indices over the common element-parameterized tree.
# Invalid-path behavior differs from native wrapping; the container is shared.

type Path is Data:
  Here{}
  Left{rest: Path}
  Right{rest: Path}

def cat(xs: List<&2,F32>, ys: List<&2,F32>) -> List<&2,F32>:
  List.append(&2,F32,xs,ys)

def flatten(tree: Tree.Tree<F32>) -> List<&2,F32>:
  match tree:
    case Tree.Leaf{x}:
      x <> Nil{}
    case Tree.Node{l,r}:
      cat(flatten(l),flatten(r))

def observe(a: Array<F32>) -> List<&2,F32>:
  match a:
    case ALeaf{x}:
      x <> Nil{}
    case ANode{l,r}:
      cat(observe(l),observe(r))

law Routes:
  for tree: Tree.Tree<F32>
  Data

def Routes(tree):
  match tree:
    case Tree.Leaf{x}:
      Unit
    case Tree.Node{l,r}:
      Either<&2,&2,Routes(l),Routes(r)>

law encode:
  for +tree: Tree.Tree<F32>
  for route: Routes(tree)
  Path

def encode(tree,route):
  match tree:
    case Tree.Leaf{x}:
      Here{}
    case Tree.Node{+l,+r}:
      match route:
        case Inl{p}:
          Left{encode(l,p)}
        case Inr{p}:
          Right{encode(r,p)}

law index:
  for +tree: Tree.Tree<F32>
  for route: Routes(tree)
  Nat

def index(tree,route):
  match tree:
    case Tree.Leaf{x}:
      0n
    case Tree.Node{+l,+r}:
      match route:
        case Inl{p}:
          index(l,p)
        case Inr{p}:
          Nat.add(ConvSpec.length(flatten(l)),index(r,p))

law value:
  for +tree: Tree.Tree<F32>
  for route: Routes(tree)
  F32

def value(tree,route):
  match tree:
    case Tree.Leaf{x}:
      x
    case Tree.Node{+l,+r}:
      match route:
        case Inl{p}:
          value(l,p)
        case Inr{p}:
          value(r,p)

# Malformed paths return zero; never truncate or wrap to an existing leaf.
def selected(path: Path, tree: Tree.Tree<F32>) -> F32:
  match path tree:
    case Here{} Tree.Leaf{x}:
      x
    case Here{} Tree.Node{l,r}:
      0.0
    case Left{p} Tree.Leaf{x}:
      0.0
    case Right{p} Tree.Leaf{x}:
      0.0
    case Left{p} Tree.Node{l,r}:
      selected(p,l)
    case Right{p} Tree.Node{l,r}:
      selected(p,r)
