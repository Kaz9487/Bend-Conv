import Base
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../specification.bend as ConvSpec
import ./tree_indexing.bend as TreeIndex
import ./native_array_model.bend as NativeArrayModel

law packed_layout:
  for +tree: Tree.Tree<F32>
  {TreeIndex.observe(Tree.pack(F32,tree)) == TreeIndex.flatten(tree) : List<&2,F32>}

# Includes returned storage, not merely the selected value.
law read_path:
  for +path: TreeIndex.Path
  for +tree: Tree.Tree<F32>
  {NativeArrayModel.read(path,Tree.pack(F32,tree)) == (Tree.pack(F32,tree),TreeIndex.selected(path,tree)) : Array<F32> & F32}

# A typed route can only point at an actual leaf. The index is an unbounded Nat.
law read_dense:
  for +tree: Tree.Tree<F32>
  for +route: TreeIndex.Routes(tree)
  {NativeArrayModel.read(TreeIndex.encode(tree,route),Tree.pack(F32,tree)) == (Tree.pack(F32,tree),ConvSpec.at(TreeIndex.index(tree,
          route),TreeIndex.flatten(tree))) : Array<F32> & F32}

def packed_layout(tree):
  match tree:
    case Tree.Leaf{x}:
      {==}
    case Tree.Node{+l,+r}:
      %packed_layout(l) : {TreeIndex.cat(TreeIndex.observe(Tree.pack(F32,l)),
          TreeIndex.observe(Tree.pack(F32,r))) == TreeIndex.cat(_,TreeIndex.flatten(r)) : List<&2,F32>}
      %packed_layout(r) : {TreeIndex.cat(TreeIndex.observe(Tree.pack(F32,l)),
          TreeIndex.observe(Tree.pack(F32,r))) == TreeIndex.cat(TreeIndex.observe(Tree.pack(F32,l)),_) : List<&2,F32>}
      {==}

def read_path(path,tree):
  match path tree:
    case TreeIndex.Here{} Tree.Leaf{x}:
      {==}
    case TreeIndex.Here{} Tree.Node{l,r}:
      {==}
    case TreeIndex.Left{p} Tree.Leaf{x}:
      {==}
    case TreeIndex.Right{p} Tree.Leaf{x}:
      {==}
    case TreeIndex.Left{+p} Tree.Node{+l,+r}:
      %read_path(p,l) : {NativeArrayModel.restore_left(Tree.pack(F32,r),NativeArrayModel.read(p,
            Tree.pack(F32,l))) == NativeArrayModel.restore_left(Tree.pack(F32,r),_) : Array<F32> & F32}
      {==}
    case TreeIndex.Right{+p} Tree.Node{+l,+r}:
      %read_path(p,r) : {NativeArrayModel.restore_right(Tree.pack(F32,l),NativeArrayModel.read(p,
            Tree.pack(F32,r))) == NativeArrayModel.restore_right(Tree.pack(F32,l),_) : Array<F32> & F32}
      {==}

law cat_assoc:
  for +xs: List<&2,F32>
  for +ys: List<&2,F32>
  for +zs: List<&2,F32>
  {TreeIndex.cat(xs,TreeIndex.cat(ys,zs)) == TreeIndex.cat(TreeIndex.cat(xs,ys),zs) : List<&2,F32>}

def cat_assoc(xs,ys,zs):
  match xs:
    case Nil{}:
      {==}
    case h <> +tail:
      %cat_assoc(tail,ys,zs) : {h <> TreeIndex.cat(tail,TreeIndex.cat(ys,zs)) == h <> _ : List<&2,F32>}
      {==}

law at_skip:
  for +xs: List<&2,F32>
  for +ys: List<&2,F32>
  for +i: Nat
  {ConvSpec.at(i,ys) == ConvSpec.at(Nat.add(ConvSpec.length(xs),i),TreeIndex.cat(xs,ys)) : F32}

def at_skip(xs,ys,i):
  match xs:
    case Nil{}:
      {==}
    case h <> tail:
      at_skip(tail,ys,i)

law logical_lookup:
  for +tree: Tree.Tree<F32>
  for +route: TreeIndex.Routes(tree)
  for +tail: List<&2,F32>
  {TreeIndex.value(tree,route) == ConvSpec.at(TreeIndex.index(tree,route),TreeIndex.cat(TreeIndex.flatten(tree),tail)) : F32}

def logical_lookup(tree,route,tail):
  match tree:
    case Tree.Leaf{x}:
      {==}
    case Tree.Node{+l,+r}:
      match route:
        case Inl{+p}:
          %cat_assoc(TreeIndex.flatten(l),TreeIndex.flatten(r),tail) : {TreeIndex.value(l,p) == ConvSpec.at(TreeIndex.index(l,p),_) : F32}
          logical_lookup(l,p,TreeIndex.cat(TreeIndex.flatten(r),tail))
        case Inr{+p}:
          %cat_assoc(TreeIndex.flatten(l),TreeIndex.flatten(r),tail) : {TreeIndex.value(r,
              p) == ConvSpec.at(Nat.add(ConvSpec.length(TreeIndex.flatten(l)),TreeIndex.index(r,p)),_) : F32}
          %at_skip(TreeIndex.flatten(l),TreeIndex.cat(TreeIndex.flatten(r),tail),TreeIndex.index(r,p)) : {TreeIndex.value(r,p) == _ : F32}
          logical_lookup(r,p,tail)

law encoded_value:
  for +tree: Tree.Tree<F32>
  for +route: TreeIndex.Routes(tree)
  {TreeIndex.selected(TreeIndex.encode(tree,route),tree) == TreeIndex.value(tree,route) : F32}

def encoded_value(tree,route):
  match tree:
    case Tree.Leaf{x}:
      {==}
    case Tree.Node{+l,+r}:
      match route:
        case Inl{p}:
          encoded_value(l,p)
        case Inr{p}:
          encoded_value(r,p)

law cat_nil:
  for +xs: List<&2,F32>
  {TreeIndex.cat(xs,Nil{}) == xs : List<&2,F32>}

def cat_nil(xs):
  match xs:
    case Nil{}:
      {==}
    case h <> +tail:
      %cat_nil(tail) : {h <> TreeIndex.cat(tail,Nil{}) == h <> _ : List<&2,F32>}
      {==}

def read_dense(tree,route):
  %cat_nil(TreeIndex.flatten(tree)) : {NativeArrayModel.read(TreeIndex.encode(tree,route),Tree.pack(F32,tree)) == (Tree.pack(F32,tree),
      ConvSpec.at(TreeIndex.index(tree,route),_)) : Array<F32> & F32}
  %logical_lookup(tree,route,Nil{}) : {NativeArrayModel.read(TreeIndex.encode(tree,route),Tree.pack(F32,tree)) == (Tree.pack(F32,tree),
      _) : Array<F32> & F32}
  %encoded_value(tree,route) : {NativeArrayModel.read(TreeIndex.encode(tree,route),Tree.pack(F32,tree)) == (Tree.pack(F32,tree),
      _) : Array<F32> & F32}
  read_path(TreeIndex.encode(tree,route),tree)
