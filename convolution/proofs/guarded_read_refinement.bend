import Base
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ./tree_indexing.bend as TreeIndex
import ./native_array_model.bend as NativeArrayModel
import ../specification.bend as ConvSpec
import ./array_tree_refinement.bend as ArrayTreeRefinement

# Exact decoded dense address; NOT yet a theorem that it equals U32.to_nat(i).
law native_dense:
  for +tree: Tree.Tree<F32>
  for +i: U32
  {Array.get(F32,Tree.pack(F32,tree),i) == (Tree.pack(F32,tree),ConvSpec.at(TreeIndex.index(tree,NativeArrayModel.wrapped_route(tree,
          i)),TreeIndex.flatten(tree))) : Array<F32> & F32}

law bounded:
  for +tree: Tree.Tree<F32>
  for +i: U32
  for +logical: U32
  {Storage.read_or(F32,0.0,Tree.pack(F32,tree),i,logical) == Bool.pick(Array<F32> & F32,U32.is_lt(i,logical) && U32.is_le(logical,
        Tree.capacity(F32,tree)),(Tree.pack(F32,tree),ConvSpec.at(TreeIndex.index(tree,NativeArrayModel.wrapped_route(tree,i)),
          TreeIndex.flatten(tree))),(Tree.pack(F32,tree),0.0)) : Array<F32> & F32}

law size:
  for +tree: Tree.Tree<F32>
  {Array.size(F32,Tree.pack(F32,tree)) == (Tree.pack(F32,tree),Tree.capacity(F32,tree)) : Array<F32> & U32}

def size(tree):
  StorageLaws.size(F32,tree)

# Base passes the branch z (i < n/2) computed by its caller; the route model
# picks the same branch, so the node lemma holds for any z.
law node:
  for +l: Tree.Tree<F32>
  for +r: Tree.Tree<F32>
  for +n: U32
  for +i: U32
  for +z: Bool
  for pl: {Array.get.go(F32,Tree.pack(F32,l),U32.shr(n),i,U32.is_lt(i,U32.shr(U32.shr(n)))) ==
      (Tree.pack(F32,l),TreeIndex.value(l,NativeArrayModel.route(l,U32.shr(n),i))) : Array<F32> & F32}
  for pr: {Array.get.go(F32,Tree.pack(F32,r),U32.shr(n),U32.sub(i,U32.shr(n)),U32.is_lt(U32.sub(i,U32.shr(n)),U32.shr(U32.shr(n)))) ==
      (Tree.pack(F32,r),TreeIndex.value(r,NativeArrayModel.route(r,U32.shr(n),U32.sub(i,U32.shr(n))))) : Array<F32> & F32}
  {Array.get.go(F32,Tree.pack(F32,Tree.Node{l,r}),n,i,z) == (Tree.pack(F32,Tree.Node{l,r}),
      TreeIndex.value(Tree.Node{l,r},Bool.pick(Either<&2,&2,TreeIndex.Routes(l),TreeIndex.Routes(r)>,z,
          Inl{NativeArrayModel.route(l,U32.shr(n),i)},Inr{NativeArrayModel.route(r,U32.shr(n),U32.sub(i,U32.shr(n)))}))) : Array<F32> & F32}

def node(l,r,n,i,z,pl,pr):
  match z:
    case True{}:
      %pl : {Array.swap.lo(F32,Tree.pack(F32,r),Array.get.go(F32,Tree.pack(F32,l),U32.shr(n),i,U32.is_lt(i,U32.shr(U32.shr(n))))) ==
          Array.swap.lo(F32,Tree.pack(F32,r),_) : Array<F32> & F32}
      {==}
    case False{}:
      %pr : {Array.swap.hi(F32,Tree.pack(F32,l),Array.get.go(F32,Tree.pack(F32,r),U32.shr(n),U32.sub(i,U32.shr(n)),
          U32.is_lt(U32.sub(i,U32.shr(n)),U32.shr(U32.shr(n))))) == Array.swap.hi(F32,Tree.pack(F32,l),_) : Array<F32> & F32}
      {==}

law go:
  for +tree: Tree.Tree<F32>
  for +n: U32
  for +i: U32
  {Array.get.go(F32,Tree.pack(F32,tree),n,i,U32.is_lt(i,U32.shr(n))) == (Tree.pack(F32,tree),TreeIndex.value(tree,NativeArrayModel.route(tree,n,
          i))) : Array<F32> & F32}

def go(tree,n,i):
  match tree:
    case Tree.Leaf{x}:
      {==}
    case Tree.Node{+l,+r}:
      node(l,r,n,i,U32.is_lt(i,U32.shr(n)),go(l,U32.shr(n),i),go(r,U32.shr(n),U32.sub(i,U32.shr(n))))

law raw:
  for +tree: Tree.Tree<F32>
  for +i: U32
  {Array.get(F32,Tree.pack(F32,tree),i) == (Tree.pack(F32,tree),TreeIndex.value(tree,NativeArrayModel.wrapped_route(tree,
          i))) : Array<F32> & F32}

def raw(tree,i):
  %go(tree,Tree.capacity(F32,tree),U32.and(i,U32.sub(Tree.capacity(F32,tree),1))) : {Array.get.at(F32,i,Array.size(F32,
        Tree.pack(F32,tree))) == _ : Array<F32> & F32}
  %size(tree) : {Array.get.at(F32,i,Array.size(F32,Tree.pack(F32,tree))) == Array.get.at(F32,i,_) : Array<F32> & F32}
  {==}

def native_dense(tree,i):
  %ArrayTreeRefinement.cat_nil(TreeIndex.flatten(tree)) : {Array.get(F32,Tree.pack(F32,tree),i) == (Tree.pack(F32,tree),
      ConvSpec.at(TreeIndex.index(tree,NativeArrayModel.wrapped_route(tree,i)),_)) : Array<F32> & F32}
  %ArrayTreeRefinement.logical_lookup(tree,NativeArrayModel.wrapped_route(tree,i),Nil{}) : {Array.get(F32,Tree.pack(F32,tree),
      i) == (Tree.pack(F32,tree),_) : Array<F32> & F32}
  raw(tree,i)

law guard:
  for +ok: Bool
  for +tree: Tree.Tree<F32>
  for +i: U32
  {Storage.read_when(F32,0.0,ok,Tree.pack(F32,tree),i) == Bool.pick(Array<F32> & F32,ok,(Tree.pack(F32,tree),
        ConvSpec.at(TreeIndex.index(tree,NativeArrayModel.wrapped_route(tree,i)),TreeIndex.flatten(tree))),(Tree.pack(F32,tree),
        0.0)) : Array<F32> & F32}

def guard(ok,tree,i):
  match ok:
    case False{}:
      {==}
    case True{}:
      native_dense(tree,i)

def bounded(tree,i,logical):
  %guard(U32.is_lt(i,logical) && U32.is_le(logical,Tree.capacity(F32,tree)),tree,i) : {Storage.read_sized(F32,0.0,i,logical,Array.size(F32,
        Tree.pack(F32,tree))) == _ : Array<F32> & F32}
  %size(tree) : {Storage.read_sized(F32,0.0,i,logical,Array.size(F32,Tree.pack(F32,tree))) == Storage.read_sized(F32,0.0,i,logical,_) : Array<F32> & F32}
  {==}
