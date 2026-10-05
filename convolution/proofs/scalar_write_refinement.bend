import ../../lib/proofs/storage_shape.bend as Shape
# The dense specification of one native Array.set, with an in-bounds index.
import Base
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./tree_indexing.bend as TreeIndex
import ./array_tree_refinement.bend as ArrayTreeRefinement
import ./native_array_model.bend as NativeArrayModel
import ./address_encoding.bend as AddressEncoding
import ../specification.bend as ConvSpec

def set(n: Nat,v: F32,xs: List<&2,F32>) -> List<&2,F32>:
  match n xs:
    case 0n Nil{}:
      []
    case 0n h <> tail:
      v <> tail
    case 1n+rest Nil{}:
      []
    case 1n+rest h <> tail:
      h <> set(rest,v,tail)

def replace(+tree: Tree.Tree<F32>,route: TreeIndex.Routes(tree),v: F32) -> Tree.Tree<F32>:
  match tree:
    case Tree.Leaf{x}:
      Tree.Leaf{v}
    case Tree.Node{+l,+r}:
      match route:
        case Inl{p}:
          Tree.Node{replace(l,p,v),r}
        case Inr{p}:
          Tree.Node{l,replace(r,p,v)}

law set_skip:
  for +xs: List<&2,F32>
  for +ys: List<&2,F32>
  for +i: Nat
  for +v: F32
  {set(Nat.add(ConvSpec.length(xs),i),v,TreeIndex.cat(xs,ys)) == TreeIndex.cat(xs,set(i,v,ys)) : List<&2,F32>}
def set_skip(xs,ys,i,v):
  match xs:
    case []:
      {==}
    case +h <> +tail:
      %set_skip(tail,ys,i,v) : {h <> set(Nat.add(ConvSpec.length(tail),i),v,TreeIndex.cat(tail,ys)) == h <> _ : List<&2,F32>}
      {==}

law replace_flatten_tail:
  for +tree: Tree.Tree<F32>
  for +route: TreeIndex.Routes(tree)
  for +v: F32
  for +tail: List<&2,F32>
  {TreeIndex.cat(TreeIndex.flatten(replace(tree,route,v)),tail) == set(TreeIndex.index(tree,route),v,
      TreeIndex.cat(TreeIndex.flatten(tree),tail)) : List<&2,F32>}
def replace_flatten_tail(tree,route,v,tail):
  match tree:
    case Tree.Leaf{x}:
      {==}
    case Tree.Node{+l,+r}:
      match route:
        case Inl{+p}:
          %ArrayTreeRefinement.cat_assoc(TreeIndex.flatten(replace(l,p,v)),TreeIndex.flatten(r),tail) : {_ == set(TreeIndex.index(l,p),v,
              TreeIndex.cat(TreeIndex.cat(TreeIndex.flatten(l),TreeIndex.flatten(r)),tail)) : List<&2,F32>}
          %ArrayTreeRefinement.cat_assoc(TreeIndex.flatten(l),TreeIndex.flatten(r),tail) : {TreeIndex.cat(TreeIndex.flatten(replace(l,p,
                  v)),TreeIndex.cat(TreeIndex.flatten(r),tail)) == set(TreeIndex.index(l,p),v,_) : List<&2,F32>}
          replace_flatten_tail(l,p,v,TreeIndex.cat(TreeIndex.flatten(r),tail))
        case Inr{+p}:
          %ArrayTreeRefinement.cat_assoc(TreeIndex.flatten(l),TreeIndex.flatten(replace(r,p,v)),
            tail) : {_ == set(Nat.add(ConvSpec.length(TreeIndex.flatten(l)),TreeIndex.index(r,p)),v,
              TreeIndex.cat(TreeIndex.cat(TreeIndex.flatten(l),TreeIndex.flatten(r)),tail)) : List<&2,F32>}
          %ArrayTreeRefinement.cat_assoc(TreeIndex.flatten(l),TreeIndex.flatten(r),tail) : {TreeIndex.cat(TreeIndex.flatten(l),
              TreeIndex.cat(TreeIndex.flatten(replace(r,p,v)),tail)) == set(Nat.add(ConvSpec.length(TreeIndex.flatten(l)),
                TreeIndex.index(r,p)),v,_) : List<&2,F32>}
          %Equal.sym(List<&2,F32>,set(Nat.add(ConvSpec.length(TreeIndex.flatten(l)),TreeIndex.index(r,p)),v,
              TreeIndex.cat(TreeIndex.flatten(l),TreeIndex.cat(TreeIndex.flatten(r),tail))),TreeIndex.cat(TreeIndex.flatten(l),
              set(TreeIndex.index(r,p),v,TreeIndex.cat(TreeIndex.flatten(r),tail))),set_skip(TreeIndex.flatten(l),
              TreeIndex.cat(TreeIndex.flatten(r),tail),TreeIndex.index(r,p),v)) : {TreeIndex.cat(TreeIndex.flatten(l),
              TreeIndex.cat(TreeIndex.flatten(replace(r,p,v)),tail)) == _ : List<&2,F32>}
          %replace_flatten_tail(r,p,v,tail) : {TreeIndex.cat(TreeIndex.flatten(l),TreeIndex.cat(TreeIndex.flatten(replace(r,p,v)),
                tail)) == TreeIndex.cat(TreeIndex.flatten(l),_) : List<&2,F32>}
          {==}

law replace_flatten:
  for +tree: Tree.Tree<F32>
  for +route: TreeIndex.Routes(tree)
  for +v: F32
  {TreeIndex.flatten(replace(tree,route,v)) == set(TreeIndex.index(tree,route),v,TreeIndex.flatten(tree)) : List<&2,F32>}
def replace_flatten(tree,route,v):
  %ArrayTreeRefinement.cat_nil(TreeIndex.flatten(replace(tree,route,v))) : {_ == set(TreeIndex.index(tree,route),v,
      TreeIndex.flatten(tree)) : List<&2,F32>}
  %ArrayTreeRefinement.cat_nil(TreeIndex.flatten(tree)) : {TreeIndex.cat(TreeIndex.flatten(replace(tree,route,v)),
      []) == set(TreeIndex.index(tree,route),v,_) : List<&2,F32>}
  replace_flatten_tail(tree,route,v,[])

def StoreGoal(+tree: Tree.Tree<F32>,+n: U32,+i: U32,+v: F32) -> Type:
  {Tree.store(F32,tree,n,i,v) == replace(tree,NativeArrayModel.route(tree,
        n,i),v) : Tree.Tree<F32>}

def store_node(+l: Tree.Tree<F32>,+r: Tree.Tree<F32>,+h: U32,+i: U32,+v: F32,+z: Bool,pl: StoreGoal(l,h,i,v),pr: StoreGoal(r,h,
    (i - h : U32),v)) -> {Bool.pick(Tree.Tree<F32>,z,Tree.Node{Tree.store(F32,l,h,i,v),r},
      Tree.Node{l,Tree.store(F32,r,h,(i - h : U32),v)}) == replace(Tree.Node{l,
      r},Bool.pick(Either<&2,&2,TreeIndex.Routes(l),TreeIndex.Routes(r)>,z,
      Inl{NativeArrayModel.route(l,h,i)},Inr{NativeArrayModel.route(r,h,(i - h : U32))}),
    v) : Tree.Tree<F32>}:
  match z:
    case True{}:
      %pl : {Tree.Node{Tree.store(F32,l,h,i,v),r} == Tree.Node{_,
          r} : Tree.Tree<F32>}
      {==}
    case False{}:
      %pr : {Tree.Node{l,Tree.store(F32,r,h,(i - h : U32),
              v)} == Tree.Node{l,_} : Tree.Tree<F32>}
      {==}

def store(+tree: Tree.Tree<F32>,+n: U32,+i: U32,+v: F32) -> StoreGoal(tree,n,i,v):
  match tree:
    case Tree.Leaf{x}:
      {==}
    case Tree.Node{+l,+r}:
      store_node(l,r,U32.shr(n),i,v,U32.is_lt(i,U32.shr(n)),store(l,U32.shr(n),i,v),store(r,U32.shr(n),(i - U32.shr(n) : U32),v))

law write_route:
  for +tree: Tree.Tree<F32>
  for +i: U32
  for +v: F32
  {Tree.write(F32,tree,i,v) == replace(tree,
      NativeArrayModel.wrapped_route(tree,i),v) : Tree.Tree<F32>}
def write_route(tree,i,v):
  store(tree,Tree.capacity(F32,tree),U32.and(i,U32.sub(Tree.capacity(F32,tree),1)),v)

law dense:
  for +depth: Nat
  for +tree: Tree.Tree<F32>
  for +i: U32
  for +v: F32
  for balanced: Shape.Balanced(F32,depth,tree)
  for depth_ok: {Nat.is_le(depth,30n) == True{} : Bool}
  for in_bounds: {U32.is_lt(i,Tree.capacity(F32,tree)) == True{} : Bool}
  {TreeIndex.flatten(Tree.write(F32,tree,i,v)) == set(U32.to_nat(i),v,
      TreeIndex.flatten(tree)) : List<&2,F32>}
def dense(depth,tree,i,v,balanced,depth_ok,in_bounds):
  %Equal.sym(Tree.Tree<F32>,Tree.write(F32,tree,i,v),replace(tree,
      NativeArrayModel.wrapped_route(tree,i),v),write_route(tree,i,v)) : {TreeIndex.flatten(_) == set(U32.to_nat(i),v,
      TreeIndex.flatten(tree)) : List<&2,F32>}
  %AddressEncoding.natural_address(depth,tree,i,balanced,depth_ok,
    in_bounds) : {TreeIndex.flatten(replace(tree,NativeArrayModel.wrapped_route(tree,i),v)) == set(_,
      v,TreeIndex.flatten(tree)) : List<&2,F32>}
  replace_flatten(tree,NativeArrayModel.wrapped_route(tree,i),v)
