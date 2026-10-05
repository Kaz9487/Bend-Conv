import Base
import ../storage_buffer.bend as Storage
import ./storage_tree_model.bend as Tree
import ./storage_witness.bend as Witness
import ./loop_projection.bend as Projection
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_encoding.bend as WordEncoding
import ./u32_comparison.bend as Comparison
import ./u32_division.bend as U32Division

# The tree model covers every native Array owner, including capacities larger
# than its logical sequence. These equalities allocate no runtime conversion.
def reflect_packed(-Element: Data,+tree: Tree.Tree<Element>) ->
  {Tree.reflect(Element,Tree.pack(Element,tree)) == tree : Tree.Tree<Element>}:
  match tree:
    case Tree.Leaf{value}: {==}
    case Tree.Node{+left,+right}:
      %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,left)),left,reflect_packed(Element,left)) :
        {Tree.Node{_,Tree.reflect(Element,Tree.pack(Element,right))} == Tree.Node{left,right} : Tree.Tree<Element>}
      %Equal.sym(Tree.Tree<Element>,Tree.reflect(Element,Tree.pack(Element,right)),right,reflect_packed(Element,right)) :
        {Tree.Node{left,_} == Tree.Node{left,right} : Tree.Tree<Element>}
      {==}

def pack_reflected(-Element: Data,storage: Array<Element>) ->
  {Tree.pack(Element,Tree.reflect(Element,storage)) == storage : Array<Element>}:
  match storage:
    case ALeaf{value}: {==}
    case ANode{left,right}:
      %pack_reflected(Element,left) :
        {ANode{Tree.pack(Element,Tree.reflect(Element,left)),Tree.pack(Element,Tree.reflect(Element,right))} == ANode{_,right} : Array<Element>}
      %pack_reflected(Element,right) :
        {ANode{Tree.pack(Element,Tree.reflect(Element,left)),Tree.pack(Element,Tree.reflect(Element,right))} == ANode{Tree.pack(Element,Tree.reflect(Element,left)),_} : Array<Element>}
      {==}

law fresh:
  for -T: Data
  for +depth: Nat
  for +value: T
  {Array.new(T,depth,value) == Tree.pack(T,Tree.fresh(T,depth,value)) : Array<T>}
law clone:
  for -T: Data
  for +tree: Tree.Tree<T>
  {Array.clone(T,Tree.pack(T,tree)) == (Tree.pack(T,tree),Tree.pack(T,tree)) : Array<T> & Array<T>}
law read:
  for -T: Data
  for +tree: Tree.Tree<T>
  for +i: U32
  {Array.get(T,Tree.pack(T,tree),i) == (Tree.pack(T,tree),Tree.read(T,tree,i)) : Array<T> & T}
law write:
  for -T: Data
  for +tree: Tree.Tree<T>
  for +i: U32
  for +value: T
  {Array.set(T,Tree.pack(T,tree),i,value) == Tree.pack(T,Tree.write(T,tree,i,value)) : Array<T>}

def fresh(T,depth,value):
  Witness.fresh_model(T,depth,value)

def clone(T,tree):
  match tree:
    case Tree.Leaf{v}:
      {==}
    case Tree.Node{+l,+r}:
      %Equal.sym(Array<T> & Array<T>,Array.clone(T,Tree.pack(T,l)),(Tree.pack(T,l),Tree.pack(T,l)),
        clone(T,l)) : {Array.clone.node(T,_,Array.clone(T,Tree.pack(T,r))) == (Tree.pack(T,
            Tree.Node{l,r}),Tree.pack(T,Tree.Node{l,r})) : Array<T> & Array<T>}
      %Equal.sym(Array<T> & Array<T>,Array.clone(T,Tree.pack(T,r)),(Tree.pack(T,r),Tree.pack(T,r)),
        clone(T,r)) : {Array.clone.node(T,(Tree.pack(T,l),Tree.pack(T,l)),_) == (Tree.pack(T,
            Tree.Node{l,r}),Tree.pack(T,Tree.Node{l,r})) : Array<T> & Array<T>}
      {==}

law size:
  for -T: Data
  for +tree: Tree.Tree<T>
  {Array.size(T,Tree.pack(T,tree)) == (Tree.pack(T,tree),Tree.capacity(T,tree)) : Array<T> & U32}
def size(T,tree):
  match tree:
    case Tree.Leaf{v}:
      {==}
    case Tree.Node{+l,+r}:
      %Equal.sym(Array<T> & U32,Array.size(T,Tree.pack(T,l)),(Tree.pack(T,l),Tree.capacity(T,l)),size(T,
          l)) : {Array.size.node(T,Tree.pack(T,r),_) == (Tree.pack(T,Tree.Node{l,r}),U32.shl(Tree.capacity(T,
              l))) : Array<T> & U32}
      {==}

# Base passes the branch z that the caller computed as i < n/2; each child
# receives its own comparison against its half. The tree model picks the same
# branch, so every lemma holds for any z and the recursion supplies its value.
law get_node:
  for -T: Data
  for +l: Tree.Tree<T>
  for +r: Tree.Tree<T>
  for +n: U32
  for +i: U32
  for +z: Bool
  for pl: {Array.get.go(T,Tree.pack(T,l),U32.shr(n),i,U32.is_lt(i,U32.shr(U32.shr(n)))) ==
      (Tree.pack(T,l),Tree.fetch(T,l,U32.shr(n),i)) : Array<T> & T}
  for pr: {Array.get.go(T,Tree.pack(T,r),U32.shr(n),U32.sub(i,U32.shr(n)),U32.is_lt(U32.sub(i,U32.shr(n)),U32.shr(U32.shr(n)))) ==
      (Tree.pack(T,r),Tree.fetch(T,r,U32.shr(n),U32.sub(i,U32.shr(n)))) : Array<T> & T}
  {Array.get.go(T,Tree.pack(T,Tree.Node{l,r}),n,i,z) == (Tree.pack(T,Tree.Node{l,r}),Bool.pick(T,z,
        Tree.fetch(T,l,U32.shr(n),i),Tree.fetch(T,r,U32.shr(n),U32.sub(i,U32.shr(n))))) : Array<T> & T}
def get_node(T,l,r,n,i,z,pl,pr):
  match z:
    case True{}:
      %pl : {Array.swap.lo(T,Tree.pack(T,r),Array.get.go(T,Tree.pack(T,l),U32.shr(n),i,U32.is_lt(i,U32.shr(U32.shr(n))))) ==
          Array.swap.lo(T,Tree.pack(T,r),_) : Array<T> & T}
      {==}
    case False{}:
      %pr : {Array.swap.hi(T,Tree.pack(T,l),Array.get.go(T,Tree.pack(T,r),U32.shr(n),U32.sub(i,U32.shr(n)),
          U32.is_lt(U32.sub(i,U32.shr(n)),U32.shr(U32.shr(n))))) == Array.swap.hi(T,Tree.pack(T,l),_) : Array<T> & T}
      {==}

law get_go:
  for -T: Data
  for +tree: Tree.Tree<T>
  for +n: U32
  for +i: U32
  {Array.get.go(T,Tree.pack(T,tree),n,i,U32.is_lt(i,U32.shr(n))) == (Tree.pack(T,tree),Tree.fetch(T,tree,n,i)) : Array<T> & T}
def get_go(T,tree,n,i):
  match tree:
    case Tree.Leaf{v}:
      {==}
    case Tree.Node{+l,+r}:
      get_node(T,l,r,n,i,U32.is_lt(i,U32.shr(n)),get_go(T,l,U32.shr(n),i),get_go(T,r,U32.shr(n),U32.sub(i,U32.shr(n))))

def read(T,tree,i):
  %get_go(T,tree,Tree.capacity(T,tree),U32.and(i,U32.sub(Tree.capacity(T,tree),1))) : {Array.get.at(T,i,Array.size(T,
        Tree.pack(T,tree))) == _ : Array<T> & T}
  %size(T,tree) : {Array.get.at(T,i,Array.size(T,Tree.pack(T,tree))) == Array.get.at(T,i,_) : Array<T> & T}
  {==}

law swap_node:
  for -T: Data
  for +l: Tree.Tree<T>
  for +r: Tree.Tree<T>
  for +n: U32
  for +i: U32
  for +v: T
  for +z: Bool
  for pl: {Array.swap.go(T,Tree.pack(T,l),U32.shr(n),i,v,U32.is_lt(i,U32.shr(U32.shr(n)))) ==
      (Tree.pack(T,Tree.store(T,l,U32.shr(n),i,v)),Tree.fetch(T,l,U32.shr(n),i)) : Array<T> & T}
  for pr: {Array.swap.go(T,Tree.pack(T,r),U32.shr(n),U32.sub(i,U32.shr(n)),v,U32.is_lt(U32.sub(i,U32.shr(n)),U32.shr(U32.shr(n)))) ==
      (Tree.pack(T,Tree.store(T,r,U32.shr(n),U32.sub(i,U32.shr(n)),v)),Tree.fetch(T,r,U32.shr(n),U32.sub(i,U32.shr(n)))) : Array<T> & T}
  {Array.swap.go(T,Tree.pack(T,Tree.Node{l,r}),n,i,v,z) == (Tree.pack(T,Bool.pick(Tree.Tree<T>,z,
          Tree.Node{Tree.store(T,l,U32.shr(n),i,v),r},Tree.Node{l,Tree.store(T,r,U32.shr(n),U32.sub(i,U32.shr(n)),v)})),
        Bool.pick(T,z,Tree.fetch(T,l,U32.shr(n),i),Tree.fetch(T,r,U32.shr(n),U32.sub(i,U32.shr(n))))) : Array<T> & T}
def swap_node(T,l,r,n,i,v,z,pl,pr):
  match z:
    case True{}:
      %pl : {Array.swap.lo(T,Tree.pack(T,r),Array.swap.go(T,Tree.pack(T,l),U32.shr(n),i,v,U32.is_lt(i,U32.shr(U32.shr(n))))) ==
          Array.swap.lo(T,Tree.pack(T,r),_) : Array<T> & T}
      {==}
    case False{}:
      %pr : {Array.swap.hi(T,Tree.pack(T,l),Array.swap.go(T,Tree.pack(T,r),U32.shr(n),U32.sub(i,U32.shr(n)),v,
          U32.is_lt(U32.sub(i,U32.shr(n)),U32.shr(U32.shr(n))))) == Array.swap.hi(T,Tree.pack(T,l),_) : Array<T> & T}
      {==}

law swap_go:
  for -T: Data
  for +tree: Tree.Tree<T>
  for +n: U32
  for +i: U32
  for +v: T
  {Array.swap.go(T,Tree.pack(T,tree),n,i,v,U32.is_lt(i,U32.shr(n))) == (Tree.pack(T,Tree.store(T,tree,n,i,v)),
      Tree.fetch(T,tree,n,i)) : Array<T> & T}
def swap_go(T,tree,n,i,v):
  match tree:
    case Tree.Leaf{old}:
      {==}
    case Tree.Node{+l,+r}:
      swap_node(T,l,r,n,i,v,U32.is_lt(i,U32.shr(n)),swap_go(T,l,U32.shr(n),i,v),swap_go(T,r,U32.shr(n),U32.sub(i,U32.shr(n)),v))

law swap:
  for -T: Data
  for +tree: Tree.Tree<T>
  for +i: U32
  for +v: T
  {Array.swap(T,Tree.pack(T,tree),i,v) == (Tree.pack(T,Tree.write(T,tree,i,v)),Tree.read(T,tree,i)) : Array<T> & T}
def swap(T,tree,i,v):
  %swap_go(T,tree,Tree.capacity(T,tree),U32.and(i,U32.sub(Tree.capacity(T,tree),1)),v) : {Array.swap.at(T,i,v,Array.size(T,
        Tree.pack(T,tree))) == _ : Array<T> & T}
  %size(T,tree) : {Array.swap.at(T,i,v,Array.size(T,Tree.pack(T,tree))) == Array.swap.at(T,i,v,_) : Array<T> & T}
  {==}

def write(T,tree,i,value):
  %Equal.sym(Array<T> & T,Array.swap(T,Tree.pack(T,tree),i,value),(Tree.pack(T,Tree.write(T,tree,i,value)),
      Tree.read(T,tree,i)),swap(T,tree,i,value)) : {Array.set.fin(T,_) == Tree.pack(T,Tree.write(T,tree,i,value)) : Array<T>}
  {==}

def guarded_read(~Element: Data,+fallback: Element,+valid: Bool,+tree: Tree.Tree<Element>,+index: U32) ->
  {Storage.read_when(Element,fallback,valid,Tree.pack(Element,tree),index) ==
    (Tree.pack(Element,tree),Bool.pick(Element,valid,Tree.read(Element,tree,index),fallback)) : Array<Element> & Element}:
  match valid:
    case False{}: {==}
    case True{}: read(Element,tree,index)

def bounded_read(~Element: Data,+fallback: Element,+tree: Tree.Tree<Element>,+index: U32,+logical_length: U32) ->
  {Storage.read_or(Element,fallback,Tree.pack(Element,tree),index,logical_length) ==
    (Tree.pack(Element,tree),Bool.pick(Element,U32.is_lt(index,logical_length) && U32.is_le(logical_length,Tree.capacity(Element,tree)),
      Tree.read(Element,tree,index),fallback)) : Array<Element> & Element}:
  %Equal.sym(Array<Element> & U32,Array.size(Element,Tree.pack(Element,tree)),(Tree.pack(Element,tree),Tree.capacity(Element,tree)),
    size(Element,tree)) :
    {Storage.read_sized(Element,fallback,index,logical_length,_) ==
      (Tree.pack(Element,tree),Bool.pick(Element,U32.is_lt(index,logical_length) && U32.is_le(logical_length,Tree.capacity(Element,tree)),
        Tree.read(Element,tree,index),fallback)) : Array<Element> & Element}
  guarded_read(~Element,fallback,U32.is_lt(index,logical_length) && U32.is_le(logical_length,Tree.capacity(Element,tree)),tree,index)

def bounded_read_is_get(~Element: Data,+fallback: Element,+tree: Tree.Tree<Element>,+index: U32,+logical_length: U32,
  inside: {U32.is_lt(index,logical_length) == True{} : Bool},
  room: {U32.is_le(logical_length,Tree.capacity(Element,tree)) == True{} : Bool}) ->
  {Storage.read_or(Element,fallback,Tree.pack(Element,tree),index,logical_length) ==
    Array.get(Element,Tree.pack(Element,tree),index) : Array<Element> & Element}:
  %Equal.sym(Array<Element> & U32,Array.size(Element,Tree.pack(Element,tree)),(Tree.pack(Element,tree),Tree.capacity(Element,tree)),
    size(Element,tree)) :
    {Storage.read_sized(Element,fallback,index,logical_length,_) == Array.get(Element,Tree.pack(Element,tree),index) : Array<Element> & Element}
  %Equal.sym(Bool,U32.is_lt(index,logical_length),True{},inside) :
    {Storage.read_when(Element,fallback,_ && U32.is_le(logical_length,Tree.capacity(Element,tree)),Tree.pack(Element,tree),index) ==
      Array.get(Element,Tree.pack(Element,tree),index) : Array<Element> & Element}
  %Equal.sym(Bool,U32.is_le(logical_length,Tree.capacity(Element,tree)),True{},room) :
    {Storage.read_when(Element,fallback,_,Tree.pack(Element,tree),index) == Array.get(Element,Tree.pack(Element,tree),index) : Array<Element> & Element}
  {==}

# Sequence transfers preserve native modular addressing for every element type.
# Logical bounds belong to callers, independently of this representation law.
def write_sequence(~Element: Data,+values: List<&2,Element>,+offset: U32,+tree: Tree.Tree<Element>) ->
  {Storage.write_list(Element,values,offset,Tree.pack(Element,tree)) ==
    Tree.pack(Element,Tree.write_sequence(~Element,values,offset,tree)) : Array<Element>}:
  match values:
    case Nil{}: {==}
    case +head <> +tail:
      %Equal.sym(Array<Element>,Array.set(Element,Tree.pack(Element,tree),offset,head),
        Tree.pack(Element,Tree.write(Element,tree,offset,head)),write(Element,tree,offset,head)) :
        {Storage.write_list(Element,tail,(offset + 1 : U32),_) ==
          Tree.pack(Element,Tree.write_sequence(~Element,tail,(offset + 1 : U32),Tree.write(Element,tree,offset,head))) : Array<Element>}
      write_sequence(~Element,tail,(offset + 1 : U32),Tree.write(Element,tree,offset,head))

def read_sequence(~Element: Data,+count: Nat,+offset: U32,+tree: Tree.Tree<Element>) ->
  {Storage.read_list(Element,count,offset,Tree.pack(Element,tree)) ==
    Tree.read_sequence(~Element,count,offset,tree) : List<&2,Element>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Array<Element> & Element,Array.get(Element,Tree.pack(Element,tree),offset),
        (Tree.pack(Element,tree),Tree.read(Element,tree,offset)),read(Element,tree,offset)) :
        {Storage.bind_pair(Array<Element>,Element,List<&2,Element>,_,storage => value =>
          value <> Storage.read_list(Element,rest,(offset + 1 : U32),storage)) ==
          Tree.read(Element,tree,offset) <> Tree.read_sequence(~Element,rest,(offset + 1 : U32),tree) : List<&2,Element>}
      %read_sequence(~Element,rest,(offset + 1 : U32),tree) :
        {Tree.read(Element,tree,offset) <> Storage.read_list(Element,rest,(offset + 1 : U32),Tree.pack(Element,tree)) ==
          Tree.read(Element,tree,offset) <> _ : List<&2,Element>}
      {==}

def sequence_length(~Element: Data,+count: Nat,+offset: U32,+tree: Tree.Tree<Element>) ->
  {List.length(&2,Element,Tree.read_sequence(~Element,count,offset,tree)) == count : Nat}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %sequence_length(~Element,rest,(offset + 1 : U32),tree) :
        {1n+List.length(&2,Element,Tree.read_sequence(~Element,rest,(offset + 1 : U32),tree)) == 1n+_ : Nat}
      {==}

# An observed logical prefix has the actual stored element at every valid index.
# Capacity need not equal count; address overflow remains explicitly modular.
def sequence_element(~Element: Data,+count: Nat,+offset: U32,+tree: Tree.Tree<Element>,+index: Nat,
  inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {List.get(&2,Element,Tree.read_sequence(~Element,count,offset,tree),index) ==
    Some{Tree.read(Element,tree,(offset + U32.from_nat(index) : U32))} : Maybe<&2,Element>}:
  match count index:
    case 0n 0n:
      Empty.absurd({None{} == Some{Tree.read(Element,tree,(offset + 0 : U32))} : Maybe<&2,Element>},WordEncoding.false_true(inside))
    case 0n 1n+previous:
      Empty.absurd({None{} == Some{Tree.read(Element,tree,(offset + U32.from_nat(1n+previous) : U32))} : Maybe<&2,Element>},WordEncoding.false_true(inside))
    case 1n+rest 0n:
      %Equal.sym(U32,(offset + 0 : U32),offset,Arithmetic.add_zero(offset)) :
        {Some{Tree.read(Element,tree,offset)} == Some{Tree.read(Element,tree,_)} : Maybe<&2,Element>}
      {==}
    case 1n+ +rest 1n+ +previous:
      %Equal.sym(U32,(offset + U32.from_nat(1n+previous) : U32),((offset + 1 : U32) + U32.from_nat(previous) : U32),
        Projection.indexed_successor(offset,previous)) :
        {List.get(&2,Element,Tree.read_sequence(~Element,rest,(offset + 1 : U32),tree),previous) ==
          Some{Tree.read(Element,tree,_)} : Maybe<&2,Element>}
      sequence_element(~Element,rest,(offset + 1 : U32),tree,previous,inside)

def sequence_missing(~Element: Data,+count: Nat,+offset: U32,+tree: Tree.Tree<Element>,+index: Nat,
  outside: {Nat.is_lt(index,count) == False{} : Bool}) ->
  {List.get(&2,Element,Tree.read_sequence(~Element,count,offset,tree),index) == None{} : Maybe<&2,Element>}:
  match count index:
    case 0n index: {==}
    case 1n+rest 0n:
      Empty.absurd({Some{Tree.read(Element,tree,offset)} == None{} : Maybe<&2,Element>},
        WordEncoding.false_true(Equal.sym(Bool,True{},False{},outside)))
    case 1n+rest 1n+previous:
      sequence_missing(~Element,rest,(offset + 1 : U32),tree,previous,outside)

def sequence_lookup_case(~Element: Data,+present: Bool,+length: U32,+tree: Tree.Tree<Element>,+index: U32,
  order: {U32.is_lt(index,length) == present : Bool}) ->
  {List.get(&2,Element,Tree.read_sequence(~Element,U32.to_nat(length),0,tree),U32.to_nat(index)) ==
    Bool.pick(Maybe<&2,Element>,present,Some{Tree.read(Element,tree,index)},None{}) : Maybe<&2,Element>}:
  match present:
    case False{}:
      sequence_missing(~Element,U32.to_nat(length),0,tree,U32.to_nat(index),
        %Comparison.lt(index,length) : {_ == False{} : Bool}
        order)
    case True{}:
      %Arithmetic.native_roundtrip(index) :
        {List.get(&2,Element,Tree.read_sequence(~Element,U32.to_nat(length),0,tree),U32.to_nat(index)) ==
          Some{Tree.read(Element,tree,_)} : Maybe<&2,Element>}
      %Arithmetic.add_zero_left(U32.from_nat(U32.to_nat(index))) :
        {List.get(&2,Element,Tree.read_sequence(~Element,U32.to_nat(length),0,tree),U32.to_nat(index)) ==
          Some{Tree.read(Element,tree,_)} : Maybe<&2,Element>}
      sequence_element(~Element,U32.to_nat(length),0,tree,U32.to_nat(index),
        %Comparison.lt(index,length) : {_ == True{} : Bool}
        order)

def default_after_presence(~Element: Data,+fallback: Element,+present: Bool,+value: Element) ->
  {Bool.pick(Element,present && True{},value,fallback) ==
    Maybe.default(&2,Element,Bool.pick(Maybe<&2,Element>,present,Some{value},None{}),fallback) : Element}:
  match present:
    case False{}: {==}
    case True{}: {==}

def bounded_read_observation_case(~Element: Data,+fallback: Element,+present: Bool,+length: U32,+tree: Tree.Tree<Element>,+index: U32,
  +order: {U32.is_lt(index,length) == present : Bool},room: {U32.is_le(length,Tree.capacity(Element,tree)) == True{} : Bool}) ->
  {Storage.read_or(Element,fallback,Tree.pack(Element,tree),index,length) ==
    (Tree.pack(Element,tree),Maybe.default(&2,Element,List.get(&2,Element,
      Tree.read_sequence(~Element,U32.to_nat(length),0,tree),U32.to_nat(index)),fallback)) : Array<Element> & Element}:
  %Equal.sym(Array<Element> & Element,Storage.read_or(Element,fallback,Tree.pack(Element,tree),index,length),
    (Tree.pack(Element,tree),Bool.pick(Element,U32.is_lt(index,length) && U32.is_le(length,Tree.capacity(Element,tree)),Tree.read(Element,tree,index),fallback)),
    bounded_read(~Element,fallback,tree,index,length)) :
    {_ == (Tree.pack(Element,tree),Maybe.default(&2,Element,List.get(&2,Element,
      Tree.read_sequence(~Element,U32.to_nat(length),0,tree),U32.to_nat(index)),fallback)) : Array<Element> & Element}
  %Equal.sym(Bool,U32.is_le(length,Tree.capacity(Element,tree)),True{},room) :
    {(Tree.pack(Element,tree),Bool.pick(Element,U32.is_lt(index,length) && _,Tree.read(Element,tree,index),fallback)) ==
      (Tree.pack(Element,tree),Maybe.default(&2,Element,List.get(&2,Element,
        Tree.read_sequence(~Element,U32.to_nat(length),0,tree),U32.to_nat(index)),fallback)) : Array<Element> & Element}
  %Equal.sym(Bool,U32.is_lt(index,length),present,order) :
    {(Tree.pack(Element,tree),Bool.pick(Element,_ && True{},Tree.read(Element,tree,index),fallback)) ==
      (Tree.pack(Element,tree),Maybe.default(&2,Element,List.get(&2,Element,
        Tree.read_sequence(~Element,U32.to_nat(length),0,tree),U32.to_nat(index)),fallback)) : Array<Element> & Element}
  %Equal.sym(Maybe<&2,Element>,List.get(&2,Element,Tree.read_sequence(~Element,U32.to_nat(length),0,tree),U32.to_nat(index)),
    Bool.pick(Maybe<&2,Element>,present,Some{Tree.read(Element,tree,index)},None{}),sequence_lookup_case(~Element,present,length,tree,index,order)) :
    {(Tree.pack(Element,tree),Bool.pick(Element,present && True{},Tree.read(Element,tree,index),fallback)) ==
      (Tree.pack(Element,tree),Maybe.default(&2,Element,_,fallback)) : Array<Element> & Element}
  %default_after_presence(~Element,fallback,present,Tree.read(Element,tree,index)) :
    {(Tree.pack(Element,tree),Bool.pick(Element,present && True{},Tree.read(Element,tree,index),fallback)) ==
      (Tree.pack(Element,tree),_) : Array<Element> & Element}
  {==}

# Full logical-read contract, including out-of-range fallback. Only real
# capacity is required; the logical length need not fill the allocation.
def bounded_read_observation(~Element: Data,+fallback: Element,+length: U32,+tree: Tree.Tree<Element>,+index: U32,
  room: {U32.is_le(length,Tree.capacity(Element,tree)) == True{} : Bool}) ->
  {Storage.read_or(Element,fallback,Tree.pack(Element,tree),index,length) ==
    (Tree.pack(Element,tree),Maybe.default(&2,Element,List.get(&2,Element,
      Tree.read_sequence(~Element,U32.to_nat(length),0,tree),U32.to_nat(index)),fallback)) : Array<Element> & Element}:
  bounded_read_observation_case(~Element,fallback,U32.is_lt(index,length),length,tree,index,{==},room)

# A fact about an existing storage prefix, not another runtime container.
# Both the allocator and an external Array owner must establish these fields.
type SequenceEvidence<-Element: Data,-length: Nat,-values: List<&2,Element>> is Data:
  SequenceEvidence{storage: Tree.Tree<Element>,
    contents: {Tree.read_sequence(~Element,length,0,storage) == values : List<&2,Element>},
    room: {Nat.is_le(length,U32.to_nat(Tree.capacity(Element,storage))) == True{} : Bool}}

def sequence_storage(-Element: Data,-length: Nat,-values: List<&2,Element>,
  evidence: SequenceEvidence<Element,length,values>) -> Tree.Tree<Element>:
  SequenceEvidence{storage,contents,room} = evidence
  storage

def sequence_owner(-Element: Data,-length: Nat,-values: List<&2,Element>,
  evidence: SequenceEvidence<Element,length,values>) -> Array<Element>:
  Tree.pack(Element,sequence_storage(Element,length,values,evidence))

# The unchecked primitive is usable after the caller establishes its logical
# index bound. This is independent of allocation shape and unused capacity.
def stored_sequence_element(~Element: Data,+fallback: Element,+length: Nat,+values: List<&2,Element>,
  +evidence: SequenceEvidence<Element,length,values>,+index: Nat,
  inside: {Nat.is_lt(index,length) == True{} : Bool}) ->
  {Tree.read(Element,sequence_storage(Element,length,values,evidence),U32.from_nat(index)) ==
    Maybe.default(&2,Element,List.get(&2,Element,values,index),fallback) : Element}:
  SequenceEvidence{+storage,+contents,room} = evidence
  %contents :
    {Tree.read(Element,storage,U32.from_nat(index)) == Maybe.default(&2,Element,List.get(&2,Element,_,index),fallback) : Element}
  %Equal.sym(Maybe<&2,Element>,List.get(&2,Element,Tree.read_sequence(~Element,length,0,storage),index),
    Some{Tree.read(Element,storage,(0 + U32.from_nat(index) : U32))},sequence_element(~Element,length,0,storage,index,inside)) :
    {Tree.read(Element,storage,U32.from_nat(index)) == Maybe.default(&2,Element,_,fallback) : Element}
  %Equal.sym(U32,(0 + U32.from_nat(index) : U32),U32.from_nat(index),Arithmetic.add_zero_left(U32.from_nat(index))) :
    {Tree.read(Element,storage,U32.from_nat(index)) == Tree.read(Element,storage,_) : Element}
  {==}

def read_sequence_evidence(~Element: Data,+fallback: Element,+length: Nat,+values: List<&2,Element>,
  +evidence: SequenceEvidence<Element,length,values>,+index: U32,+native_length: U32,
  +length_exact: {U32.to_nat(native_length) == length : Nat}) ->
  {Storage.read_or(Element,fallback,sequence_owner(Element,length,values,evidence),index,native_length) ==
    (sequence_owner(Element,length,values,evidence),Maybe.default(&2,Element,List.get(&2,Element,values,U32.to_nat(index)),fallback)) : Array<Element> & Element}:
  SequenceEvidence{+storage,+contents,room} = evidence
  %contents :
    {Storage.read_or(Element,fallback,Tree.pack(Element,storage),index,native_length) ==
      (Tree.pack(Element,storage),Maybe.default(&2,Element,List.get(&2,Element,_,U32.to_nat(index)),fallback)) : Array<Element> & Element}
  %length_exact :
    {Storage.read_or(Element,fallback,Tree.pack(Element,storage),index,native_length) ==
      (Tree.pack(Element,storage),Maybe.default(&2,Element,List.get(&2,Element,Tree.read_sequence(~Element,_,0,storage),U32.to_nat(index)),fallback)) : Array<Element> & Element}
  bounded_read_observation(~Element,fallback,native_length,storage,index,
    U32Division.nat_le_native(native_length,Tree.capacity(Element,storage),
      %Equal.sym(Nat,U32.to_nat(native_length),length,length_exact) :
        {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,storage))) == True{} : Bool}
      room))

# The logical address conversion is separate from the physical owner.
def read_encoded_sequence(~Element: Data,+fallback: Element,+length: Nat,+values: List<&2,Element>,
  +evidence: SequenceEvidence<Element,length,values>,+address: U32,+index: Nat,+native_length: U32,
  address_exact: {U32.to_nat(address) == index : Nat},length_exact: {U32.to_nat(native_length) == length : Nat}) ->
  {Storage.read_or(Element,fallback,sequence_owner(Element,length,values,evidence),address,native_length) ==
    (sequence_owner(Element,length,values,evidence),Maybe.default(&2,Element,List.get(&2,Element,values,index),fallback)) : Array<Element> & Element}:
  %address_exact :
    {Storage.read_or(Element,fallback,sequence_owner(Element,length,values,evidence),address,native_length) ==
      (sequence_owner(Element,length,values,evidence),Maybe.default(&2,Element,List.get(&2,Element,values,_),fallback)) : Array<Element> & Element}
  read_sequence_evidence(~Element,fallback,length,values,evidence,address,native_length,length_exact)

# Used only in propositions to compare reads from different physical owners.
def replace_read_owner(~Owner: Type,~Element: Data,owner: Owner,result: Owner & Element) -> Owner & Element:
  (previous,value) = result
  (owner,value)

def equivalent_sequence_read(~Element: Data,+fallback: Element,+length: Nat,+values: List<&2,Element>,
  +left: SequenceEvidence<Element,length,values>,+right: SequenceEvidence<Element,length,values>,
  +index: U32,+native_length: U32,+length_exact: {U32.to_nat(native_length) == length : Nat}) ->
  {Storage.read_or(Element,fallback,sequence_owner(Element,length,values,left),index,native_length) ==
    replace_read_owner(~Array<Element>,~Element,sequence_owner(Element,length,values,left),
      Storage.read_or(Element,fallback,sequence_owner(Element,length,values,right),index,native_length)) : Array<Element> & Element}:
  %Equal.sym(Array<Element> & Element,
    Storage.read_or(Element,fallback,sequence_owner(Element,length,values,right),index,native_length),
    (sequence_owner(Element,length,values,right),Maybe.default(&2,Element,List.get(&2,Element,values,U32.to_nat(index)),fallback)),
    read_sequence_evidence(~Element,fallback,length,values,right,index,native_length,length_exact)) :
    {Storage.read_or(Element,fallback,sequence_owner(Element,length,values,left),index,native_length) ==
      replace_read_owner(~Array<Element>,~Element,sequence_owner(Element,length,values,left),_) : Array<Element> & Element}
  read_sequence_evidence(~Element,fallback,length,values,left,index,native_length,length_exact)

# Reify an affine owner once for proof consumers that need both its Data model
# and the relation. This helper is never imported by a numerical runtime path.
type Observation<-Element: Data,-storage: Array<Element>> is Data:
  Observation{tree: Tree.Tree<Element>,packed: {Tree.pack(Element,tree) == storage : Array<Element>},
    reflected: {tree == Tree.reflect(Element,storage) : Tree.Tree<Element>}}

def observed_node_equation(-Element: Data,-left: Array<Element>,-right: Array<Element>,+left_tree: Tree.Tree<Element>,+right_tree: Tree.Tree<Element>,
  left_packed: {Tree.pack(Element,left_tree) == left : Array<Element>},right_packed: {Tree.pack(Element,right_tree) == right : Array<Element>}) ->
  {Tree.pack(Element,Tree.Node{left_tree,right_tree}) == ANode{left,right} : Array<Element>}:
  %left_packed : {Tree.pack(Element,Tree.Node{left_tree,right_tree}) == ANode{_,right} : Array<Element>}
  %right_packed : {Tree.pack(Element,Tree.Node{left_tree,right_tree}) == ANode{Tree.pack(Element,left_tree),_} : Array<Element>}
  {==}

def observed_node_reflection(-Element: Data,-left: Array<Element>,-right: Array<Element>,+left_tree: Tree.Tree<Element>,+right_tree: Tree.Tree<Element>,
  left_reflected: {left_tree == Tree.reflect(Element,left) : Tree.Tree<Element>},
  right_reflected: {right_tree == Tree.reflect(Element,right) : Tree.Tree<Element>}) ->
  {Tree.Node{left_tree,right_tree} == Tree.reflect(Element,ANode{left,right}) : Tree.Tree<Element>}:
  %left_reflected : {Tree.Node{left_tree,right_tree} == Tree.Node{_,Tree.reflect(Element,right)} : Tree.Tree<Element>}
  %right_reflected : {Tree.Node{left_tree,right_tree} == Tree.Node{left_tree,_} : Tree.Tree<Element>}
  {==}

def observed_node(-Element: Data,-left: Array<Element>,-right: Array<Element>,
  left_observation: Observation<Element,left>,right_observation: Observation<Element,right>) -> Observation<Element,ANode{left,right}>:
  Observation{+left_tree,left_packed,left_reflected} = left_observation
  Observation{+right_tree,right_packed,right_reflected} = right_observation
  Observation{Tree.Node{left_tree,right_tree},observed_node_equation(Element,left,right,left_tree,right_tree,left_packed,right_packed),
    observed_node_reflection(Element,left,right,left_tree,right_tree,left_reflected,right_reflected)}
def observe(-Element: Data,storage: Array<Element>) -> Observation<Element,storage>:
  match storage:
    case ALeaf{+value}: Observation{Tree.Leaf{value},{==},{==}}
    case ANode{left,right}: observed_node(Element,left,right,observe(Element,left),observe(Element,right))

# Transport a concrete owner observation into a proposition checked on trees.
# The proposition and canonical proof are proof-only; they are not loop inputs.
def lift_observation(~Element: Data,~Context: Data,
  ~Proposition: Context -> @-storage: Array<Element> -> @-tree: Tree.Tree<Element> -> Type,
  +context: Context,-storage: Array<Element>,observation: Observation<Element,storage>,
  canonical: @+tree: Tree.Tree<Element> -> Proposition(context,Tree.pack(Element,tree),tree)) ->
  Proposition(context,storage,Tree.reflect(Element,storage)):
  Observation{+tree,packed,reflected} = observation
  %reflected : Proposition(context,storage,_)
  %packed : Proposition(context,_,tree)
  canonical(tree)

def lift_pair_observation(~Element: Data,~Context: Data,
  ~Proposition: Context -> @-input: Array<Element> -> @-output: Array<Element> ->
    @-input_tree: Tree.Tree<Element> -> @-output_tree: Tree.Tree<Element> -> Type,
  +context: Context,-input: Array<Element>,-output: Array<Element>,
  input_observation: Observation<Element,input>,output_observation: Observation<Element,output>,
  canonical: @+input_tree: Tree.Tree<Element> -> @+output_tree: Tree.Tree<Element> ->
    Proposition(context,Tree.pack(Element,input_tree),Tree.pack(Element,output_tree),input_tree,output_tree)) ->
  Proposition(context,input,output,Tree.reflect(Element,input),Tree.reflect(Element,output)):
  Observation{+input_tree,input_packed,input_reflected} = input_observation
  Observation{+output_tree,output_packed,output_reflected} = output_observation
  %input_reflected : Proposition(context,input,output,_,Tree.reflect(Element,output))
  %output_reflected : Proposition(context,input,output,input_tree,_)
  %input_packed : Proposition(context,_,output,input_tree,output_tree)
  %output_packed : Proposition(context,Tree.pack(Element,input_tree),_,input_tree,output_tree)
  canonical(input_tree)(output_tree)
