# Storage readback starts with the concrete native tree operations. These
# equations require neither floating-point laws nor a caller schedule premise.
import Base
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./storage_address_bounds.bend as Address
import ./word_power_of_two.bend as Power
import ./nat_order.bend as Order

def capacity_node(~Element: Data,+left: Tree.Tree<Element>,+right: Tree.Tree<Element>,+half: U32,+index: U32,+value: Element,choice: Bool,
  left_capacity: {Tree.capacity(Element,Tree.store(Element,left,half,index,value)) == Tree.capacity(Element,left) : U32}) ->
  {Tree.capacity(Element,Bool.pick(Tree.Tree<Element>,choice,Tree.Node{Tree.store(Element,left,half,index,value),right},
      Tree.Node{left,Tree.store(Element,right,half,(index - half : U32),value)})) == U32.shl(Tree.capacity(Element,left)) : U32}:
  match choice:
    case True{}:
      Equal.cong(U32,U32,U32.shl,Tree.capacity(Element,Tree.store(Element,left,half,index,value)),Tree.capacity(Element,left),left_capacity)
    case False{}: {==}

def store_capacity(~Element: Data,tree: Tree.Tree<Element>,+capacity: U32,+index: U32,+value: Element) ->
  {Tree.capacity(Element,Tree.store(Element,tree,capacity,index,value)) == Tree.capacity(Element,tree) : U32}:
  match tree:
    case Tree.Leaf{old}: {==}
    case Tree.Node{+left,+right}:
      capacity_node(~Element,left,right,U32.shr(capacity),index,value,U32.is_lt(index,U32.shr(capacity)),store_capacity(~Element,left,U32.shr(capacity),index,value))

def write_capacity(~Element: Data,+tree: Tree.Tree<Element>,+index: U32,+value: Element) ->
  {Tree.capacity(Element,Tree.write(Element,tree,index,value)) == Tree.capacity(Element,tree) : U32}:
  store_capacity(~Element,tree,Tree.capacity(Element,tree),U32.and(index,U32.sub(Tree.capacity(Element,tree),1)),value)

def fetch_node(~Element: Data,+left: Tree.Tree<Element>,+right: Tree.Tree<Element>,+capacity: U32,+index: U32,+value: Element,choice: Bool,
  choice_equal: {U32.is_lt(index,U32.shr(capacity)) == choice : Bool},
  left_read: {Tree.fetch(Element,Tree.store(Element,left,U32.shr(capacity),index,value),U32.shr(capacity),index) == value : Element},
  right_read: {Tree.fetch(Element,Tree.store(Element,right,U32.shr(capacity),(index - U32.shr(capacity) : U32),value),U32.shr(capacity),(index - U32.shr(capacity) : U32)) == value : Element}) ->
  {Tree.fetch(Element,Bool.pick(Tree.Tree<Element>,choice,Tree.Node{Tree.store(Element,left,U32.shr(capacity),index,value),right},
      Tree.Node{left,Tree.store(Element,right,U32.shr(capacity),(index - U32.shr(capacity) : U32),value)}),capacity,index) == value : Element}:
  match choice:
    case True{}:
      %Equal.sym(Bool,U32.is_lt(index,U32.shr(capacity)),True{},choice_equal) :
        {Bool.pick(Element,_,Tree.fetch(Element,Tree.store(Element,left,U32.shr(capacity),index,value),U32.shr(capacity),index),
          Tree.fetch(Element,right,U32.shr(capacity),(index - U32.shr(capacity) : U32))) == value : Element}
      left_read
    case False{}:
      %Equal.sym(Bool,U32.is_lt(index,U32.shr(capacity)),False{},choice_equal) :
        {Bool.pick(Element,_,Tree.fetch(Element,left,U32.shr(capacity),index),
          Tree.fetch(Element,Tree.store(Element,right,U32.shr(capacity),(index - U32.shr(capacity) : U32),value),U32.shr(capacity),(index - U32.shr(capacity) : U32))) == value : Element}
      right_read

def fetch_store(~Element: Data,tree: Tree.Tree<Element>,+capacity: U32,+index: U32,+value: Element) ->
  {Tree.fetch(Element,Tree.store(Element,tree,capacity,index,value),capacity,index) == value : Element}:
  match tree:
    case Tree.Leaf{old}: {==}
    case Tree.Node{+left,+right}:
      fetch_node(~Element,left,right,capacity,index,value,U32.is_lt(index,U32.shr(capacity)),{==},
        fetch_store(~Element,left,U32.shr(capacity),index,value),fetch_store(~Element,right,U32.shr(capacity),(index - U32.shr(capacity) : U32),value))

def read_written(~Element: Data,+tree: Tree.Tree<Element>,+index: U32,+value: Element) ->
  {Tree.read(Element,Tree.write(Element,tree,index,value),index) == value : Element}:
  %Equal.sym(U32,Tree.capacity(Element,Tree.write(Element,tree,index,value)),Tree.capacity(Element,tree),write_capacity(~Element,tree,index,value)) :
    {Tree.fetch(Element,Tree.write(Element,tree,index,value),_,U32.and(index,U32.sub(_,1))) == value : Element}
  fetch_store(~Element,tree,Tree.capacity(Element,tree),U32.and(index,U32.sub(Tree.capacity(Element,tree),1)),value)

def other_node(~Element: Data,+left: Tree.Tree<Element>,+right: Tree.Tree<Element>,+capacity: U32,+index: U32,+query: U32,+value: Element,write_left: Bool,read_left: Bool,
  +write_choice: {U32.is_lt(index,U32.shr(capacity)) == write_left : Bool},+read_choice: {U32.is_lt(query,U32.shr(capacity)) == read_left : Bool},
  left_case: {U32.is_lt(index,U32.shr(capacity)) == True{} : Bool} -> {U32.is_lt(query,U32.shr(capacity)) == True{} : Bool} ->
    {Tree.fetch(Element,Tree.store(Element,left,U32.shr(capacity),index,value),U32.shr(capacity),query) == Tree.fetch(Element,left,U32.shr(capacity),query) : Element},
  right_case: {U32.is_lt(index,U32.shr(capacity)) == False{} : Bool} -> {U32.is_lt(query,U32.shr(capacity)) == False{} : Bool} ->
    {Tree.fetch(Element,Tree.store(Element,right,U32.shr(capacity),(index - U32.shr(capacity) : U32),value),U32.shr(capacity),(query - U32.shr(capacity) : U32)) ==
      Tree.fetch(Element,right,U32.shr(capacity),(query - U32.shr(capacity) : U32)) : Element}) ->
  {Tree.fetch(Element,Bool.pick(Tree.Tree<Element>,write_left,Tree.Node{Tree.store(Element,left,U32.shr(capacity),index,value),right},
      Tree.Node{left,Tree.store(Element,right,U32.shr(capacity),(index - U32.shr(capacity) : U32),value)}),capacity,query) == Tree.fetch(Element,Tree.Node{left,right},capacity,query) : Element}:
  match write_left read_left:
    case True{} True{}:
      %Equal.sym(Bool,U32.is_lt(query,U32.shr(capacity)),True{},read_choice) :
        {Bool.pick(Element,_,Tree.fetch(Element,Tree.store(Element,left,U32.shr(capacity),index,value),U32.shr(capacity),query),Tree.fetch(Element,right,U32.shr(capacity),(query - U32.shr(capacity) : U32))) ==
          Bool.pick(Element,_,Tree.fetch(Element,left,U32.shr(capacity),query),Tree.fetch(Element,right,U32.shr(capacity),(query - U32.shr(capacity) : U32))) : Element}
      left_case(write_choice)(read_choice)
    case True{} False{}:
      %Equal.sym(Bool,U32.is_lt(query,U32.shr(capacity)),False{},read_choice) :
        {Bool.pick(Element,_,Tree.fetch(Element,Tree.store(Element,left,U32.shr(capacity),index,value),U32.shr(capacity),query),Tree.fetch(Element,right,U32.shr(capacity),(query - U32.shr(capacity) : U32))) ==
          Bool.pick(Element,_,Tree.fetch(Element,left,U32.shr(capacity),query),Tree.fetch(Element,right,U32.shr(capacity),(query - U32.shr(capacity) : U32))) : Element}
      {==}
    case False{} True{}:
      %Equal.sym(Bool,U32.is_lt(query,U32.shr(capacity)),True{},read_choice) :
        {Bool.pick(Element,_,Tree.fetch(Element,left,U32.shr(capacity),query),Tree.fetch(Element,Tree.store(Element,right,U32.shr(capacity),(index - U32.shr(capacity) : U32),value),U32.shr(capacity),(query - U32.shr(capacity) : U32))) ==
          Bool.pick(Element,_,Tree.fetch(Element,left,U32.shr(capacity),query),Tree.fetch(Element,right,U32.shr(capacity),(query - U32.shr(capacity) : U32))) : Element}
      {==}
    case False{} False{}:
      %Equal.sym(Bool,U32.is_lt(query,U32.shr(capacity)),False{},read_choice) :
        {Bool.pick(Element,_,Tree.fetch(Element,left,U32.shr(capacity),query),Tree.fetch(Element,Tree.store(Element,right,U32.shr(capacity),(index - U32.shr(capacity) : U32),value),U32.shr(capacity),(query - U32.shr(capacity) : U32))) ==
          Bool.pick(Element,_,Tree.fetch(Element,left,U32.shr(capacity),query),Tree.fetch(Element,right,U32.shr(capacity),(query - U32.shr(capacity) : U32))) : Element}
      right_case(write_choice)(read_choice)

def fetch_preserved(~Element: Data,+depth: Nat,tree: Tree.Tree<Element>,+index: U32,+query: U32,+value: Element,
  balanced: Shape.Balanced(Element,depth,tree),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +index_inside: {U32.is_lt(index,Address.capacity(depth)) == True{} : Bool},+query_inside: {U32.is_lt(query,Address.capacity(depth)) == True{} : Bool},
  +different: {U32.is_eq(index,query) == False{} : Bool}) ->
  {Tree.fetch(Element,Tree.store(Element,tree,Address.capacity(depth),index,value),Address.capacity(depth),query) == Tree.fetch(Element,tree,Address.capacity(depth),query) : Element}:
  match depth tree:
    case 0n Tree.Leaf{old}:
      Empty.absurd({value == old : Element},Address.unequal_words(index,query,different,Address.equal_values(index,query,
        Equal.trans(Nat,U32.to_nat(index),0n,U32.to_nat(query),Equal.sym(Nat,0n,U32.to_nat(index),Address.below_one_value(index,index_inside)),Address.below_one_value(query,query_inside)))))
    case 0n Tree.Node{left,right}: match balanced:
    case 1n+rest Tree.Leaf{old}: match balanced:
    case 1n+ +rest Tree.Node{+left,+right}:
      Shape.Both{+left_balanced,+right_balanced} = balanced
      +half_equal = Power.native_half_power(rest,Order.lt_succ(rest,30n,Address.child_limit(rest,limit)))
      other_node(~Element,left,right,Address.capacity(1n+rest),index,query,value,U32.is_lt(index,U32.shr(Address.capacity(1n+rest))),U32.is_lt(query,U32.shr(Address.capacity(1n+rest))),{==},{==},
        in_left => query_left =>
          %Equal.sym(U32,U32.shr(Address.capacity(1n+rest)),Address.capacity(rest),half_equal) :
            {Tree.fetch(Element,Tree.store(Element,left,_,index,value),_,query) == Tree.fetch(Element,left,_,query) : Element}
          fetch_preserved(~Element,rest,left,index,query,value,left_balanced,Address.child_limit(rest,limit),
            %half_equal : {U32.is_lt(index,_) == True{} : Bool} in_left,
            %half_equal : {U32.is_lt(query,_) == True{} : Bool} query_left,different),
        in_right => query_right =>
          %Equal.sym(U32,U32.shr(Address.capacity(1n+rest)),Address.capacity(rest),half_equal) :
            {Tree.fetch(Element,Tree.store(Element,right,_,(index - _ : U32),value),_,(query - _ : U32)) == Tree.fetch(Element,right,_,(query - _ : U32)) : Element}
          fetch_preserved(~Element,rest,right,(index - Address.capacity(rest) : U32),(query - Address.capacity(rest) : U32),value,right_balanced,Address.child_limit(rest,limit),
            Address.right_stays_in_child(rest,index,limit,index_inside,%half_equal : {U32.is_lt(index,_) == False{} : Bool} in_right),
            Address.right_stays_in_child(rest,query,limit,query_inside,%half_equal : {U32.is_lt(query,_) == False{} : Bool} query_right),Address.subtract_distinct(index,query,Address.capacity(rest),different)))

def no_wrap(~Element: Data,+depth: Nat,+tree: Tree.Tree<Element>,+index: U32,+balanced: Shape.Balanced(Element,depth,tree),
  inside: {U32.is_lt(index,Tree.capacity(Element,tree)) == True{} : Bool}) ->
  {U32.and(index,U32.sub(Tree.capacity(Element,tree),1)) == index : U32}:
  %Equal.sym(U32,Tree.capacity(Element,tree),Address.capacity(depth),Shape.capacity(Element,depth,tree,balanced)) :
    {U32.and(index,U32.sub(_,1)) == index : U32}
  Power.native_mask_preserves_bounded(depth,index)(
    %Shape.capacity(Element,depth,tree,balanced) : {U32.is_lt(index,_) == True{} : Bool}
    inside)

def read_preserved(~Element: Data,+depth: Nat,+tree: Tree.Tree<Element>,+index: U32,+query: U32,+value: Element,
  +balanced: Shape.Balanced(Element,depth,tree),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +index_inside: {U32.is_lt(index,Tree.capacity(Element,tree)) == True{} : Bool},+query_inside: {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool},
  different: {U32.is_eq(index,query) == False{} : Bool}) ->
  {Tree.read(Element,Tree.write(Element,tree,index,value),query) == Tree.read(Element,tree,query) : Element}:
  +capacity = Tree.capacity(Element,tree)
  +capacity_equal = Shape.capacity(Element,depth,tree,balanced)
  %Equal.sym(U32,Tree.capacity(Element,Tree.write(Element,tree,index,value)),capacity,write_capacity(~Element,tree,index,value)) :
    {Tree.fetch(Element,Tree.write(Element,tree,index,value),_,U32.and(query,U32.sub(_,1))) == Tree.read(Element,tree,query) : Element}
  %Equal.sym(U32,U32.and(index,U32.sub(capacity,1)),index,no_wrap(~Element,depth,tree,index,balanced,index_inside)) :
    {Tree.fetch(Element,Tree.store(Element,tree,capacity,_,value),capacity,U32.and(query,U32.sub(capacity,1))) == Tree.read(Element,tree,query) : Element}
  %Equal.sym(U32,U32.and(query,U32.sub(capacity,1)),query,no_wrap(~Element,depth,tree,query,balanced,query_inside)) :
    {Tree.fetch(Element,Tree.store(Element,tree,capacity,index,value),capacity,_) == Tree.fetch(Element,tree,capacity,_) : Element}
  %Equal.sym(U32,capacity,Address.capacity(depth),capacity_equal) :
    {Tree.fetch(Element,Tree.store(Element,tree,_,index,value),_,query) == Tree.fetch(Element,tree,_,query) : Element}
  fetch_preserved(~Element,depth,tree,index,query,value,balanced,limit,
    %capacity_equal : {U32.is_lt(index,_) == True{} : Bool} index_inside,
    %capacity_equal : {U32.is_lt(query,_) == True{} : Bool} query_inside,different)

# Every address of a fresh tree reads its fill value.
def pick_same(~Element: Data,selected: Bool,+first: Element,+second: Element,+value: Element,first_equal: {first == value : Element},second_equal: {second == value : Element}) ->
  {Bool.pick(Element,selected,first,second) == value : Element}:
  match selected:
    case True{}: first_equal
    case False{}: second_equal

def fetch_fresh(~Element: Data,depth: Nat,+value: Element,+size: U32,+index: U32) -> {Tree.fetch(Element,Tree.fresh(Element,depth,value),size,index) == value : Element}:
  match depth:
    case 0n: {==}
    case 1n+ +rest:
      pick_same(~Element,U32.is_lt(index,U32.shr(size)),Tree.fetch(Element,Tree.fresh(Element,rest,value),U32.shr(size),index),
        Tree.fetch(Element,Tree.fresh(Element,rest,value),U32.shr(size),U32.sub(index,U32.shr(size))),value,
        fetch_fresh(~Element,rest,value,U32.shr(size),index),fetch_fresh(~Element,rest,value,U32.shr(size),U32.sub(index,U32.shr(size))))

def read_fresh(~Element: Data,+depth: Nat,+value: Element,+index: U32) -> {Tree.read(Element,Tree.fresh(Element,depth,value),index) == value : Element}:
  fetch_fresh(~Element,depth,value,Tree.capacity(Element,Tree.fresh(Element,depth,value)),U32.and(index,U32.sub(Tree.capacity(Element,Tree.fresh(Element,depth,value)),1)))
