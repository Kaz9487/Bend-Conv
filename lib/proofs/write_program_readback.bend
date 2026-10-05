# Readback of any bounded native store program. The observed value is the
# last matching assignment; writes at other valid addresses preserve it.
import Base
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./storage_readback.bend as StorageReadback
import ./storage_address_bounds.bend as Address
import ./write_program.bend as Writes
import ./boolean_logic.bend as Boolean

def Inside(~Element: Data,+capacity: U32,writes: List<&2,Writes.Assignment<Element>>) -> Data:
  match writes:
    case Nil{}: Unit
    case Writes.Put{index,value} <> rest:
      Shape.Both<{U32.is_lt(index,capacity) == True{} : Bool},Inside(~Element,capacity,rest)>

def selected(~Element: Data,writes: List<&2,Writes.Assignment<Element>>,+query: U32,previous: Element) -> Element:
  match writes:
    case Nil{}: previous
    case Writes.Put{index,value} <> rest:
      selected(~Element,rest,query,Bool.pick(Element,U32.is_eq(index,query),value,previous))

def inside_append(~Element: Data,+capacity: U32,first: List<&2,Writes.Assignment<Element>>,+second: List<&2,Writes.Assignment<Element>>,
  first_inside: Inside(~Element,capacity,first),+second_inside: Inside(~Element,capacity,second)) -> Inside(~Element,capacity,List.append(&2,Writes.Assignment<Element>,first,second)):
  match first:
    case Nil{}: second_inside
    case Writes.Put{index,value} <> rest:
      Shape.Both{head,tail} = first_inside
      Shape.Both{head,inside_append(~Element,capacity,rest,second,tail,second_inside)}

def selected_append(~Element: Data,first: List<&2,Writes.Assignment<Element>>,+second: List<&2,Writes.Assignment<Element>>,+query: U32,+previous: Element) ->
  {selected(~Element,List.append(&2,Writes.Assignment<Element>,first,second),query,previous) == selected(~Element,second,query,selected(~Element,first,query,previous)) : Element}:
  match first:
    case Nil{}: {==}
    case Writes.Put{index,value} <> rest:
      selected_append(~Element,rest,second,query,Bool.pick(Element,U32.is_eq(index,query),value,previous))

def one_compared(~Element: Data,+depth: Nat,+tree: Tree.Tree<Element>,+index: U32,+query: U32,+value: Element,equal: Bool,
  compared: {U32.is_eq(index,query) == equal : Bool},balanced: Shape.Balanced(Element,depth,tree),limit: {Nat.is_le(depth,30n) == True{} : Bool},
  index_inside: {U32.is_lt(index,Tree.capacity(Element,tree)) == True{} : Bool},query_inside: {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool}) ->
  {Tree.read(Element,Tree.write(Element,tree,index,value),query) == Bool.pick(Element,equal,value,Tree.read(Element,tree,query)) : Element}:
  match equal:
    case True{}:
      %Address.equal_words(index,query,compared) : {Tree.read(Element,Tree.write(Element,tree,index,value),_) == value : Element}
      StorageReadback.read_written(~Element,tree,index,value)
    case False{}: StorageReadback.read_preserved(~Element,depth,tree,index,query,value,balanced,limit,index_inside,query_inside,compared)

def one(~Element: Data,+depth: Nat,+tree: Tree.Tree<Element>,+index: U32,+query: U32,+value: Element,
  balanced: Shape.Balanced(Element,depth,tree),limit: {Nat.is_le(depth,30n) == True{} : Bool},
  index_inside: {U32.is_lt(index,Tree.capacity(Element,tree)) == True{} : Bool},query_inside: {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool}) ->
  {Tree.read(Element,Tree.write(Element,tree,index,value),query) == Bool.pick(Element,U32.is_eq(index,query),value,Tree.read(Element,tree,query)) : Element}:
  one_compared(~Element,depth,tree,index,query,value,U32.is_eq(index,query),{==},balanced,limit,index_inside,query_inside)

def program(~Element: Data,writes: List<&2,Writes.Assignment<Element>>,+depth: Nat,+tree: Tree.Tree<Element>,+query: U32,
  +balanced: Shape.Balanced(Element,depth,tree),+limit: {Nat.is_le(depth,30n) == True{} : Bool},
  +query_inside: {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool},inside: Inside(~Element,Tree.capacity(Element,tree),writes)) ->
  {Tree.read(Element,Writes.pure(Element,writes,tree),query) == selected(~Element,writes,query,Tree.read(Element,tree,query)) : Element}:
  match writes:
    case Nil{}: {==}
    case Writes.Put{+index,+value} <> +rest:
      Shape.Both{index_inside,rest_inside} = inside
      +written = Tree.write(Element,tree,index,value)
      +capacity_equal = StorageReadback.write_capacity(~Element,tree,index,value)
      Equal.trans(Element,Tree.read(Element,Writes.pure(Element,rest,written),query),selected(~Element,rest,query,Tree.read(Element,written,query)),
        selected(~Element,rest,query,Bool.pick(Element,U32.is_eq(index,query),value,Tree.read(Element,tree,query))),
        program(~Element,rest,depth,written,query,Shape.write(Element,depth,tree,index,value,balanced),limit,
          %Equal.sym(U32,Tree.capacity(Element,written),Tree.capacity(Element,tree),capacity_equal) : {U32.is_lt(query,_) == True{} : Bool} query_inside,
          %Equal.sym(U32,Tree.capacity(Element,written),Tree.capacity(Element,tree),capacity_equal) : Inside(~Element,_,rest) rest_inside),
        Equal.cong(Element,Element,previous => selected(~Element,rest,query,previous),Tree.read(Element,written,query),
          Bool.pick(Element,U32.is_eq(index,query),value,Tree.read(Element,tree,query)),one(~Element,depth,tree,index,query,value,balanced,limit,index_inside,query_inside)))

# A producer supplies matching values and complete coverage independently.
# Later writes need not be disjoint if they agree at the queried address.
def Matching(~Element: Data,+query: U32,+wanted: Element,writes: List<&2,Writes.Assignment<Element>>) -> Data:
  match writes:
    case []: Unit
    case Writes.Put{index,value} <> rest:
      Shape.Both<Boolean.When(U32.is_eq(index,query),{value == wanted : Element}),Matching(~Element,query,wanted,rest)>

def contains(~Element: Data,+query: U32,writes: List<&2,Writes.Assignment<Element>>) -> Bool:
  match writes:
    case []: False{}
    case Writes.Put{index,value} <> rest: U32.is_eq(index,query) || contains(~Element,query,rest)

def value_matches(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+destination: U32,+query: U32,+actual: Element,equal: Bool,
  compared: {U32.is_eq(destination,query) == equal : Bool},correct: {actual == value(context,destination) : Element}) ->
  Boolean.When(equal,{actual == value(context,query) : Element}):
  match equal:
    case False{}: Unit{}
    case True{}:
      Equal.trans(Element,actual,value(context,destination),value(context,query),correct,
        Equal.cong(U32,Element,index => value(context,index),destination,query,Address.equal_words(destination,query,compared)))

def selected_matching(~Element: Data,writes: List<&2,Writes.Assignment<Element>>,+query: U32,+wanted: Element,+previous: Element,matching: Matching(~Element,query,wanted,writes)) ->
  {selected(~Element,writes,query,previous) == Bool.pick(Element,contains(~Element,query,writes),wanted,previous) : Element}:
  match writes:
    case []: {==}
    case Writes.Put{+index,+value} <> +rest:
      Shape.Both{head,tail} = matching
      %Equal.sym(Element,selected(~Element,rest,query,Bool.pick(Element,U32.is_eq(index,query),value,previous)),
        Bool.pick(Element,contains(~Element,query,rest),wanted,Bool.pick(Element,U32.is_eq(index,query),value,previous)),
        selected_matching(~Element,rest,query,wanted,Bool.pick(Element,U32.is_eq(index,query),value,previous),tail)) :
        {_ == Bool.pick(Element,U32.is_eq(index,query) || contains(~Element,query,rest),wanted,previous) : Element}
      Boolean.choose_matching(Element,U32.is_eq(index,query),contains(~Element,query,rest),value,wanted,previous,head)

def matching_append(~Element: Data,+query: U32,+wanted: Element,left: List<&2,Writes.Assignment<Element>>,+right: List<&2,Writes.Assignment<Element>>,
  left_matching: Matching(~Element,query,wanted,left),right_matching: Matching(~Element,query,wanted,right)) ->
  Matching(~Element,query,wanted,List.append(&2,Writes.Assignment<Element>,left,right)):
  match left:
    case []: right_matching
    case Writes.Put{index,value} <> rest:
      Shape.Both{head,tail} = left_matching
      Shape.Both{head,matching_append(~Element,query,wanted,rest,right,tail,right_matching)}

def contains_append(~Element: Data,+query: U32,left: List<&2,Writes.Assignment<Element>>,+right: List<&2,Writes.Assignment<Element>>) ->
  {contains(~Element,query,List.append(&2,Writes.Assignment<Element>,left,right)) == contains(~Element,query,left) || contains(~Element,query,right) : Bool}:
  match left:
    case []: {==}
    case Writes.Put{+index,value} <> +rest:
      %Equal.sym(Bool,contains(~Element,query,List.append(&2,Writes.Assignment<Element>,rest,right)),contains(~Element,query,rest) || contains(~Element,query,right),contains_append(~Element,query,rest,right)) :
        {U32.is_eq(index,query) || _ == (U32.is_eq(index,query) || contains(~Element,query,rest)) || contains(~Element,query,right) : Bool}
      Boolean.or_associative(U32.is_eq(index,query),contains(~Element,query,rest),contains(~Element,query,right))

def covered_left(~Element: Data,+query: U32,+left: List<&2,Writes.Assignment<Element>>,+right: List<&2,Writes.Assignment<Element>>,
  covered: {contains(~Element,query,left) == True{} : Bool}) -> {contains(~Element,query,List.append(&2,Writes.Assignment<Element>,left,right)) == True{} : Bool}:
  %Equal.sym(Bool,contains(~Element,query,List.append(&2,Writes.Assignment<Element>,left,right)),contains(~Element,query,left) || contains(~Element,query,right),contains_append(~Element,query,left,right)) : {_ == True{} : Bool}
  %Equal.sym(Bool,contains(~Element,query,left),True{},covered) : {_ || contains(~Element,query,right) == True{} : Bool}
  {==}

def covered_right(~Element: Data,+query: U32,+left: List<&2,Writes.Assignment<Element>>,+right: List<&2,Writes.Assignment<Element>>,
  covered: {contains(~Element,query,right) == True{} : Bool}) -> {contains(~Element,query,List.append(&2,Writes.Assignment<Element>,left,right)) == True{} : Bool}:
  %Equal.sym(Bool,contains(~Element,query,List.append(&2,Writes.Assignment<Element>,left,right)),contains(~Element,query,left) || contains(~Element,query,right),contains_append(~Element,query,left,right)) : {_ == True{} : Bool}
  %Equal.sym(Bool,contains(~Element,query,right),True{},covered) : {contains(~Element,query,left) || _ == True{} : Bool}
  Boolean.or_true_right(contains(~Element,query,left))

def covered(~Element: Data,+writes: List<&2,Writes.Assignment<Element>>,+depth: Nat,+tree: Tree.Tree<Element>,+query: U32,+wanted: Element,
  balanced: Shape.Balanced(Element,depth,tree),limit: {Nat.is_le(depth,30n) == True{} : Bool},
  query_inside: {U32.is_lt(query,Tree.capacity(Element,tree)) == True{} : Bool},inside: Inside(~Element,Tree.capacity(Element,tree),writes),
  matching: Matching(~Element,query,wanted,writes),coverage: {contains(~Element,query,writes) == True{} : Bool}) ->
  {Tree.read(Element,Writes.pure(Element,writes,tree),query) == wanted : Element}:
  Equal.trans(Element,Tree.read(Element,Writes.pure(Element,writes,tree),query),selected(~Element,writes,query,Tree.read(Element,tree,query)),wanted,
    program(~Element,writes,depth,tree,query,balanced,limit,query_inside,inside),
    %coverage : {selected(~Element,writes,query,Tree.read(Element,tree,query)) == Bool.pick(Element,_,wanted,Tree.read(Element,tree,query)) : Element}
    selected_matching(~Element,writes,query,wanted,Tree.read(Element,tree,query),matching))
