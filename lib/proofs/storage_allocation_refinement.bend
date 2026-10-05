import Base
import ../storage_buffer.bend as Storage
import ../storage_allocation.bend as Allocation
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_buffer_model.bend as Buffer

# Dispatch changes neither the initialized array nor its live certificate.
# This is an unconditional equation for every element type, depth and length.
def dispatch_same(-Element: Data,-allocator: Nat -> Element -> Nat -> Storage.Buffer<Element>,depth: Nat,+value: Element,+length: Nat) ->
  {Allocation.dispatch(~Element,~allocator,depth,value,length) == allocator(depth,value,length) : Storage.Buffer<Element>}:
  match depth:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 8n: {==}
    case 9n: {==}
    case 10n: {==}
    case 11n: {==}
    case 12n: {==}
    case 13n: {==}
    case 14n: {==}
    case 15n: {==}
    case 16n: {==}
    case 17n: {==}
    case 18n: {==}
    case 19n: {==}
    case 20n: {==}
    case 21n: {==}
    case 22n: {==}
    case 23n: {==}
    case 24n: {==}
    case 25n: {==}
    case 26n: {==}
    case 27n: {==}
    case 28n: {==}
    case 29n: {==}
    case 30n: {==}
    case 31n+rest: {==}

def array_same(-Element: Data,+depth: Nat,+value: Element) ->
  {Allocation.array(~Element,depth,value) == Array.new(Element,depth,value) : Array<Element>}:
  %Equal.sym(Storage.Buffer<Element>,Allocation.allocate(~Element,depth,value,0n),Storage.allocate(~Element,depth,value,0n),
    dispatch_same(Element,Allocation.allocated(~Element),depth,value,0n)) :
    {Allocation.storage(~Element,_) == Array.new(Element,depth,value) : Array<Element>}
  {==}

def fresh(~Element: Data,+depth: Nat,+value: Element) ->
  {Allocation.array(~Element,depth,value) == Tree.pack(Element,Tree.fresh(Element,depth,value)) : Array<Element>}:
  Equal.trans(Array<Element>,Allocation.array(~Element,depth,value),Array.new(Element,depth,value),
    Tree.pack(Element,Tree.fresh(Element,depth,value)),array_same(Element,depth,value),StorageLaws.fresh(Element,depth,value))

def contents(~Element: Data,+depth: Nat,+value: Element,+length: Nat) ->
  {Buffer.contents(~Element,Allocation.allocate(~Element,depth,value,length)) == Buffer.Content{Tree.fresh(Element,depth,value),length} : Buffer.Content<Element>}:
  %Equal.sym(Storage.Buffer<Element>,Allocation.allocate(~Element,depth,value,length),Storage.allocate(~Element,depth,value,length),dispatch_same(Element,Allocation.allocated(~Element),depth,value,length)) :
    {Buffer.contents(~Element,_) == Buffer.Content{Tree.fresh(Element,depth,value),length} : Buffer.Content<Element>}
  %Equal.sym(Array<Element>,Array.new(Element,depth,value),Tree.pack(Element,Tree.fresh(Element,depth,value)),StorageLaws.fresh(Element,depth,value)) :
    {Buffer.Content{Tree.reflect(Element,_),length} == Buffer.Content{Tree.fresh(Element,depth,value),length} : Buffer.Content<Element>}
  Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,length},Tree.reflect(Element,Tree.pack(Element,Tree.fresh(Element,depth,value))),Tree.fresh(Element,depth,value),StorageLaws.reflect_packed(Element,Tree.fresh(Element,depth,value)))
