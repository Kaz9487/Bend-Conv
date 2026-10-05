# Public Buffer proofs obtain balance from the controlled representation.
# These observations belong only to semantic proofs, never to numerical loops.
import Base
import ../storage_buffer.bend as Storage
import ./storage_certificate.bend as Certificate
import ./storage_witness.bend as Witness
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./write_program.bend as Writes

def storage(~Element: Data,buffer: Storage.Buffer<Element>) -> Array<Element>:
  Storage.Buffer{storage,length,certificate} = buffer
  storage

def length(~Element: Data,buffer: Storage.Buffer<Element>) -> Nat:
  Storage.Buffer{storage,length,certificate} = buffer
  length

def tree(~Element: Data,buffer: Storage.Buffer<Element>) -> Tree.Tree<Element>:
  Tree.reflect(Element,storage(~Element,buffer))

type Balanced<-Element: Data,-buffer: Storage.Buffer<Element>> is Data:
  Balanced{depth: Nat,proof: Shape.Balanced(Element,depth,tree(~Element,buffer))}

def balanced(~Element: Data,buffer: Storage.Buffer<Element>) -> Balanced<Element,buffer>:
  Storage.Buffer{storage,length,Certificate.Certificate{+depth,equation}} = buffer
  Balanced{depth,Witness.balanced(Element,depth,storage,Certificate.witness_at(Element,depth,storage,equation))}

def allocate(~Element: Data,+depth: Nat,+value: Element,+length: Nat) -> Balanced<Element,Storage.allocate(~Element,depth,value,length)>:
  balanced(~Element,Storage.allocate(~Element,depth,value,length))

def from_list(~Element: Data,+values: List<&2,Element>,+depth: Nat,+fallback: Element,+length: Nat) ->
  Balanced<Element,Storage.from_list(~Element,values,depth,fallback,length)>:
  balanced(~Element,Storage.from_list(~Element,values,depth,fallback,length))

type Written<-Element: Data,-buffer: Storage.Buffer<Element>,-writes: List<&2,Writes.Assignment<Element>>> is Data:
  Written{depth: Nat,proof: Shape.Balanced(Element,depth,Writes.pure(Element,writes,tree(~Element,buffer)))}

# Reuse storage_shape.write/store, through the existing write-program invariant.
# The required initial balance is discharged from Buffer, not a caller premise.
def written(~Element: Data,+writes: List<&2,Writes.Assignment<Element>>,buffer: Storage.Buffer<Element>) -> Written<Element,buffer,writes>:
  Storage.Buffer{storage,length,Certificate.Certificate{+depth,equation}} = buffer
  Written{depth,{
    %Certificate.restored(Element,depth,storage,Certificate.witness_at(Element,depth,storage,equation)) :
      Shape.Balanced(Element,depth,Writes.pure(Element,writes,Tree.reflect(Element,_)))
    Writes.writes_preserve_balance(Element,writes,Tree.reflect(Element,Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,storage)))),depth,
      Witness.balanced(Element,depth,Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,storage))),
        Certificate.normalized_witness(Element,depth,Tree.reflect(Element,storage)))) :
      Shape.Balanced(Element,depth,Writes.pure(Element,writes,Tree.reflect(Element,storage)))}}
