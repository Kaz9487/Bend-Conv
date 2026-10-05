import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/storage_shape.bend as Shape
# Whole-list roundtrip through the actual native pack/unpack API.
import Base
import ./encoded_array_read.bend as EncodedRead
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./list_array_storage.bend as ListStorage
import ./list_array_allocation.bend as ListArrayAllocation
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/u32_successor.bend as U32Successor
import ./storage_capacity.bend as StorageCapacity
import ../../lib/proofs/u32_division.bend as U32Division
import ../specification.bend as ConvSpec
import ../scalar_kernel.bend as ScalarConv

def enumerate(n: Nat,+i: Nat,+xs: List<&2,F32>) -> List<&2,F32>:
  match n:
    case 0n:
      []
    case 1n+rest:
      ConvSpec.at(i,xs) <> enumerate(rest,1n+i,xs)

def small_index(+xs: List<&2,F32>,+i: U32,proof: ListArrayAllocation.Ready(xs),h: {Nat.is_lt(U32.to_nat(i),
      ConvSpec.length(xs)) == True{} : Bool}) -> {U32.is_lt(i,2147483648) == True{} : Bool}:
  Shape.Both{limit,room} = proof
  Bounds.uint_trans(i,Tree.capacity(F32,ListArrayAllocation.initial(xs)),2147483648,U32Division.nat_lt_native(i,
      Tree.capacity(F32,ListArrayAllocation.initial(xs)),Bounds.nat_trans(U32.to_nat(i),ConvSpec.length(xs),
        U32.to_nat(Tree.capacity(F32,ListArrayAllocation.initial(xs))),h,room)),StorageCapacity.small(ListArrayAllocation.depth(xs),
      ListArrayAllocation.initial(xs),Shape.fresh(F32,ListArrayAllocation.depth(xs),0.0),limit))

def next_room(+xs: List<&2,F32>,+i: U32,+n: Nat,+proof: ListArrayAllocation.Ready(xs),+room: {Nat.is_le(Nat.add(U32.to_nat(i),1n+n),
      ConvSpec.length(xs)) == True{} : Bool}) -> {Nat.is_le(Nat.add(U32.to_nat((i + 1 : U32)),n),ConvSpec.length(xs)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat((i + 1 : U32)),1n+U32.to_nat(i),U32Successor.successor(i)(small_index(xs,i,proof,Order.room_index(U32.to_nat(i),n,
          ConvSpec.length(xs),room)))) : {Nat.is_le(Nat.add(_,n),ConvSpec.length(xs)) == True{} : Bool}
  Order.advance_room(U32.to_nat(i),n,ConvSpec.length(xs),room)

law readback:
  for +n: Nat
  for +xs: List<&2,F32>
  for +i: U32
  for +proof: ListArrayAllocation.Ready(xs)
  for room: {Nat.is_le(Nat.add(U32.to_nat(i),n),ConvSpec.length(xs)) == True{} : Bool}
  {Tree.read_sequence(~F32,n,i,ListStorage.representation(xs)) == enumerate(n,U32.to_nat(i),xs) : List<&2,F32>}
def readback(n,xs,i,proof,room):
  match n:
    case 0n:
      {==}
    case 1n+ +rest:
      +room = room
      +inside = Order.room_index(U32.to_nat(i),rest,ConvSpec.length(xs),room)
      %ListArrayAllocation.read(xs,i,proof,inside) : {Tree.read(F32,ListStorage.representation(xs),
          i) <> Tree.read_sequence(~F32,rest,(i + 1 : U32),ListStorage.representation(xs)) == _ <> enumerate(rest,
          1n+U32.to_nat(i),xs) : List<&2,F32>}
      %U32Successor.successor(i)(small_index(xs,i,proof,inside)) : {Tree.read(F32,ListStorage.representation(xs),
          i) <> Tree.read_sequence(~F32,rest,(i + 1 : U32),ListStorage.representation(xs)) == Tree.read(F32,
          ListStorage.representation(xs),i) <> enumerate(rest,_,xs) : List<&2,F32>}
      %readback(rest,xs,(i + 1 : U32),proof,next_room(xs,i,rest,proof,room)) : {Tree.read(F32,ListStorage.representation(xs),
          i) <> Tree.read_sequence(~F32,rest,(i + 1 : U32),ListStorage.representation(xs)) == Tree.read(F32,
          ListStorage.representation(xs),i) <> _ : List<&2,F32>}
      {==}

def shift(+n: Nat,+i: Nat,+h: F32,+tail: List<&2,F32>) -> {enumerate(n,1n+i,h <> tail) == enumerate(n,i,tail) : List<&2,F32>}:
  match n:
    case 0n:
      {==}
    case 1n+ +rest:
      %shift(rest,1n+i,h,tail) : {ConvSpec.at(i,tail) <> enumerate(rest,2n+i,h <> tail) == ConvSpec.at(i,tail) <> _ : List<&2,F32>}
      {==}

def enumerated(+xs: List<&2,F32>) -> {enumerate(ConvSpec.length(xs),0n,xs) == xs : List<&2,F32>}:
  match xs:
    case []:
      {==}
    case +h <> +tail:
      %Equal.sym(List<&2,F32>,enumerate(ConvSpec.length(tail),1n,h <> tail),enumerate(ConvSpec.length(tail),0n,tail),shift(ConvSpec.length(tail),
          0n,h,tail)) : {h <> _ == h <> tail : List<&2,F32>}
      %enumerated(tail) : {h <> enumerate(ConvSpec.length(tail),0n,tail) == h <> _ : List<&2,F32>}
      {==}

def sequence_evidence(+values: List<&2,F32>,+ready: ListArrayAllocation.Ready(values)) ->
  StorageLaws.SequenceEvidence<F32,ConvSpec.length(values),values>:
  StorageLaws.SequenceEvidence{ListStorage.representation(values),
    Equal.trans(List<&2,F32>,Tree.read_sequence(~F32,ConvSpec.length(values),0,ListStorage.representation(values)),
      enumerate(ConvSpec.length(values),0n,values),values,
      readback(ConvSpec.length(values),values,0,ready,Order.reflexive(ConvSpec.length(values))),enumerated(values)),
    EncodedRead.length_within_capacity(values,ready)}
