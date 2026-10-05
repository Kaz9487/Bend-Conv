import ../../lib/proofs/storage_shape.bend as Shape
# Native get with a logical Nat index. Ready is proved from actual guards in
# SUPPORTED; encoded and index_bound prevent both U32 and Array-mask aliasing.
import Base
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ./list_array_storage.bend as ListStorage
import ./list_array_allocation.bend as ListArrayAllocation
import ../specification.bend as ConvSpec
import ../scalar_kernel.bend as ScalarConv
import ./dense_array_initialization.bend as DenseArrayInitialization
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/kernel_spatial_packing.bend as Packing

def get(+xs: List<&2,F32>,+address: U32,+index: Nat,+proof: ListArrayAllocation.Ready(xs),encoded: {address == U32.from_nat(index) : U32},
  index_bound: {Nat.is_lt(index,ConvSpec.length(xs)) == True{} : Bool}) -> {Array.get(F32,ScalarConv.pack(xs),address) == (ScalarConv.pack(xs),
    ConvSpec.at(index,xs)) : Array<F32> & F32}:
  %Equal.sym(Array<F32>,ScalarConv.pack(xs),Tree.pack(F32,ListStorage.representation(xs)),
    ListStorage.packed(xs)) : {Array.get(F32,_,address) == (_,ConvSpec.at(index,xs)) : Array<F32> & F32}
  %ListArrayAllocation.read_encoded(xs,address,index,proof,encoded,index_bound) : {Array.get(F32,Tree.pack(F32,
        ListStorage.representation(xs)),address) == (Tree.pack(F32,ListStorage.representation(xs)),_) : Array<F32> & F32}
  StorageLaws.read(F32,ListStorage.representation(xs),address)

# The scalar backend checks both logical length and physical capacity on every
# read. Discharge those checks from allocation evidence, not an assumed read.
def length_within_capacity(+xs: List<&2,F32>,proof: ListArrayAllocation.Ready(xs)) ->
  {Nat.is_le(ConvSpec.length(xs),U32.to_nat(Tree.capacity(F32,ListStorage.representation(xs)))) == True{} : Bool}:
  Shape.Both{limit,room} = proof
  %Equal.sym(U32,Tree.capacity(F32,ListStorage.representation(xs)),Tree.capacity(F32,ListArrayAllocation.initial(xs)),
    DenseArrayInitialization.capacity(xs,0,ListArrayAllocation.initial(xs),ListArrayAllocation.depth(xs),
      Shape.fresh(F32,ListArrayAllocation.depth(xs),0.0))) : {Nat.is_le(ConvSpec.length(xs),U32.to_nat(_)) == True{} : Bool}
  room

def bounded_get(+xs: List<&2,F32>,+address: U32,+index: Nat,+logical_length: U32,+proof: ListArrayAllocation.Ready(xs),
  +encoded: {address == U32.from_nat(index) : U32},
  +index_bound: {Nat.is_lt(index,ConvSpec.length(xs)) == True{} : Bool},
  +length_exact: {U32.to_nat(logical_length) == ConvSpec.length(xs) : Nat}) ->
  {Storage.read_or(F32,0.0,ScalarConv.pack(xs),address,logical_length) == (ScalarConv.pack(xs),ConvSpec.at(index,xs)) : Array<F32> & F32}:
  +address_exact = ListArrayAllocation.expression_exact(xs,address,index,proof,encoded,index_bound)
  +inside = U32Division.nat_lt_native(address,logical_length,
    %Equal.sym(Nat,U32.to_nat(address),index,address_exact) : {Nat.is_lt(_,U32.to_nat(logical_length)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(logical_length),ConvSpec.length(xs),length_exact) : {Nat.is_lt(index,_) == True{} : Bool}
    index_bound)
  +room = U32Division.nat_le_native(logical_length,Tree.capacity(F32,ListStorage.representation(xs)),
    %Equal.sym(Nat,U32.to_nat(logical_length),ConvSpec.length(xs),length_exact) :
      {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,ListStorage.representation(xs)))) == True{} : Bool}
    length_within_capacity(xs,proof))
  Equal.trans(Array<F32> & F32,Storage.read_or(F32,0.0,ScalarConv.pack(xs),address,logical_length),Array.get(F32,ScalarConv.pack(xs),address),
    (ScalarConv.pack(xs),ConvSpec.at(index,xs)),
    %Equal.sym(Array<F32>,ScalarConv.pack(xs),Tree.pack(F32,ListStorage.representation(xs)),ListStorage.packed(xs)) :
      {Storage.read_or(F32,0.0,_,address,logical_length) == Array.get(F32,_,address) : Array<F32> & F32}
    StorageLaws.bounded_read_is_get(~F32,0.0,ListStorage.representation(xs),address,logical_length,inside,room),
    get(xs,address,index,proof,encoded,index_bound))

# Padding is independent of representation: outside reads return zero, while
# inside reads discharge the actual guarded-read checks against the allocation.
def sample_matches_masked_get(+inside: Bool,+xs: List<&2,F32>,+address: U32,+index: Nat,+logical_length: U32,
  +proof: ListArrayAllocation.Ready(xs),+encoded: {address == U32.from_nat(index) : U32},
  in_bounds: {inside == True{} : Bool} -> {Nat.is_lt(index,ConvSpec.length(xs)) == True{} : Bool},
  length_exact: {U32.to_nat(logical_length) == ConvSpec.length(xs) : Nat}) ->
  {ScalarConv.sample(~Array<F32>,~Storage.read_or(F32,0.0),inside,ScalarConv.pack(xs),address,logical_length) ==
    Packing.masked(inside,Array.get(F32,ScalarConv.pack(xs),address)) : Array<F32> & F32}:
  match inside:
    case False{}:
      %Equal.sym(Array<F32>,ScalarConv.pack(xs),Tree.pack(F32,ListStorage.representation(xs)),ListStorage.packed(xs)) :
        {(_,0.0) == Packing.masked(False{},Array.get(F32,_,address)) : Array<F32> & F32}
      %Equal.sym(Array<F32> & F32,Array.get(F32,Tree.pack(F32,ListStorage.representation(xs)),address),
        (Tree.pack(F32,ListStorage.representation(xs)),Tree.read(F32,ListStorage.representation(xs),address)),
        StorageLaws.read(F32,ListStorage.representation(xs),address)) :
        {(Tree.pack(F32,ListStorage.representation(xs)),0.0) == Packing.masked(False{},_) : Array<F32> & F32}
      {==}
    case True{}:
      +bound = in_bounds({==})
      %Equal.sym(Array<F32> & F32,Array.get(F32,ScalarConv.pack(xs),address),(ScalarConv.pack(xs),ConvSpec.at(index,xs)),get(xs,address,index,proof,encoded,bound)) :
        {Storage.read_or(F32,0.0,ScalarConv.pack(xs),address,logical_length) == Packing.masked(True{},_) : Array<F32> & F32}
      bounded_get(xs,address,index,logical_length,proof,encoded,bound,length_exact)
