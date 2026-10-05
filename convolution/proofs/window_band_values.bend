# Rows [first, first + rows) of every input channel as the list a window
# problem reads. Band index i holds input index source(i).
import Base
import ../specification.bend as ConvSpec
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ./input_storage.bend as Inputs

type Band is Data:
  Band{values: List<&2,F32>,height: Nat,width: Nat,first: Nat,rows: Nat}

def input_values(band: Band) -> List<&2,F32>:
  Band{values,height,width,first,rows} = band
  values

def source(+band: Band,+index: Nat) -> Nat:
  Band{values,+height,+width,+first,+rows} = band
  +plane = Nat.mul(rows,width)
  Nat.add(Nat.mul(Nat.div(index,plane),Nat.mul(height,width)),Nat.add(Nat.mul(first,width),Nat.mod(index,plane)))

def value(+band: Band,+index: Nat) -> F32:
  ConvSpec.at(source(band,index),input_values(band))

def values(count: Nat,+index: Nat,+band: Band) -> List<&2,F32>:
  match count:
    case 0n: []
    case 1n+rest: value(band,index) <> values(rest,1n+index,band)

def length(+count: Nat,+index: Nat,+band: Band) -> {ConvSpec.length(values(count,index,band)) == count : Nat}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %length(rest,1n+index,band) : {1n+ConvSpec.length(values(rest,1n+index,band)) == 1n+_ : Nat}
      {==}

def get(+count: Nat,+index: Nat,+band: Band,+position: Nat,+inside: {Nat.is_lt(position,count) == True{} : Bool}) ->
  {List.get(&2,F32,values(count,index,band),position) == Some{value(band,Nat.add(index,position))} : Maybe<&2,F32>}:
  match count position:
    case 0n 0n:
      Empty.absurd({None{} == Some{value(band,Nat.add(index,0n))} : Maybe<&2,F32>},WordEncoding.false_true(inside))
    case 0n 1n+previous:
      Empty.absurd({None{} == Some{value(band,Nat.add(index,1n+previous))} : Maybe<&2,F32>},WordEncoding.false_true(inside))
    case 1n+rest 0n:
      %Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)) : {Some{value(band,index)} == Some{value(band,_)} : Maybe<&2,F32>}
      {==}
    case 1n+ +rest 1n+ +previous:
      %Equal.sym(Nat,Nat.add(index,1n+previous),1n+Nat.add(index,previous),Addition.successor_right(index,previous)) :
        {List.get(&2,F32,values(rest,1n+index,band),previous) == Some{value(band,_)} : Maybe<&2,F32>}
      get(rest,1n+index,band,previous,inside)

def at(+count: Nat,+band: Band,+position: Nat,+inside: {Nat.is_lt(position,count) == True{} : Bool}) ->
  {ConvSpec.at(position,values(count,0n,band)) == value(band,position) : F32}:
  %Equal.sym(F32,ConvSpec.at(position,values(count,0n,band)),Maybe.default(&2,F32,List.get(&2,F32,values(count,0n,band),position),0.0),
    Inputs.lookup_matches_list(position,values(count,0n,band))) : {_ == value(band,position) : F32}
  %Equal.sym(Maybe<&2,F32>,List.get(&2,F32,values(count,0n,band),position),Some{value(band,position)},get(count,0n,band,position,inside)) :
    {Maybe.default(&2,F32,_,0.0) == value(band,position) : F32}
  {==}

# A window tree whose leading reads are exactly the band.
def Contents(+count: Nat,+band: Band,+tree: Tree.Tree<F32>) -> Data:
  {Tree.read_sequence(~F32,count,0,tree) == values(count,0n,band) : List<&2,F32>}

def evidence(+count: Nat,+band: Band,+tree: Tree.Tree<F32>,+contents: Contents(count,band,tree),
  +room: {Nat.is_le(count,U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool}) ->
  StorageLaws.SequenceEvidence<F32,ConvSpec.length(values(count,0n,band)),values(count,0n,band)>:
  StorageLaws.SequenceEvidence{tree,
    %Equal.sym(Nat,ConvSpec.length(values(count,0n,band)),count,length(count,0n,band)) :
      {Tree.read_sequence(~F32,_,0,tree) == values(count,0n,band) : List<&2,F32>}
    contents,
    %Equal.sym(Nat,ConvSpec.length(values(count,0n,band)),count,length(count,0n,band)) :
      {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool}
    room}

def read(+count: Nat,+band: Band,+tree: Tree.Tree<F32>,+contents: Contents(count,band,tree),
  +room: {Nat.is_le(count,U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},+position: Nat,
  +inside: {Nat.is_lt(position,count) == True{} : Bool}) ->
  {Tree.read(F32,tree,U32.from_nat(position)) == value(band,position) : F32}:
  %Equal.sym(F32,Tree.read(F32,tree,U32.from_nat(position)),Maybe.default(&2,F32,List.get(&2,F32,values(count,0n,band),position),0.0),
    StorageLaws.stored_sequence_element(~F32,0.0,count,values(count,0n,band),StorageLaws.SequenceEvidence{tree,contents,room},position,inside)) :
    {_ == value(band,position) : F32}
  %Equal.sym(Maybe<&2,F32>,List.get(&2,F32,values(count,0n,band),position),Some{value(band,position)},get(count,0n,band,position,inside)) :
    {Maybe.default(&2,F32,_,0.0) == value(band,position) : F32}
  {==}

# A tree whose reads agree with a band at every index below the limit holds
# the band as its leading read sequence.
def listed(~Context: Data,~tree_of: Context -> Tree.Tree<F32>,~band_of: Context -> Band,~limit_of: Context -> Nat,
  ~reads: @+context: Context -> @+index: Nat -> @+inside: {Nat.is_lt(index,limit_of(context)) == True{} : Bool} ->
    {Tree.read(F32,tree_of(context),U32.from_nat(index)) == value(band_of(context),index) : F32},
  +count: Nat,+index: Nat,+context: Context,+room: {Nat.is_le(Nat.add(index,count),limit_of(context)) == True{} : Bool}) ->
  {Tree.read_sequence(~F32,count,U32.from_nat(index),tree_of(context)) == values(count,index,band_of(context)) : List<&2,F32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +tree = tree_of(context)
      +band = band_of(context)
      %Equal.sym(F32,Tree.read(F32,tree,U32.from_nat(index)),value(band,index),reads(context,index,Order.room_index(index,rest,limit_of(context),room))) :
        {_ <> Tree.read_sequence(~F32,rest,(U32.from_nat(index) + 1 : U32),tree) == values(1n+rest,index,band) : List<&2,F32>}
      %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.add_one(U32.from_nat(index))) :
        {value(band,index) <> Tree.read_sequence(~F32,rest,_,tree) == values(1n+rest,index,band) : List<&2,F32>}
      Equal.cong(List<&2,F32>,List<&2,F32>,tail => value(band,index) <> tail,Tree.read_sequence(~F32,rest,U32.from_nat(1n+index),tree),values(rest,1n+index,band),
        listed(~Context,~tree_of,~band_of,~limit_of,~reads,rest,1n+index,context,Order.advance_room(index,rest,limit_of(context),room)))
