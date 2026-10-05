# Reading a copied window: each channel plane is one contiguous write
# sequence, so a read inside plane c returns the plane's source element and
# later planes, written above it, leave it unchanged.
import Base
import ../specification.bend as ConvSpec
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/contiguous_copy.bend as Contiguous
import ../../lib/proofs/kernel_window_copy.bend as WindowCopy
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/word_encoding.bend as WordEncoding
import ./dense_array_initialization.bend as DenseArrayInitialization
import ../../lib/proofs/nat_interval.bend as Interval
import ./input_storage.bend as Inputs
import ./window_samples.bend as Samples

def plane_values(+channel: U32,+count: U32,+source_stride: U32,+source_start: U32,+input: Tree.Tree<F32>) -> List<&2,F32>:
  Tree.read_sequence(~F32,U32.to_nat(count),(channel * source_stride + source_start : U32),input)

# The planes as nested write sequences, first plane innermost.
def planes(remaining: Nat,+channel: U32,+count: U32,+source_stride: U32,+source_start: U32,+target_stride: U32,+input: Tree.Tree<F32>,
  tree: Tree.Tree<F32>) -> Tree.Tree<F32>:
  match remaining:
    case 0n: tree
    case 1n+rest:
      planes(rest,(channel + 1 : U32),count,source_stride,source_start,target_stride,input,
        Tree.write_sequence(~F32,plane_values(channel,count,source_stride,source_start,input),(channel * target_stride : U32),tree))

def planes_exact(+remaining: Nat,+channel: U32,+count: U32,+source_stride: U32,+source_start: U32,+target_stride: U32,+input: Tree.Tree<F32>,
  +tree: Tree.Tree<F32>) ->
  {Writes.pure(F32,WindowCopy.writes(remaining,channel,count,source_stride,source_start,target_stride,input),tree) ==
    planes(remaining,channel,count,source_stride,source_start,target_stride,input,tree) : Tree.Tree<F32>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +here = WindowCopy.plane_writes(channel,count,source_stride,source_start,target_stride,input)
      +later = WindowCopy.writes(rest,(channel + 1 : U32),count,source_stride,source_start,target_stride,input)
      +source = (channel * source_stride + source_start : U32)
      +target = (channel * target_stride : U32)
      +written = Tree.write_sequence(~F32,plane_values(channel,count,source_stride,source_start,input),target,tree)
      %Equal.sym(Tree.Tree<F32>,Writes.pure(F32,List.append(&2,Writes.Assignment<F32>,here,later),tree),Writes.pure(F32,later,Writes.pure(F32,here,tree)),
        Writes.concatenated_writes(F32,here,later,tree)) :
        {_ == planes(rest,(channel + 1 : U32),count,source_stride,source_start,target_stride,input,written) : Tree.Tree<F32>}
      %Equal.sym(Tree.Tree<F32>,Writes.pure(F32,here,tree),
        Tree.write_sequence(~F32,Tree.read_sequence(~F32,U32.to_nat(count),(source + 0 : U32),input),(target + 0 : U32),tree),
        Contiguous.sequence(~F32,U32.to_nat(count),input,0,source,target,tree)) :
        {Writes.pure(F32,later,_) == planes(rest,(channel + 1 : U32),count,source_stride,source_start,target_stride,input,written) : Tree.Tree<F32>}
      %Equal.sym(U32,(source + 0 : U32),source,Arithmetic.add_zero(source)) :
        {Writes.pure(F32,later,Tree.write_sequence(~F32,Tree.read_sequence(~F32,U32.to_nat(count),_,input),(target + 0 : U32),tree)) ==
          planes(rest,(channel + 1 : U32),count,source_stride,source_start,target_stride,input,written) : Tree.Tree<F32>}
      %Equal.sym(U32,(target + 0 : U32),target,Arithmetic.add_zero(target)) :
        {Writes.pure(F32,later,Tree.write_sequence(~F32,plane_values(channel,count,source_stride,source_start,input),_,tree)) ==
          planes(rest,(channel + 1 : U32),count,source_stride,source_start,target_stride,input,written) : Tree.Tree<F32>}
      planes_exact(rest,(channel + 1 : U32),count,source_stride,source_start,target_stride,input,written)

def successor_le(+value: Nat,+rest: Nat) -> {Nat.is_le(1n+value,Nat.add(value,1n+rest)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(value,1n+rest),1n+Nat.add(value,rest),Addition.successor_right(value,rest)) : {Nat.is_le(1n+value,_) == True{} : Bool}
  Order.le_add(value,rest)

def next_channel(+channel: U32,+rest: Nat,+limit: U32,+limit_ok: {U32.is_le(limit,2147483648) == True{} : Bool},
  +channels_fit: {Nat.is_le(Nat.add(U32.to_nat(channel),1n+rest),U32.to_nat(limit)) == True{} : Bool}) ->
  {U32.to_nat((channel + 1 : U32)) == 1n+U32.to_nat(channel) : Nat}:
  Equal.trans(Nat,U32.to_nat((channel + 1 : U32)),Nat.add(U32.to_nat(channel),1n),1n+U32.to_nat(channel),
    Arithmetic.add(channel,1,limit,limit_ok,
      Order.transitive(Nat.add(U32.to_nat(channel),1n),Nat.add(U32.to_nat(channel),1n+rest),U32.to_nat(limit),
        %Equal.sym(Nat,Nat.add(U32.to_nat(channel),1n),1n+U32.to_nat(channel),Addition.successor_width(U32.to_nat(channel))) :
          {Nat.is_le(_,Nat.add(U32.to_nat(channel),1n+rest)) == True{} : Bool}
        successor_le(U32.to_nat(channel),rest),channels_fit)),
    Addition.successor_width(U32.to_nat(channel)))

def plane_length(+channel: U32,+count: U32,+source_stride: U32,+source_start: U32,+input: Tree.Tree<F32>) ->
  {ConvSpec.length(plane_values(channel,count,source_stride,source_start,input)) == U32.to_nat(count) : Nat}:
  Equal.trans(Nat,ConvSpec.length(plane_values(channel,count,source_stride,source_start,input)),
    List.length(&2,F32,plane_values(channel,count,source_stride,source_start,input)),U32.to_nat(count),
    Inputs.length_matches_list(plane_values(channel,count,source_stride,source_start,input)),
    StorageLaws.sequence_length(~F32,U32.to_nat(count),(channel * source_stride + source_start : U32),input))

def plane_start(+channel: U32,+count: U32,+rest: Nat,+limit: U32,+limit_ok: {U32.is_le(limit,2147483648) == True{} : Bool},
  +bound: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(channel),1n+rest),U32.to_nat(count)),U32.to_nat(limit)) == True{} : Bool}) ->
  {U32.to_nat((channel * count : U32)) == Nat.mul(U32.to_nat(channel),U32.to_nat(count)) : Nat}:
  Arithmetic.multiply(channel,count,limit,limit_ok,
    Order.transitive(Nat.mul(U32.to_nat(channel),U32.to_nat(count)),Nat.mul(1n+U32.to_nat(channel),U32.to_nat(count)),U32.to_nat(limit),
      Samples.le_after(U32.to_nat(count),Nat.mul(U32.to_nat(channel),U32.to_nat(count))),
      Order.transitive(Nat.mul(1n+U32.to_nat(channel),U32.to_nat(count)),Nat.mul(Nat.add(U32.to_nat(channel),1n+rest),U32.to_nat(count)),U32.to_nat(limit),
        RowMajor.mul_le(1n+U32.to_nat(channel),Nat.add(U32.to_nat(channel),1n+rest),U32.to_nat(count),successor_le(U32.to_nat(channel),rest)),bound)))

# Planes at or after this channel start above the query, so they keep its value.
def outside_planes(+remaining: Nat,+channel: U32,+count: U32,+source_stride: U32,+source_start: U32,+input: Tree.Tree<F32>,
  +tree: Tree.Tree<F32>,+depth: Nat,+query: U32,+limit: U32,+limit_ok: {U32.is_le(limit,2147483648) == True{} : Bool},
  +balanced: Shape.Balanced(F32,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +channels_fit: {Nat.is_le(Nat.add(U32.to_nat(channel),remaining),U32.to_nat(limit)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(channel),remaining),U32.to_nat(count)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},
  +whole_fits: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(channel),remaining),U32.to_nat(count)),U32.to_nat(limit)) == True{} : Bool},
  +query_inside: {U32.is_lt(query,Tree.capacity(F32,tree)) == True{} : Bool},
  +below: {Nat.is_lt(U32.to_nat(query),Nat.mul(U32.to_nat(channel),U32.to_nat(count))) == True{} : Bool}) ->
  {Tree.read(F32,planes(remaining,channel,count,source_stride,source_start,count,input,tree),query) == Tree.read(F32,tree,query) : F32}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +channel_nat = U32.to_nat(channel)
      +count_nat = U32.to_nat(count)
      +capacity = Tree.capacity(F32,tree)
      +start = (channel * count : U32)
      +values = plane_values(channel,count,source_stride,source_start,input)
      +written = Tree.write_sequence(~F32,values,start,tree)
      +next = (channel + 1 : U32)
      +start_exact = plane_start(channel,count,rest,limit,limit_ok,whole_fits)
      +length_exact = plane_length(channel,count,source_stride,source_start,input)
      +grow = RowMajor.mul_le(1n+channel_nat,Nat.add(channel_nat,1n+rest),count_nat,successor_le(channel_nat,rest))
      +plane_room = {%Equal.sym(Nat,U32.to_nat(start),Nat.mul(channel_nat,count_nat),start_exact) :
          {Nat.is_le(Nat.add(_,ConvSpec.length(values)),U32.to_nat(capacity)) == True{} : Bool}
        %Equal.sym(Nat,ConvSpec.length(values),count_nat,length_exact) : {Nat.is_le(Nat.add(Nat.mul(channel_nat,count_nat),_),U32.to_nat(capacity)) == True{} : Bool}
        %Order.commutative(count_nat,Nat.mul(channel_nat,count_nat)) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
        Order.transitive(Nat.mul(1n+channel_nat,count_nat),Nat.mul(Nat.add(channel_nat,1n+rest),count_nat),U32.to_nat(capacity),grow,room) :
        {Nat.is_le(Nat.add(U32.to_nat(start),ConvSpec.length(values)),U32.to_nat(capacity)) == True{} : Bool}}
      +outside = {Interval.Before{%Equal.sym(Nat,U32.to_nat(start),Nat.mul(channel_nat,count_nat),start_exact) : {Nat.is_lt(U32.to_nat(query),_) == True{} : Bool}
        below} : Interval.Outside<U32.to_nat(start),ConvSpec.length(values),U32.to_nat(query)>}
      +written_capacity = DenseArrayInitialization.capacity(values,start,tree,depth,balanced)
      +next_exact = next_channel(channel,rest,limit,limit_ok,channels_fit)
      +later = outside_planes(rest,next,count,source_stride,source_start,input,written,depth,query,limit,limit_ok,DenseArrayInitialization.balanced(values,start,tree,depth,balanced),depth_ok,
        %Equal.sym(Nat,U32.to_nat(next),1n+channel_nat,next_exact) : {Nat.is_le(Nat.add(_,rest),U32.to_nat(limit)) == True{} : Bool}
        %Addition.successor_right(channel_nat,rest) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool}
        channels_fit,
        %Equal.sym(U32,Tree.capacity(F32,written),capacity,written_capacity) : {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(next),rest),count_nat),U32.to_nat(_)) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(next),1n+channel_nat,next_exact) : {Nat.is_le(Nat.mul(Nat.add(_,rest),count_nat),U32.to_nat(capacity)) == True{} : Bool}
        %Addition.successor_right(channel_nat,rest) : {Nat.is_le(Nat.mul(_,count_nat),U32.to_nat(capacity)) == True{} : Bool}
        room,
        %Equal.sym(Nat,U32.to_nat(next),1n+channel_nat,next_exact) : {Nat.is_le(Nat.mul(Nat.add(_,rest),count_nat),U32.to_nat(limit)) == True{} : Bool}
        %Addition.successor_right(channel_nat,rest) : {Nat.is_le(Nat.mul(_,count_nat),U32.to_nat(limit)) == True{} : Bool}
        whole_fits,
        %Equal.sym(U32,Tree.capacity(F32,written),capacity,written_capacity) : {U32.is_lt(query,_) == True{} : Bool}
        query_inside,
        %Equal.sym(Nat,U32.to_nat(next),1n+channel_nat,next_exact) : {Nat.is_lt(U32.to_nat(query),Nat.mul(_,count_nat)) == True{} : Bool}
        Samples.lt_le(U32.to_nat(query),Nat.mul(channel_nat,count_nat),Nat.mul(1n+channel_nat,count_nat),below,Samples.le_after(count_nat,Nat.mul(channel_nat,count_nat))))
      Equal.trans(F32,Tree.read(F32,planes(rest,next,count,source_stride,source_start,count,input,written),query),Tree.read(F32,written,query),
        Tree.read(F32,tree,query),later,DenseArrayInitialization.read_filled_outside(values,start,tree,depth,query,balanced,depth_ok,plane_room,query_inside,outside))

def before_distinct(+smaller: Nat,+larger: Nat,+ordered: {Nat.is_le(smaller,larger) == True{} : Bool},
  +distinct: {Nat.is_eq(smaller,larger) == False{} : Bool}) -> {Nat.is_lt(smaller,larger) == True{} : Bool}:
  match smaller larger:
    case 0n 0n: Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},distinct)))
    case 0n 1n+rest: {==}
    case 1n+rest 0n: Empty.absurd({Nat.is_lt(1n+rest,0n) == True{} : Bool},WordEncoding.false_true(ordered))
    case 1n+ +left 1n+ +right: before_distinct(left,right,ordered,distinct)

def no_plane(+channel: Nat,+target: Nat,+after: {Nat.is_le(channel,target) == True{} : Bool},
  +inside: {Nat.is_lt(target,Nat.add(channel,0n)) == True{} : Bool}) -> {Nat.is_eq(channel,channel) == False{} : Bool}:
  Order.less_not_equal(channel,channel,
    Order.le_lt_trans(channel,target,channel,after,
      %Addition.zero_right(channel) : {Nat.is_lt(target,_) == True{} : Bool}
      inside))

# The read of a query in plane target returns the source element of that plane.
def inside_planes(+remaining: Nat,+same: Bool,+channel: U32,+count: U32,+source_stride: U32,+source_start: U32,+input: Tree.Tree<F32>,
  +tree: Tree.Tree<F32>,+depth: Nat,+query: U32,+limit: U32,+limit_ok: {U32.is_le(limit,2147483648) == True{} : Bool},
  +balanced: Shape.Balanced(F32,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +channels_fit: {Nat.is_le(Nat.add(U32.to_nat(channel),remaining),U32.to_nat(limit)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(channel),remaining),U32.to_nat(count)),U32.to_nat(Tree.capacity(F32,tree))) == True{} : Bool},
  +whole_fits: {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(channel),remaining),U32.to_nat(count)),U32.to_nat(limit)) == True{} : Bool},
  +query_inside: {U32.is_lt(query,Tree.capacity(F32,tree)) == True{} : Bool},
  +target: Nat,+column: Nat,+same_exact: {Nat.is_eq(U32.to_nat(channel),target) == same : Bool},
  +query_exact: {U32.to_nat(query) == Nat.add(Nat.mul(target,U32.to_nat(count)),column) : Nat},
  +column_inside: {Nat.is_lt(column,U32.to_nat(count)) == True{} : Bool},
  +after: {Nat.is_le(U32.to_nat(channel),target) == True{} : Bool},
  +target_inside: {Nat.is_lt(target,Nat.add(U32.to_nat(channel),remaining)) == True{} : Bool},
  +source_fits: {Nat.is_le(Nat.add(Nat.add(Nat.mul(target,U32.to_nat(source_stride)),U32.to_nat(source_start)),column),U32.to_nat(limit)) == True{} : Bool}) ->
  {Tree.read(F32,planes(remaining,channel,count,source_stride,source_start,count,input,tree),query) ==
    Tree.read(F32,input,U32.from_nat(Nat.add(Nat.add(Nat.mul(target,U32.to_nat(source_stride)),U32.to_nat(source_start)),column))) : F32}:
  match remaining same:
    case 0n True{}:
      Empty.absurd({Tree.read(F32,tree,query) == Tree.read(F32,input,U32.from_nat(Nat.add(Nat.add(Nat.mul(target,U32.to_nat(source_stride)),U32.to_nat(source_start)),column))) : F32},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(U32.to_nat(channel),U32.to_nat(channel)),True{},
          Equal.sym(Bool,Nat.is_eq(U32.to_nat(channel),U32.to_nat(channel)),False{},no_plane(U32.to_nat(channel),target,after,target_inside)),
          Order.equal_reflexive(U32.to_nat(channel)))))
    case 0n False{}:
      Empty.absurd({Tree.read(F32,tree,query) == Tree.read(F32,input,U32.from_nat(Nat.add(Nat.add(Nat.mul(target,U32.to_nat(source_stride)),U32.to_nat(source_start)),column))) : F32},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(U32.to_nat(channel),U32.to_nat(channel)),True{},
          Equal.sym(Bool,Nat.is_eq(U32.to_nat(channel),U32.to_nat(channel)),False{},no_plane(U32.to_nat(channel),target,after,target_inside)),
          Order.equal_reflexive(U32.to_nat(channel)))))
    case 1n+ +rest True{}:
      +channel_nat = U32.to_nat(channel)
      +count_nat = U32.to_nat(count)
      +capacity = Tree.capacity(F32,tree)
      +equal = Order.equal_of_is_eq(channel_nat,target,same_exact)
      +start = (channel * count : U32)
      +source = (channel * source_stride + source_start : U32)
      +values = plane_values(channel,count,source_stride,source_start,input)
      +written = Tree.write_sequence(~F32,values,start,tree)
      +next = (channel + 1 : U32)
      +start_exact = plane_start(channel,count,rest,limit,limit_ok,whole_fits)
      +length_exact = plane_length(channel,count,source_stride,source_start,input)
      +grow = RowMajor.mul_le(1n+channel_nat,Nat.add(channel_nat,1n+rest),count_nat,successor_le(channel_nat,rest))
      +plane_room = {%Equal.sym(Nat,U32.to_nat(start),Nat.mul(channel_nat,count_nat),start_exact) :
          {Nat.is_le(Nat.add(_,ConvSpec.length(values)),U32.to_nat(capacity)) == True{} : Bool}
        %Equal.sym(Nat,ConvSpec.length(values),count_nat,length_exact) : {Nat.is_le(Nat.add(Nat.mul(channel_nat,count_nat),_),U32.to_nat(capacity)) == True{} : Bool}
        %Order.commutative(count_nat,Nat.mul(channel_nat,count_nat)) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
        Order.transitive(Nat.mul(1n+channel_nat,count_nat),Nat.mul(Nat.add(channel_nat,1n+rest),count_nat),U32.to_nat(capacity),grow,room) :
        {Nat.is_le(Nat.add(U32.to_nat(start),ConvSpec.length(values)),U32.to_nat(capacity)) == True{} : Bool}}
      +written_capacity = DenseArrayInitialization.capacity(values,start,tree,depth,balanced)
      +next_exact = next_channel(channel,rest,limit,limit_ok,channels_fit)
      +later = outside_planes(rest,next,count,source_stride,source_start,input,written,depth,query,limit,limit_ok,DenseArrayInitialization.balanced(values,start,tree,depth,balanced),depth_ok,
        %Equal.sym(Nat,U32.to_nat(next),1n+channel_nat,next_exact) : {Nat.is_le(Nat.add(_,rest),U32.to_nat(limit)) == True{} : Bool}
        %Addition.successor_right(channel_nat,rest) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool}
        channels_fit,
        %Equal.sym(U32,Tree.capacity(F32,written),capacity,written_capacity) : {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(next),rest),count_nat),U32.to_nat(_)) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(next),1n+channel_nat,next_exact) : {Nat.is_le(Nat.mul(Nat.add(_,rest),count_nat),U32.to_nat(capacity)) == True{} : Bool}
        %Addition.successor_right(channel_nat,rest) : {Nat.is_le(Nat.mul(_,count_nat),U32.to_nat(capacity)) == True{} : Bool}
        room,
        %Equal.sym(Nat,U32.to_nat(next),1n+channel_nat,next_exact) : {Nat.is_le(Nat.mul(Nat.add(_,rest),count_nat),U32.to_nat(limit)) == True{} : Bool}
        %Addition.successor_right(channel_nat,rest) : {Nat.is_le(Nat.mul(_,count_nat),U32.to_nat(limit)) == True{} : Bool}
        whole_fits,
        %Equal.sym(U32,Tree.capacity(F32,written),capacity,written_capacity) : {U32.is_lt(query,_) == True{} : Bool}
        query_inside,
        %Equal.sym(Nat,U32.to_nat(next),1n+channel_nat,next_exact) : {Nat.is_lt(U32.to_nat(query),Nat.mul(_,count_nat)) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(query),Nat.add(Nat.mul(target,U32.to_nat(count)),column),query_exact) : {Nat.is_lt(_,Nat.mul(1n+channel_nat,count_nat)) == True{} : Bool}
        %equal : {Nat.is_lt(Nat.add(Nat.mul(_,count_nat),column),Nat.mul(1n+channel_nat,count_nat)) == True{} : Bool}
        %Order.commutative(column,Nat.mul(channel_nat,count_nat)) : {Nat.is_lt(_,Nat.mul(1n+channel_nat,count_nat)) == True{} : Bool}
        Order.add_lt(column,count_nat,Nat.mul(channel_nat,count_nat),column_inside))
      +filled = DenseArrayInitialization.read_filled_range(values,start,tree,depth,query,column,balanced,depth_ok,
        %Equal.sym(Nat,U32.to_nat(start),Nat.mul(channel_nat,count_nat),start_exact) : {U32.to_nat(query) == Nat.add(_,column) : Nat}
        %Equal.sym(Nat,channel_nat,target,equal) : {U32.to_nat(query) == Nat.add(Nat.mul(_,count_nat),column) : Nat}
        query_exact,
        plane_room,
        %Equal.sym(Nat,ConvSpec.length(values),count_nat,length_exact) : {Nat.is_lt(column,_) == True{} : Bool}
        column_inside)
      +element = Equal.trans(F32,ConvSpec.at(column,values),Maybe.default(&2,F32,List.get(&2,F32,values,column),0.0),
        Tree.read(F32,input,(source + U32.from_nat(column) : U32)),Inputs.lookup_matches_list(column,values),
        Equal.cong(Maybe<&2,F32>,F32,found => Maybe.default(&2,F32,found,0.0),List.get(&2,F32,values,column),Some{Tree.read(F32,input,(source + U32.from_nat(column) : U32))},
          StorageLaws.sequence_element(~F32,count_nat,source,input,column,column_inside)))
      +stride_nat = U32.to_nat(source_stride)
      +start_nat = U32.to_nat(source_start)
      +scaled = Nat.mul(channel_nat,stride_nat)
      +scaled_fits = {%Equal.sym(Nat,channel_nat,target,equal) : {Nat.is_le(Nat.mul(_,stride_nat),U32.to_nat(limit)) == True{} : Bool}
        Order.transitive(Nat.mul(target,stride_nat),Nat.add(Nat.mul(target,stride_nat),start_nat),U32.to_nat(limit),Order.le_add(Nat.mul(target,stride_nat),start_nat),
          Order.transitive(Nat.add(Nat.mul(target,stride_nat),start_nat),Nat.add(Nat.add(Nat.mul(target,stride_nat),start_nat),column),U32.to_nat(limit),
            Order.le_add(Nat.add(Nat.mul(target,stride_nat),start_nat),column),source_fits)) : {Nat.is_le(scaled,U32.to_nat(limit)) == True{} : Bool}}
      +scaled_exact = Arithmetic.multiply(channel,source_stride,limit,limit_ok,scaled_fits)
      +source_exact = {Equal.trans(Nat,U32.to_nat(source),Nat.add(U32.to_nat((channel * source_stride : U32)),start_nat),Nat.add(Nat.mul(target,stride_nat),start_nat),
          Arithmetic.add((channel * source_stride : U32),source_start,limit,limit_ok,
            %Equal.sym(Nat,U32.to_nat((channel * source_stride : U32)),Nat.mul(channel_nat,stride_nat),scaled_exact) : {Nat.is_le(Nat.add(_,start_nat),U32.to_nat(limit)) == True{} : Bool}
            %Equal.sym(Nat,channel_nat,target,equal) : {Nat.is_le(Nat.add(Nat.mul(_,stride_nat),start_nat),U32.to_nat(limit)) == True{} : Bool}
            Order.transitive(Nat.add(Nat.mul(target,stride_nat),start_nat),Nat.add(Nat.add(Nat.mul(target,stride_nat),start_nat),column),U32.to_nat(limit),
              Order.le_add(Nat.add(Nat.mul(target,stride_nat),start_nat),column),source_fits)),
          %Equal.sym(Nat,U32.to_nat((channel * source_stride : U32)),Nat.mul(channel_nat,stride_nat),scaled_exact) : {Nat.add(_,start_nat) == Nat.add(Nat.mul(target,stride_nat),start_nat) : Nat}
          %Equal.sym(Nat,channel_nat,target,equal) : {Nat.add(Nat.mul(_,stride_nat),start_nat) == Nat.add(Nat.mul(target,stride_nat),start_nat) : Nat}
          {==}) : {U32.to_nat(source) == Nat.add(Nat.mul(target,stride_nat),start_nat) : Nat}}
      +column_exact = Bounds.roundtrip(column,limit,limit_ok,
        Order.transitive(column,Nat.add(Nat.add(Nat.mul(target,stride_nat),start_nat),column),U32.to_nat(limit),
          Samples.le_after(Nat.add(Nat.mul(target,stride_nat),start_nat),column),source_fits))
      +address = {Equal.trans(U32,(source + U32.from_nat(column) : U32),U32.from_nat(Nat.add(U32.to_nat(source),U32.to_nat(U32.from_nat(column)))),
          U32.from_nat(Nat.add(Nat.add(Nat.mul(target,stride_nat),start_nat),column)),
          Arithmetic.native_add(source,U32.from_nat(column)),
          %Equal.sym(Nat,U32.to_nat(source),Nat.add(Nat.mul(target,stride_nat),start_nat),source_exact) : {U32.from_nat(Nat.add(_,U32.to_nat(U32.from_nat(column)))) == U32.from_nat(Nat.add(Nat.add(Nat.mul(target,stride_nat),start_nat),column)) : U32}
          %Equal.sym(Nat,U32.to_nat(U32.from_nat(column)),column,column_exact) : {U32.from_nat(Nat.add(Nat.add(Nat.mul(target,stride_nat),start_nat),_)) == U32.from_nat(Nat.add(Nat.add(Nat.mul(target,stride_nat),start_nat),column)) : U32}
          {==}) : {(source + U32.from_nat(column) : U32) == U32.from_nat(Nat.add(Nat.add(Nat.mul(target,stride_nat),start_nat),column)) : U32}}
      Equal.trans(F32,Tree.read(F32,planes(rest,next,count,source_stride,source_start,count,input,written),query),Tree.read(F32,written,query),
        Tree.read(F32,input,U32.from_nat(Nat.add(Nat.add(Nat.mul(target,stride_nat),start_nat),column))),later,
        Equal.trans(F32,Tree.read(F32,written,query),ConvSpec.at(column,values),Tree.read(F32,input,U32.from_nat(Nat.add(Nat.add(Nat.mul(target,stride_nat),start_nat),column))),
          filled,
          Equal.trans(F32,ConvSpec.at(column,values),Tree.read(F32,input,(source + U32.from_nat(column) : U32)),
            Tree.read(F32,input,U32.from_nat(Nat.add(Nat.add(Nat.mul(target,stride_nat),start_nat),column))),element,
            Equal.cong(U32,F32,index => Tree.read(F32,input,index),(source + U32.from_nat(column) : U32),
              U32.from_nat(Nat.add(Nat.add(Nat.mul(target,stride_nat),start_nat),column)),address))))
    case 1n+ +rest False{}:
      +channel_nat = U32.to_nat(channel)
      +count_nat = U32.to_nat(count)
      +capacity = Tree.capacity(F32,tree)
      +start = (channel * count : U32)
      +values = plane_values(channel,count,source_stride,source_start,input)
      +written = Tree.write_sequence(~F32,values,start,tree)
      +next = (channel + 1 : U32)
      +written_capacity = DenseArrayInitialization.capacity(values,start,tree,depth,balanced)
      +next_exact = next_channel(channel,rest,limit,limit_ok,channels_fit)
      +before = before_distinct(channel_nat,target,after,same_exact)
      inside_planes(rest,Nat.is_eq(U32.to_nat(next),target),next,count,source_stride,source_start,input,written,depth,query,limit,limit_ok,
        DenseArrayInitialization.balanced(values,start,tree,depth,balanced),depth_ok,
        %Equal.sym(Nat,U32.to_nat(next),1n+channel_nat,next_exact) : {Nat.is_le(Nat.add(_,rest),U32.to_nat(limit)) == True{} : Bool}
        %Addition.successor_right(channel_nat,rest) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool}
        channels_fit,
        %Equal.sym(U32,Tree.capacity(F32,written),capacity,written_capacity) : {Nat.is_le(Nat.mul(Nat.add(U32.to_nat(next),rest),count_nat),U32.to_nat(_)) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(next),1n+channel_nat,next_exact) : {Nat.is_le(Nat.mul(Nat.add(_,rest),count_nat),U32.to_nat(capacity)) == True{} : Bool}
        %Addition.successor_right(channel_nat,rest) : {Nat.is_le(Nat.mul(_,count_nat),U32.to_nat(capacity)) == True{} : Bool}
        room,
        %Equal.sym(Nat,U32.to_nat(next),1n+channel_nat,next_exact) : {Nat.is_le(Nat.mul(Nat.add(_,rest),count_nat),U32.to_nat(limit)) == True{} : Bool}
        %Addition.successor_right(channel_nat,rest) : {Nat.is_le(Nat.mul(_,count_nat),U32.to_nat(limit)) == True{} : Bool}
        whole_fits,
        %Equal.sym(U32,Tree.capacity(F32,written),capacity,written_capacity) : {U32.is_lt(query,_) == True{} : Bool}
        query_inside,
        target,column,{==},query_exact,column_inside,
        %Equal.sym(Nat,U32.to_nat(next),1n+channel_nat,next_exact) : {Nat.is_le(_,target) == True{} : Bool}
        Order.strict_follow(channel_nat,target,before),
        %Equal.sym(Nat,U32.to_nat(next),1n+channel_nat,next_exact) : {Nat.is_lt(target,Nat.add(_,rest)) == True{} : Bool}
        %Addition.successor_right(channel_nat,rest) : {Nat.is_lt(target,_) == True{} : Bool}
        target_inside,
        source_fits)
