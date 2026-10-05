# Full output coverage of the shared driver. Numeric sample-to-matrix geometry
# is separate: here the wanted value is the actual ordered job value.
import Base
import ../shape.bend as ConvShape
import ../scalar_kernel.bend as ScalarConv
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/traversal_output_refinement.bend as OutputLaws
import ../../lib/proofs/traversal_array_refinement.bend as ArrayLaws
import ../../lib/proofs/indexed_output_readback.bend as OutputReadback
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/natural_addition.bend as Addition
import ./execution_storage.bend as ExecutionStorage
import ./dense_array_initialization.bend as DenseArrayInitialization
import ../../lib/proofs/nat_interval.bend as Interval
import ./storage_capacity.bend as StorageCapacity

type DirectQuery is Data:
  DirectQuery{shape: ConvShape.Shape,source: OutputLaws.Sources<F32,F32>,count: Nat,index: Nat,depth: Nat,
    depth_bound: {Nat.is_le(depth,30n) == True{} : Bool},
    room: {Nat.is_le(count,U32.to_nat(U32.shln(1,depth))) == True{} : Bool},
    inside: {Nat.is_lt(index,count) == True{} : Bool}}

def direct_context(query: DirectQuery) -> OutputLaws.OutputContext<F32,F32,ConvShape.Shape>:
  DirectQuery{shape,source,count,index,depth,depth_bound,room,inside} = query
  OutputLaws.OutputContext{shape,source}

def direct_limit(query: DirectQuery) -> Nat:
  DirectQuery{shape,source,count,index,depth,depth_bound,room,inside} = query
  count

def direct_target(query: DirectQuery) -> Nat:
  DirectQuery{shape,source,count,index,depth,depth_bound,room,inside} = query
  index

def direct_wanted(query: DirectQuery) -> F32:
  DirectQuery{shape,source,count,index,depth,depth_bound,room,inside} = query
  ExecutionStorage.direct_value(shape,U32.from_nat(index),source)

def DirectFrame(query: DirectQuery,output: Tree.Tree<F32>) -> Data:
  DirectQuery{shape,source,count,index,depth,depth_bound,room,inside} = query
  Shape.Balanced(F32,depth,output)

def direct_read(query: DirectQuery,output: Tree.Tree<F32>) -> F32:
  Tree.read(F32,output,U32.from_nat(direct_target(query)))

def direct_step(index: U32,query: DirectQuery,output: Tree.Tree<F32>) -> Tree.Tree<F32>:
  OutputLaws.output_update(~F32,~F32,~F32,~F32,~ConvShape.Shape,~ScalarConv.DirectContext,~F32,
    ~ScalarConv.direct_job,~ExecutionStorage.direct_input_value,~ExecutionStorage.direct_weight_value,~ScalarConv.direct_update,~ExecutionStorage.direct_write_values,
    index,direct_context(query),output)

def direct_step_same(+index: U32,+query: DirectQuery,+output: Tree.Tree<F32>) ->
  {direct_step(index,query,output) ==
    OutputLaws.output_update(~F32,~F32,~F32,~F32,~ConvShape.Shape,~ScalarConv.DirectContext,~F32,
      ~ScalarConv.direct_job,~ExecutionStorage.direct_input_value,~ExecutionStorage.direct_weight_value,~ScalarConv.direct_update,~ExecutionStorage.direct_write_values,
      index,direct_context(query),output) : Tree.Tree<F32>}:
  {==}

def direct_frame_step(+query: DirectQuery,+index: U32,+output: Tree.Tree<F32>,frame: DirectFrame(query,output)) ->
  DirectFrame(query,direct_step(index,query,output)):
  DirectQuery{+shape,+source,count,target,+depth,depth_bound,room,inside} = query
  OutputLaws.output_update_preserves_property(~F32,~F32,~F32,~F32,~ConvShape.Shape,~ScalarConv.DirectContext,~F32,~Nat,
    ~ScalarConv.direct_job,~ExecutionStorage.direct_input_value,~ExecutionStorage.direct_weight_value,~ScalarConv.direct_update,~ExecutionStorage.direct_write_values,
    ~(depth => output => Shape.Balanced(F32,depth,output)),~ExecutionStorage.direct_write_preserves_balance,
    OutputLaws.OutputContext{shape,source},depth,index,output,frame)

def direct_capacity_room(+query: DirectQuery,+output: Tree.Tree<F32>,frame: DirectFrame(query,output)) ->
  {Nat.is_le(direct_limit(query),U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool}:
  DirectQuery{shape,source,+count,index,+depth,depth_bound,room,inside} = query
  %Equal.sym(U32,Tree.capacity(F32,output),U32.shln(1,depth),Shape.capacity(F32,depth,output,frame)) :
    {Nat.is_le(count,U32.to_nat(_)) == True{} : Bool}
  room

def direct_index_exact(+query: DirectQuery,+output: Tree.Tree<F32>,+index: Nat,+frame: DirectFrame(query,output),
  inside: {Nat.is_lt(index,direct_limit(query)) == True{} : Bool}) -> {U32.to_nat(U32.from_nat(index)) == index : Nat}:
  DirectQuery{+shape,+source,+count,+target,+depth,+depth_bound,+room,+target_inside} = query
  Bounds.roundtrip(index,Tree.capacity(F32,output),StorageCapacity.small(depth,output,frame,depth_bound),
    Bounds.strict_le(index,U32.to_nat(Tree.capacity(F32,output)),Bounds.nat_trans(index,count,U32.to_nat(Tree.capacity(F32,output)),inside,
      direct_capacity_room(query,output,frame))))

def direct_write_room(+query: DirectQuery,+output: Tree.Tree<F32>,+index: Nat,+frame: DirectFrame(query,output),
  +inside: {Nat.is_lt(index,direct_limit(query)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(U32.from_nat(index)),1n),U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,direct_index_exact(query,output,index,frame,inside)) :
    {Nat.is_le(Nat.add(_,1n),U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(index,1n),1n+index,Order.commutative(index,1n)) :
    {Nat.is_le(_,U32.to_nat(Tree.capacity(F32,output))) == True{} : Bool}
  Order.transitive(1n+index,direct_limit(query),U32.to_nat(Tree.capacity(F32,output)),Order.strict_follow(index,direct_limit(query),inside),direct_capacity_room(query,output,frame))

def direct_at_target(+query: DirectQuery,+output: Tree.Tree<F32>,+frame: DirectFrame(query,output)) ->
  {direct_read(query,direct_step(U32.from_nat(direct_target(query)),query,output)) == direct_wanted(query) : F32}:
  DirectQuery{+shape,+source,+count,+index,+depth,+depth_bound,+room,+inside} = query
  +value = ExecutionStorage.direct_value(shape,U32.from_nat(index),source)
  %Equal.sym(Tree.Tree<F32>,direct_step(U32.from_nat(index),query,output),Tree.write(F32,output,U32.from_nat(index),value),
    ExecutionStorage.direct_job_writes_index(shape,U32.from_nat(index),source,output)) :
    {Tree.read(F32,_,U32.from_nat(index)) == value : F32}
  DenseArrayInitialization.read_filled_range([value],U32.from_nat(index),output,depth,U32.from_nat(index),0n,frame,depth_bound,
    Equal.sym(Nat,Nat.add(U32.to_nat(U32.from_nat(index)),0n),U32.to_nat(U32.from_nat(index)),Addition.zero_right(U32.to_nat(U32.from_nat(index)))),
    direct_write_room(query,output,index,frame,inside),{==})

def direct_later_preserves(+query: DirectQuery,+index: Nat,+output: Tree.Tree<F32>,+frame: DirectFrame(query,output),
  +after: {Nat.is_lt(direct_target(query),index) == True{} : Bool},+inside: {Nat.is_lt(index,direct_limit(query)) == True{} : Bool}) ->
  {direct_read(query,direct_step(U32.from_nat(index),query,output)) == direct_read(query,output) : F32}:
  DirectQuery{+shape,+source,+count,+target,+depth,+depth_bound,+room,+target_inside} = query
  +value = ExecutionStorage.direct_value(shape,U32.from_nat(index),source)
  +target_exact = direct_index_exact(query,output,target,frame,target_inside)
  +index_exact = direct_index_exact(query,output,index,frame,inside)
  %Equal.sym(Tree.Tree<F32>,direct_step(U32.from_nat(index),query,output),Tree.write(F32,output,U32.from_nat(index),value),
    ExecutionStorage.direct_job_writes_index(shape,U32.from_nat(index),source,output)) :
    {Tree.read(F32,_,U32.from_nat(target)) == Tree.read(F32,output,U32.from_nat(target)) : F32}
  DenseArrayInitialization.read_filled_outside([value],U32.from_nat(index),output,depth,U32.from_nat(target),frame,depth_bound,direct_write_room(query,output,index,frame,inside),
    Bounds.native_bound(target,Tree.capacity(F32,output),target_exact,
      Bounds.nat_trans(target,count,U32.to_nat(Tree.capacity(F32,output)),target_inside,direct_capacity_room(query,output,frame))),
    Interval.Before{
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(target)),target,target_exact) : {Nat.is_lt(_,U32.to_nat(U32.from_nat(index))) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,index_exact) : {Nat.is_lt(target,_) == True{} : Bool}
      after})

def direct_result(query: DirectQuery,output: Tree.Tree<F32>) -> Tree.Tree<F32>:
  DirectQuery{shape,source,count,index,depth,depth_bound,room,inside} = query
  ExecutionStorage.direct_outputs(count,0,shape,source,output)

def direct_complete_readback(+query: DirectQuery,+output: Tree.Tree<F32>,frame: DirectFrame(query,output)) ->
  {Tree.read(F32,direct_result(query,output),U32.from_nat(direct_target(query))) == direct_wanted(query) : F32}:
  DirectQuery{+shape,+source,+count,+index,+depth,+depth_bound,+room,+inside} = query
  %ArrayLaws.indexed_fold_change_context(~Tree.Tree<F32>,~DirectQuery,~OutputLaws.OutputContext<F32,F32,ConvShape.Shape>,~direct_context,
    ~direct_step,~OutputLaws.output_update(~F32,~F32,~F32,~F32,~ConvShape.Shape,~ScalarConv.DirectContext,~F32,
      ~ScalarConv.direct_job,~ExecutionStorage.direct_input_value,~ExecutionStorage.direct_weight_value,~ScalarConv.direct_update,~ExecutionStorage.direct_write_values),
    ~direct_step_same,count,0,query,output) :
    {Tree.read(F32,_,U32.from_nat(index)) == ExecutionStorage.direct_value(shape,U32.from_nat(index),source) : F32}
  OutputReadback.complete_readback(~Tree.Tree<F32>,~DirectQuery,~F32,~direct_limit,~direct_target,~direct_wanted,~DirectFrame,~direct_read,~direct_step,
    ~direct_frame_step,~(query => output => frame => direct_at_target(query,output,frame)),
    ~(query => index => output => frame => after => inside => direct_later_preserves(query,index,output,frame,after,inside)),query,output,frame,inside)
