# One coverage argument for any indexed output traversal. Each query names its
# producing job; that job writes the specified value and later jobs preserve it.
# A concrete strategy must discharge these local obligations and target bounds.
import Base
import ../traversal_array.bend as Traversal
import ./traversal_array_refinement.bend as ArrayLaws
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./nat_to_u32_bounds.bend as Bounds

def suffix_preserves_readback(~State: Data,~Context: Data,~Value: Data,
  ~limit: Context -> Nat,~target: Context -> Nat,~Frame: Context -> State -> Data,
  ~read: Context -> State -> Value,~update: U32 -> Context -> State -> State,
  ~frame_step: @+context: Context -> @+index: U32 -> @+state: State -> Frame(context,state) -> Frame(context,update(index,context,state)),
  ~later: @+context: Context -> @+index: Nat -> @+state: State -> Frame(context,state) ->
    {Nat.is_lt(target(context),index) == True{} : Bool} -> {Nat.is_lt(index,limit(context)) == True{} : Bool} ->
    {read(context,update(U32.from_nat(index),context,state)) == read(context,state) : Value},
  +count: Nat,+first: Nat,+context: Context,+state: State,+frame: Frame(context,state),
  +after: {Nat.is_lt(target(context),first) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(first,count),limit(context)) == True{} : Bool}) ->
  {read(context,Traversal.indexed_fold(~State,~Context,~update,count,U32.from_nat(first),context,state)) == read(context,state) : Value}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(U32,(U32.from_nat(first) + 1 : U32),U32.from_nat(1n+first),Bounds.add_one(U32.from_nat(first))) :
        {read(context,Traversal.indexed_fold(~State,~Context,~update,rest,_,context,update(U32.from_nat(first),context,state))) == read(context,state) : Value}
      %later(context,first,state,frame,after,Order.room_index(first,rest,limit(context),room)) :
        {read(context,Traversal.indexed_fold(~State,~Context,~update,rest,U32.from_nat(1n+first),context,update(U32.from_nat(first),context,state))) == _ : Value}
      suffix_preserves_readback(~State,~Context,~Value,~limit,~target,~Frame,~read,~update,~frame_step,~later,
        rest,1n+first,context,update(U32.from_nat(first),context,state),frame_step(context,U32.from_nat(first),state,frame),
        Order.lt_succ(target(context),first,Bounds.strict_le(target(context),first,after)),Order.advance_room(first,rest,limit(context),room))

def prefix_preserves_frame(~State: Data,~Context: Data,
  ~limit: Context -> Nat,~Frame: Context -> State -> Data,~update: U32 -> Context -> State -> State,
  ~frame_step: @+context: Context -> @+index: U32 -> @+state: State -> Frame(context,state) -> Frame(context,update(index,context,state)),
  +count: Nat,+context: Context,+state: State,room: {Nat.is_le(count,limit(context)) == True{} : Bool},frame: Frame(context,state)) ->
  Frame(context,Traversal.indexed_fold(~State,~Context,~update,count,0,context,state)):
  ArrayLaws.indexed_fold_preserves_invariant(~State,~Context,~limit,~(context => index => state => Frame(context,state)),~update,
    ~(context => index => state => inside => evidence => frame_step(context,U32.from_nat(index),state,evidence)),count,0n,context,state,room,frame)

def suffix_room(+total: Nat,+index: Nat,inside: {Nat.is_lt(index,total) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(1n+index,Nat.sub(total,1n+index)),total) == True{} : Bool}:
  %Addition.successor_right(index,Nat.sub(total,1n+index)) :
    {Nat.is_le(_,total) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(index,1n+Nat.sub(total,1n+index)),total,Order.split_at_element(total,index,inside)) :
    {Nat.is_le(_,total) == True{} : Bool}
  Order.reflexive(total)

# No FP32-specific reasoning, lane enumeration, or second whole-loop proof is
# required when a new execution strategy uses the common indexed traversal.
def complete_readback(~State: Data,~Context: Data,~Value: Data,
  ~limit: Context -> Nat,~target: Context -> Nat,~wanted: Context -> Value,~Frame: Context -> State -> Data,
  ~read: Context -> State -> Value,~update: U32 -> Context -> State -> State,
  ~frame_step: @+context: Context -> @+index: U32 -> @+state: State -> Frame(context,state) -> Frame(context,update(index,context,state)),
  ~at_target: @+context: Context -> @+state: State -> Frame(context,state) ->
    {read(context,update(U32.from_nat(target(context)),context,state)) == wanted(context) : Value},
  ~later: @+context: Context -> @+index: Nat -> @+state: State -> Frame(context,state) ->
    {Nat.is_lt(target(context),index) == True{} : Bool} -> {Nat.is_lt(index,limit(context)) == True{} : Bool} ->
    {read(context,update(U32.from_nat(index),context,state)) == read(context,state) : Value},
  +context: Context,+state: State,frame: Frame(context,state),+inside: {Nat.is_lt(target(context),limit(context)) == True{} : Bool}) ->
  {read(context,Traversal.indexed_fold(~State,~Context,~update,limit(context),0,context,state)) == wanted(context) : Value}:
  +index = target(context)
  +rest = Nat.sub(limit(context),1n+index)
  +prefix = Traversal.indexed_fold(~State,~Context,~update,index,0,context,state)
  +ready = prefix_preserves_frame(~State,~Context,~limit,~Frame,~update,~frame_step,index,context,state,Bounds.strict_le(index,limit(context),inside),frame)
  +written = update(U32.from_nat(index),context,prefix)
  %Equal.sym(Nat,limit(context),Nat.add(index,1n+rest),Equal.sym(Nat,Nat.add(index,1n+rest),limit(context),Order.split_at_element(limit(context),index,inside))) :
    {read(context,Traversal.indexed_fold(~State,~Context,~update,_,0,context,state)) == wanted(context) : Value}
  %Equal.sym(State,Traversal.indexed_fold(~State,~Context,~update,Nat.add(index,1n+rest),0,context,state),
    Traversal.indexed_fold(~State,~Context,~update,1n+rest,U32.from_nat(index),context,prefix),
    ArrayLaws.indexed_fold_split(~State,~Context,~update,index,1n+rest,0n,context,state)) :
    {read(context,_) == wanted(context) : Value}
  %Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.add_one(U32.from_nat(index))) :
    {read(context,Traversal.indexed_fold(~State,~Context,~update,rest,_,context,written)) == wanted(context) : Value}
  Equal.trans(Value,read(context,Traversal.indexed_fold(~State,~Context,~update,rest,U32.from_nat(1n+index),context,written)),read(context,written),wanted(context),
    suffix_preserves_readback(~State,~Context,~Value,~limit,~target,~Frame,~read,~update,~frame_step,~later,rest,1n+index,context,written,
      frame_step(context,U32.from_nat(index),prefix,ready),Order.lt_succ(index,index,Order.reflexive(index)),suffix_room(limit(context),index,inside)),
    at_target(context,prefix,ready))
