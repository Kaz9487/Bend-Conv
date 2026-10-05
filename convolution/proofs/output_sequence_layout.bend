# Lengths and concatenation of the canonical NCHW output sequence.
import Base
import ../specification.bend as ConvSpec
import ../../lib/ordered_matrix.bend as OrderedMatrix
import ./sequence_congruence.bend as Sequences
import ../ordered_reference.bend as OrderedReference
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/row_major_index.bend as RowMajor

def split_unpack(+first: Nat,+second: Nat,+start: Nat,+buffer: Tree.Tree<F32>) ->
  {Tree.read_sequence(~F32,Nat.add(first,second),U32.from_nat(start),buffer) ==
    OrderedReference.cat(Tree.read_sequence(~F32,first,U32.from_nat(start),buffer),Tree.read_sequence(~F32,second,U32.from_nat(Nat.add(start,first)),buffer)) : List<&2,F32>}:
  match first:
    case 0n:
      %Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)) : {Tree.read_sequence(~F32,second,U32.from_nat(start),buffer) == Tree.read_sequence(~F32,second,U32.from_nat(_),buffer) : List<&2,F32>}
      {==}
    case 1n+ +rest:
      %Equal.sym(U32,(U32.from_nat(start) + 1 : U32),U32.from_nat(1n+start),Bounds.add_one(U32.from_nat(start))) :
        {Tree.read(F32,buffer,U32.from_nat(start)) <> Tree.read_sequence(~F32,Nat.add(rest,second),_,buffer) ==
          Tree.read(F32,buffer,U32.from_nat(start)) <> OrderedReference.cat(Tree.read_sequence(~F32,rest,_,buffer),Tree.read_sequence(~F32,second,U32.from_nat(Nat.add(start,1n+rest)),buffer)) : List<&2,F32>}
      %Equal.sym(Nat,Nat.add(start,1n+rest),1n+Nat.add(start,rest),Addition.successor_right(start,rest)) :
        {Tree.read(F32,buffer,U32.from_nat(start)) <> Tree.read_sequence(~F32,Nat.add(rest,second),U32.from_nat(1n+start),buffer) ==
          Tree.read(F32,buffer,U32.from_nat(start)) <> OrderedReference.cat(Tree.read_sequence(~F32,rest,U32.from_nat(1n+start),buffer),Tree.read_sequence(~F32,second,U32.from_nat(_),buffer)) : List<&2,F32>}
      %split_unpack(rest,second,1n+start,buffer) :
        {Tree.read(F32,buffer,U32.from_nat(start)) <> Tree.read_sequence(~F32,Nat.add(rest,second),U32.from_nat(1n+start),buffer) == Tree.read(F32,buffer,U32.from_nat(start)) <> _ : List<&2,F32>}
      {==}

def next_column(+channel: Nat,+height: Nat,+width: Nat,+row: Nat,+column: Nat) ->
  {RowMajor.index(channel,height,width,row,1n+column) == 1n+RowMajor.index(channel,height,width,row,column) : Nat}:
  %Equal.sym(Nat,Nat.add(Nat.mul(row,width),1n+column),1n+Nat.add(Nat.mul(row,width),column),Addition.successor_right(Nat.mul(row,width),column)) :
    {Nat.add(Nat.mul(channel,Nat.mul(height,width)),_) == 1n+RowMajor.index(channel,height,width,row,column) : Nat}
  Addition.successor_right(Nat.mul(channel,Nat.mul(height,width)),Nat.add(Nat.mul(row,width),column))

def index_at_row_start(+channel: Nat,+height: Nat,+width: Nat,+row: Nat) ->
  {RowMajor.index(channel,height,width,row,0n) == Nat.add(Nat.mul(channel,Nat.mul(height,width)),Nat.mul(row,width)) : Nat}:
  %Addition.zero_right(Nat.mul(row,width)) : {RowMajor.index(channel,height,width,row,0n) == Nat.add(Nat.mul(channel,Nat.mul(height,width)),_) : Nat}
  {==}

def next_row(+channel: Nat,+height: Nat,+width: Nat,+row: Nat) ->
  {Nat.add(RowMajor.index(channel,height,width,row,0n),width) == RowMajor.index(channel,height,width,1n+row,0n) : Nat}:
  +base = Nat.mul(channel,Nat.mul(height,width))
  %Equal.sym(Nat,RowMajor.index(channel,height,width,row,0n),Nat.add(base,Nat.mul(row,width)),index_at_row_start(channel,height,width,row)) :
    {Nat.add(_,width) == RowMajor.index(channel,height,width,1n+row,0n) : Nat}
  %Equal.sym(Nat,RowMajor.index(channel,height,width,1n+row,0n),Nat.add(base,Nat.mul(1n+row,width)),index_at_row_start(channel,height,width,1n+row)) :
    {Nat.add(Nat.add(base,Nat.mul(row,width)),width) == _ : Nat}
  %Order.commutative(Nat.mul(row,width),width) :
    {Nat.add(Nat.add(base,Nat.mul(row,width)),width) == Nat.add(base,_) : Nat}
  Addition.associative(base,Nat.mul(row,width),width)

def index_at_channel_start(+channel: Nat,+height: Nat,+width: Nat) ->
  {RowMajor.index(channel,height,width,0n,0n) == Nat.mul(channel,Nat.mul(height,width)) : Nat}:
  Addition.zero_right(Nat.mul(channel,Nat.mul(height,width)))

def next_channel(+channel: Nat,+height: Nat,+width: Nat) ->
  {Nat.add(RowMajor.index(channel,height,width,0n,0n),Nat.mul(height,width)) == RowMajor.index(1n+channel,height,width,0n,0n) : Nat}:
  +span = Nat.mul(height,width)
  %Equal.sym(Nat,RowMajor.index(channel,height,width,0n,0n),Nat.mul(channel,span),index_at_channel_start(channel,height,width)) :
    {Nat.add(_,span) == RowMajor.index(1n+channel,height,width,0n,0n) : Nat}
  %Equal.sym(Nat,RowMajor.index(1n+channel,height,width,0n,0n),Nat.mul(1n+channel,span),index_at_channel_start(1n+channel,height,width)) :
    {Nat.add(Nat.mul(channel,span),span) == _ : Nat}
  Order.commutative(Nat.mul(channel,span),span)

# The frozen specification uses NCHW lists. This is its one serialization
# boundary: callers supply pointwise contents, never a kernel or store schedule.
def PointwiseContents(~Context: Data,~problem: Context -> ConvSpec.Problem,~output: Context -> Tree.Tree<F32>) -> Type:
  @+context: Context -> @+channel: Nat -> @+row: Nat -> @+column: Nat ->
    {Nat.is_lt(channel,ConvSpec.outputs(problem(context))) == True{} : Bool} ->
    {Nat.is_lt(row,ConvSpec.out_height(problem(context))) == True{} : Bool} ->
    {Nat.is_lt(column,ConvSpec.out_width(problem(context))) == True{} : Bool} ->
    {Tree.read(F32,output(context),U32.from_nat(RowMajor.index(channel,ConvSpec.out_height(problem(context)),ConvSpec.out_width(problem(context)),row,column))) ==
      (OrderedReference.input_channels(ConvSpec.channels(problem(context)),0n,channel,row,column,problem(context),0.0) + ConvSpec.bias_at(problem(context),channel) : F32) : F32}

def readback_columns(~Context: Data,~problem: Context -> ConvSpec.Problem,~output: Context -> Tree.Tree<F32>,
  ~readback: PointwiseContents(~Context,~problem,~output),
  +count: Nat,+column: Nat,+row: Nat,+channel: Nat,+context: Context,
  +channel_inside: {Nat.is_lt(channel,ConvSpec.outputs(problem(context))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem(context))) == True{} : Bool},
  +remaining: {Nat.is_le(Nat.add(column,count),ConvSpec.out_width(problem(context))) == True{} : Bool}) ->
  {Tree.read_sequence(~F32,count,U32.from_nat(RowMajor.index(channel,ConvSpec.out_height(problem(context)),ConvSpec.out_width(problem(context)),row,column)),output(context)) ==
    OrderedReference.output_columns(count,column,row,channel,problem(context)) : List<&2,F32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +p = problem(context)
      +start = RowMajor.index(channel,ConvSpec.out_height(p),ConvSpec.out_width(p),row,column)
      %readback(context,channel,row,column,channel_inside,row_inside,Order.room_index(column,rest,ConvSpec.out_width(p),remaining)) :
        {Tree.read(F32,output(context),U32.from_nat(start)) <> Tree.read_sequence(~F32,rest,(U32.from_nat(start) + 1 : U32),output(context)) ==
          _ <> OrderedReference.output_columns(rest,1n+column,row,channel,p) : List<&2,F32>}
      %Equal.sym(U32,(U32.from_nat(start) + 1 : U32),U32.from_nat(1n+start),Bounds.add_one(U32.from_nat(start))) :
        {Tree.read(F32,output(context),U32.from_nat(start)) <> Tree.read_sequence(~F32,rest,_,output(context)) ==
          Tree.read(F32,output(context),U32.from_nat(start)) <> OrderedReference.output_columns(rest,1n+column,row,channel,p) : List<&2,F32>}
      %next_column(channel,ConvSpec.out_height(p),ConvSpec.out_width(p),row,column) :
        {Tree.read(F32,output(context),U32.from_nat(start)) <> Tree.read_sequence(~F32,rest,U32.from_nat(_),output(context)) ==
          Tree.read(F32,output(context),U32.from_nat(start)) <> OrderedReference.output_columns(rest,1n+column,row,channel,p) : List<&2,F32>}
      %readback_columns(~Context,~problem,~output,~readback,rest,1n+column,row,channel,context,channel_inside,row_inside,
        Order.advance_room(column,rest,ConvSpec.out_width(p),remaining)) :
        {Tree.read(F32,output(context),U32.from_nat(start)) <> Tree.read_sequence(~F32,rest,
            U32.from_nat(RowMajor.index(channel,ConvSpec.out_height(p),ConvSpec.out_width(p),row,1n+column)),output(context)) ==
          Tree.read(F32,output(context),U32.from_nat(start)) <> _ : List<&2,F32>}
      {==}

def readback_rows(~Context: Data,~problem: Context -> ConvSpec.Problem,~output: Context -> Tree.Tree<F32>,
  ~readback: PointwiseContents(~Context,~problem,~output),
  +count: Nat,+row: Nat,+channel: Nat,+context: Context,
  +channel_inside: {Nat.is_lt(channel,ConvSpec.outputs(problem(context))) == True{} : Bool},
  +remaining: {Nat.is_le(Nat.add(row,count),ConvSpec.out_height(problem(context))) == True{} : Bool}) ->
  {Tree.read_sequence(~F32,Nat.mul(count,ConvSpec.out_width(problem(context))),
      U32.from_nat(RowMajor.index(channel,ConvSpec.out_height(problem(context)),ConvSpec.out_width(problem(context)),row,0n)),output(context)) ==
    OrderedReference.output_rows(count,row,channel,problem(context)) : List<&2,F32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +p = problem(context)
      +width = ConvSpec.out_width(p)
      +start = RowMajor.index(channel,ConvSpec.out_height(p),width,row,0n)
      %Equal.sym(List<&2,F32>,Tree.read_sequence(~F32,Nat.add(width,Nat.mul(rest,width)),U32.from_nat(start),output(context)),
        OrderedReference.cat(Tree.read_sequence(~F32,width,U32.from_nat(start),output(context)),Tree.read_sequence(~F32,Nat.mul(rest,width),U32.from_nat(Nat.add(start,width)),output(context))),
        split_unpack(width,Nat.mul(rest,width),start,output(context))) : {_ == OrderedReference.output_rows(1n+rest,row,channel,p) : List<&2,F32>}
      %Equal.sym(Nat,Nat.add(start,width),RowMajor.index(channel,ConvSpec.out_height(p),width,1n+row,0n),next_row(channel,ConvSpec.out_height(p),width,row)) :
        {OrderedReference.cat(Tree.read_sequence(~F32,width,U32.from_nat(start),output(context)),Tree.read_sequence(~F32,Nat.mul(rest,width),U32.from_nat(_),output(context))) ==
          OrderedReference.output_rows(1n+rest,row,channel,p) : List<&2,F32>}
      %readback_columns(~Context,~problem,~output,~readback,width,0n,row,channel,context,channel_inside,
        Order.room_index(row,rest,ConvSpec.out_height(p),remaining),Order.reflexive(width)) :
        {OrderedReference.cat(Tree.read_sequence(~F32,width,U32.from_nat(start),output(context)),Tree.read_sequence(~F32,Nat.mul(rest,width),
            U32.from_nat(RowMajor.index(channel,ConvSpec.out_height(p),width,1n+row,0n)),output(context))) == OrderedReference.cat(_,OrderedReference.output_rows(rest,1n+row,channel,p)) : List<&2,F32>}
      %readback_rows(~Context,~problem,~output,~readback,rest,1n+row,channel,context,channel_inside,Order.advance_room(row,rest,ConvSpec.out_height(p),remaining)) :
        {OrderedReference.cat(Tree.read_sequence(~F32,width,U32.from_nat(start),output(context)),Tree.read_sequence(~F32,Nat.mul(rest,width),
            U32.from_nat(RowMajor.index(channel,ConvSpec.out_height(p),width,1n+row,0n)),output(context))) ==
          OrderedReference.cat(Tree.read_sequence(~F32,width,U32.from_nat(start),output(context)),_) : List<&2,F32>}
      {==}

def readback_channels(~Context: Data,~problem: Context -> ConvSpec.Problem,~output: Context -> Tree.Tree<F32>,
  ~readback: PointwiseContents(~Context,~problem,~output),
  +count: Nat,+channel: Nat,+context: Context,
  +remaining: {Nat.is_le(Nat.add(channel,count),ConvSpec.outputs(problem(context))) == True{} : Bool}) ->
  {Tree.read_sequence(~F32,Nat.mul(count,Nat.mul(ConvSpec.out_height(problem(context)),ConvSpec.out_width(problem(context)))),
      U32.from_nat(RowMajor.index(channel,ConvSpec.out_height(problem(context)),ConvSpec.out_width(problem(context)),0n,0n)),output(context)) ==
    OrderedReference.output_channels(count,channel,problem(context)) : List<&2,F32>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +p = problem(context)
      +height = ConvSpec.out_height(p)
      +width = ConvSpec.out_width(p)
      +space = Nat.mul(height,width)
      +start = RowMajor.index(channel,height,width,0n,0n)
      %Equal.sym(List<&2,F32>,Tree.read_sequence(~F32,Nat.add(space,Nat.mul(rest,space)),U32.from_nat(start),output(context)),
        OrderedReference.cat(Tree.read_sequence(~F32,space,U32.from_nat(start),output(context)),Tree.read_sequence(~F32,Nat.mul(rest,space),U32.from_nat(Nat.add(start,space)),output(context))),
        split_unpack(space,Nat.mul(rest,space),start,output(context))) : {_ == OrderedReference.output_channels(1n+rest,channel,p) : List<&2,F32>}
      %Equal.sym(Nat,Nat.add(start,space),RowMajor.index(1n+channel,height,width,0n,0n),next_channel(channel,height,width)) :
        {OrderedReference.cat(Tree.read_sequence(~F32,space,U32.from_nat(start),output(context)),Tree.read_sequence(~F32,Nat.mul(rest,space),U32.from_nat(_),output(context))) ==
          OrderedReference.output_channels(1n+rest,channel,p) : List<&2,F32>}
      %readback_rows(~Context,~problem,~output,~readback,height,0n,channel,context,Order.room_index(channel,rest,ConvSpec.outputs(p),remaining),Order.reflexive(height)) :
        {OrderedReference.cat(Tree.read_sequence(~F32,space,U32.from_nat(start),output(context)),Tree.read_sequence(~F32,Nat.mul(rest,space),U32.from_nat(RowMajor.index(1n+channel,height,width,0n,0n)),output(context))) ==
          OrderedReference.cat(_,OrderedReference.output_channels(rest,1n+channel,p)) : List<&2,F32>}
      %readback_channels(~Context,~problem,~output,~readback,rest,1n+channel,context,Order.advance_room(channel,rest,ConvSpec.outputs(p),remaining)) :
        {OrderedReference.cat(Tree.read_sequence(~F32,space,U32.from_nat(start),output(context)),Tree.read_sequence(~F32,Nat.mul(rest,space),U32.from_nat(RowMajor.index(1n+channel,height,width,0n,0n)),output(context))) ==
          OrderedReference.cat(Tree.read_sequence(~F32,space,U32.from_nat(start),output(context)),_) : List<&2,F32>}
      {==}
