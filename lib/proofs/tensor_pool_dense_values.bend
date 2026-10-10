# Dense pooling: complete output values over the logical input. The pool is
# two checked sweeps: along the columns of every row into fresh planes
# [..,height,out_width], then along the rows into the fresh output. Each
# output holds the ordered fold, from pad, of the k lines of its window in
# increasing ky; a line is the ordered fold, from pad, of its k samples in
# increasing kx; a line or a sample outside the input is pad. No exchange,
# reassociation or law on the operation is used.
import Base
import ../tensor.bend as Tensors
import ../tensor_shape.bend as Shapes
import ../tensor_layout.bend as Layout
import ../tensor_window.bend as Window
import ../tensor_operations.bend as Operations
import ../tensor_access.bend as Access
import ../storage_buffer.bend as Storage
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_dense_contract.bend as Dense
import ./tensor_gather_values.bend as GatherValues
import ./tensor_copy_values.bend as CopyValues
import ./tensor_sweep_values.bend as Sweeps
import ./tensor_operations_refinement.bend as OperationsLaws
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding
import ./checked_word_refinement.bend as Checked
import ./storage_shape.bend as Shape
import ./storage_allocation_refinement.bend as AllocationLaws
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_refinement.bend as Count
import ./tensor_allocation_depth.bend as Depth
import ./tensor_band_count.bend as BandCount

# The logical input read through the two sweeps of a pool.
def context(~Element: Data,~Logical: Data,logical: Logical,+pool: Access.Pool,+pad: Element) -> Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>:
  Sweeps.Swept{Sweeps.Swept{logical,Access.pool_columns(pool),pad},Access.pool_rows(pool,0),pad}

def swept_cells(pool: Access.Pool,channels: U32) -> Nat:
  Access.Pool{height,width,kernel,stride,padding,out_height,out_width} = pool
  Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(out_width)))

def pooled_cells(pool: Access.Pool,channels: U32) -> Nat:
  Access.Pool{height,width,kernel,stride,padding,out_height,out_width} = pool
  Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width)))

def lines_of(pool: Access.Pool,+channels: U32) -> Maybe<&2,U32>:
  Access.Pool{+height,width,kernel,stride,padding,out_height,out_width} = pool
  Shapes.product([channels,height],Some{1})

# The outcome of a pool is the outcome of its second sweep.
def finished_content(~Element: Data,input: Storage.Buffer<Element>,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) ->
  {CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,input,outcome))) == CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome)) : Buffer.Content<Element>}:
  match outcome:
    case Operations.Accepted{(middle,Storage.Buffer{storage,length,certificate})}: {==}
    case Operations.Rejected{(middle,Storage.Buffer{storage,length,certificate})}: {==}

def finished_accepted(~Element: Data,input: Storage.Buffer<Element>,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) ->
  {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,input,outcome))) == GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome)) : Bool}:
  match outcome:
    case Operations.Accepted{(middle,output)}: {==}
    case Operations.Rejected{(middle,output)}: {==}

# The second sweep reads the planes the first one wrote.
def rows_modeled(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+input: Buffer.Model<Element>,+middle: Buffer.Model<Element>,
  +logical: Logical,+pad: Element,+pool: Access.Pool,+channels: U32,+output: Buffer.Model<Element>,+middle_shape: List<&2,U32>,+out_shape: List<&2,U32>,+limits: Machine.Limits,
  +actual: {limits == Default.limits() : Machine.Limits},+ready: Buffer.Sized<Element,output>,
  +cells: {Sweeps.output_cells(Access.pool_rows(pool,0),U32.to_nat(channels)) == Buffer.length(~Element,output) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(out_shape) : Nat},
  facts: GatherValues.Facts(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),Sweeps.Swept{logical,Access.pool_columns(pool),pad},middle_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.Accepted{(Buffer.pack(~Element,input),Buffer.pack(~Element,middle))}))),
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,Buffer.pack(~Element,input),Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,middle),Access.pool_rows(pool,0),channels,Buffer.pack(~Element,output))))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,pad),out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,Buffer.pack(~Element,input),Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,middle),Access.pool_rows(pool,0),channels,Buffer.pack(~Element,output)))))):
  Buffer.Model{+input_tree,+input_length,+input_certificate} = input
  Buffer.Model{+middle_tree,+middle_length,+middle_certificate} = middle
  Shape.Both{Dense.WellFormed{middle_exact,middle_content_ready},middle_values} = facts
  +input = {Buffer.Model{input_tree,input_length,input_certificate} : Buffer.Model<Element>}
  +middle = {Buffer.Model{middle_tree,middle_length,middle_certificate} : Buffer.Model<Element>}
  %Equal.sym(Buffer.Content<Element>,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,Buffer.pack(~Element,input),Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,middle),Access.pool_rows(pool,0),channels,Buffer.pack(~Element,output))))),CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,middle),Access.pool_rows(pool,0),channels,Buffer.pack(~Element,output)))),finished_content(~Element,Buffer.pack(~Element,input),Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,middle),Access.pool_rows(pool,0),channels,Buffer.pack(~Element,output)))) :
    GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,pad),out_shape,_)
  Sweeps.modeled_facts(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation,Sweeps.Swept{logical,Access.pool_columns(pool),pad},pad,Access.pool_rows(pool,0),channels,middle,output,out_shape,limits,actual,
    Buffer.ready_content(~Element,middle,
      %Buffer.packed_contents(~Element,middle) : Buffer.ContentReady<Element,_>
      middle_content_ready),
    ready,
    %Buffer.packed_contents(~Element,middle) : Dense.Values(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),Sweeps.Swept{logical,Access.pool_columns(pool),pad},_)
    middle_values,
    cells,length_exact,
    %finished_accepted(~Element,Buffer.pack(~Element,input),Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,middle),Access.pool_rows(pool,0),channels,Buffer.pack(~Element,output))) : {_ == True{} : Bool}
    accepted)

def rows_observed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,-input: Storage.Buffer<Element>,-middle: Storage.Buffer<Element>,
  input_observation: Buffer.Observation<Element,input>,middle_observation: Buffer.Observation<Element,middle>,
  +logical: Logical,+pad: Element,+pool: Access.Pool,+channels: U32,+output: Buffer.Model<Element>,+middle_shape: List<&2,U32>,+out_shape: List<&2,U32>,+limits: Machine.Limits,
  +actual: {limits == Default.limits() : Machine.Limits},+ready: Buffer.Sized<Element,output>,
  +cells: {Sweeps.output_cells(Access.pool_rows(pool,0),U32.to_nat(channels)) == Buffer.length(~Element,output) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(out_shape) : Nat},
  facts: GatherValues.Facts(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),Sweeps.Swept{logical,Access.pool_columns(pool),pad},middle_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.Accepted{(input,middle)}))),
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,input,Operations.sweep(~Element,~operation,pad,middle,Access.pool_rows(pool,0),channels,Buffer.pack(~Element,output))))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,pad),out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,input,Operations.sweep(~Element,~operation,pad,middle,Access.pool_rows(pool,0),channels,Buffer.pack(~Element,output)))))):
  Buffer.Observation{+input_model,+input_equation} = input_observation
  Buffer.Observation{+middle_model,+middle_equation} = middle_observation
  %input_equation : GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,pad),out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,_,Operations.sweep(~Element,~operation,pad,middle,Access.pool_rows(pool,0),channels,Buffer.pack(~Element,output))))))
  %middle_equation : GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,pad),out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,Buffer.pack(~Element,input_model),Operations.sweep(~Element,~operation,pad,_,Access.pool_rows(pool,0),channels,Buffer.pack(~Element,output))))))
  rows_modeled(~Element,~Logical,~original,~operation,input_model,middle_model,logical,pad,pool,channels,output,middle_shape,out_shape,limits,actual,ready,cells,length_exact,
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,input_model),input,input_equation) : GatherValues.Facts(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),Sweeps.Swept{logical,Access.pool_columns(pool),pad},middle_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.Accepted{(_,Buffer.pack(~Element,middle_model))})))
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,middle_model),middle,middle_equation) : GatherValues.Facts(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),Sweeps.Swept{logical,Access.pool_columns(pool),pad},middle_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.Accepted{(input,_)})))
    facts,
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,input_model),input,input_equation) : {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,_,Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,middle_model),Access.pool_rows(pool,0),channels,Buffer.pack(~Element,output))))) == True{} : Bool}
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,middle_model),middle,middle_equation) : {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_finished(~Element,input,Operations.sweep(~Element,~operation,pad,_,Access.pool_rows(pool,0),channels,Buffer.pack(~Element,output))))) == True{} : Bool}
    accepted)

def rowed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,first: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,
  +logical: Logical,+pad: Element,+pool: Access.Pool,+channels: U32,+output: Buffer.Model<Element>,+middle_shape: List<&2,U32>,+out_shape: List<&2,U32>,+limits: Machine.Limits,
  +actual: {limits == Default.limits() : Machine.Limits},+ready: Buffer.Sized<Element,output>,
  +cells: {Sweeps.output_cells(Access.pool_rows(pool,0),U32.to_nat(channels)) == Buffer.length(~Element,output) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(out_shape) : Nat},
  facts: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,first)) == True{} : Bool} -> GatherValues.Facts(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),Sweeps.Swept{logical,Access.pool_columns(pool),pad},middle_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,first))),
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_rows(~Element,~operation,pad,first,Access.pool_rows(pool,0),channels,Buffer.pack(~Element,output)))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,pad),out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_rows(~Element,~operation,pad,first,Access.pool_rows(pool,0),channels,Buffer.pack(~Element,output))))):
  match first:
    case Operations.Accepted{(input,middle)}:
      rows_observed(~Element,~Logical,~original,~operation,input,middle,Buffer.observe(~Element,input),Buffer.observe(~Element,middle),logical,pad,pool,channels,output,middle_shape,out_shape,limits,actual,ready,cells,length_exact,facts({==}),accepted)
    case Operations.Rejected{(input,middle)}:
      Empty.absurd(GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,pad),out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_rows(~Element,~operation,pad,Operations.Rejected{(input,middle)},Access.pool_rows(pool,0),channels,Buffer.pack(~Element,output))))),WordEncoding.false_true(accepted))

def swept_lines(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+lines: U32,
  +logical: Logical,+pad: Element,+pool: Access.Pool,+channels: U32,+input: Buffer.Model<Element>,+middle: Buffer.Model<Element>,+output: Buffer.Model<Element>,+middle_shape: List<&2,U32>,+out_shape: List<&2,U32>,+limits: Machine.Limits,
  +actual: {limits == Default.limits() : Machine.Limits},+input_ready: Buffer.Sized<Element,input>,+middle_ready: Buffer.Sized<Element,middle>,+ready: Buffer.Sized<Element,output>,
  +values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +middle_cells: {swept_cells(pool,channels) == Buffer.length(~Element,middle) : Nat},
  +middle_exact: {Buffer.length(~Element,middle) == ShapeLaws.elements(middle_shape) : Nat},
  +cells: {pooled_cells(pool,channels) == Buffer.length(~Element,output) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(out_shape) : Nat},
  +lines_exact: {lines_of(pool,channels) == Some{lines} : Maybe<&2,U32>},
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_rows(~Element,~operation,pad,Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,input),Access.pool_columns(pool),lines,Buffer.pack(~Element,middle)),Access.pool_rows(pool,0),channels,Buffer.pack(~Element,output)))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,pad),out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_rows(~Element,~operation,pad,Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,input),Access.pool_columns(pool),lines,Buffer.pack(~Element,middle)),Access.pool_rows(pool,0),channels,Buffer.pack(~Element,output))))):
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  +pool = {Access.Pool{height,width,kernel,stride,padding,out_height,out_width} : Access.Pool}
  +channel_count = U32.to_nat(channels)
  +rows = U32.to_nat(height)
  +columns = U32.to_nat(out_width)
  rowed(~Element,~Logical,~original,~operation,Operations.sweep(~Element,~operation,pad,Buffer.pack(~Element,input),Access.pool_columns(pool),lines,Buffer.pack(~Element,middle)),logical,pad,pool,channels,output,middle_shape,out_shape,limits,actual,ready,cells,length_exact,
    first_accepted => Sweeps.modeled_facts(~Element,~Logical,~original,~operation,logical,pad,Access.pool_columns(pool),lines,input,middle,middle_shape,limits,actual,input_ready,middle_ready,values,
      Equal.trans(Nat,Nat.mul(U32.to_nat(lines),Nat.mul(columns,1n)),Nat.mul(channel_count,Nat.mul(rows,columns)),Buffer.length(~Element,middle),
        %Equal.sym(Nat,Nat.mul(columns,1n),columns,ShapeLaws.times_one(columns)) : {Nat.mul(U32.to_nat(lines),_) == Nat.mul(channel_count,Nat.mul(rows,columns)) : Nat}
        %Sweeps.two(channels,height,lines,lines_exact) : {Nat.mul(_,columns) == Nat.mul(channel_count,Nat.mul(rows,columns)) : Nat}
        RowMajor.assoc(channel_count,rows,columns),
        middle_cells),
      middle_exact,first_accepted),
    accepted)

def lined(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,lines: Maybe<&2,U32>,
  +logical: Logical,+pad: Element,+pool: Access.Pool,+channels: U32,+input: Buffer.Model<Element>,+middle: Buffer.Model<Element>,+output: Buffer.Model<Element>,+middle_shape: List<&2,U32>,+out_shape: List<&2,U32>,+limits: Machine.Limits,
  +actual: {limits == Default.limits() : Machine.Limits},+input_ready: Buffer.Sized<Element,input>,+middle_ready: Buffer.Sized<Element,middle>,+ready: Buffer.Sized<Element,output>,
  +values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +middle_cells: {swept_cells(pool,channels) == Buffer.length(~Element,middle) : Nat},
  +middle_exact: {Buffer.length(~Element,middle) == ShapeLaws.elements(middle_shape) : Nat},
  +cells: {pooled_cells(pool,channels) == Buffer.length(~Element,output) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(out_shape) : Nat},
  +lines_exact: {lines_of(pool,channels) == lines : Maybe<&2,U32>},
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_lines(~Element,~operation,pad,lines,Buffer.pack(~Element,input),pool,channels,Buffer.pack(~Element,middle),Buffer.pack(~Element,output)))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,pad),out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_lines(~Element,~operation,pad,lines,Buffer.pack(~Element,input),pool,channels,Buffer.pack(~Element,middle),Buffer.pack(~Element,output))))):
  match lines:
    case Some{+lines}:
      swept_lines(~Element,~Logical,~original,~operation,lines,logical,pad,pool,channels,input,middle,output,middle_shape,out_shape,limits,actual,input_ready,middle_ready,ready,values,middle_cells,middle_exact,cells,length_exact,lines_exact,accepted)
    case None{}:
      Empty.absurd(GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,pad),out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool_lines(~Element,~operation,pad,None{},Buffer.pack(~Element,input),pool,channels,Buffer.pack(~Element,middle),Buffer.pack(~Element,output))))),WordEncoding.false_true(accepted))

# One accepted checked pool of a Ready input with complete values, through
# planes and an output it writes whole: the output content is well formed for
# its shape and holds the pooled value of the logical input at every index.
def pooled_facts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,
  +logical: Logical,+pad: Element,+pool: Access.Pool,+channels: U32,+input: Buffer.Model<Element>,+middle: Buffer.Model<Element>,+output: Buffer.Model<Element>,+middle_shape: List<&2,U32>,+out_shape: List<&2,U32>,+limits: Machine.Limits,
  +actual: {limits == Default.limits() : Machine.Limits},+input_ready: Buffer.Sized<Element,input>,+middle_ready: Buffer.Sized<Element,middle>,+ready: Buffer.Sized<Element,output>,
  +values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +middle_cells: {swept_cells(pool,channels) == Buffer.length(~Element,middle) : Nat},
  +middle_exact: {Buffer.length(~Element,middle) == ShapeLaws.elements(middle_shape) : Nat},
  +cells: {pooled_cells(pool,channels) == Buffer.length(~Element,output) : Nat},
  +length_exact: {Buffer.length(~Element,output) == ShapeLaws.elements(out_shape) : Nat},
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),pool,channels,Buffer.pack(~Element,middle),Buffer.pack(~Element,output)))) == True{} : Bool}) ->
  GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,pad),out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),pool,channels,Buffer.pack(~Element,middle),Buffer.pack(~Element,output))))):
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  +pool = {Access.Pool{height,width,kernel,stride,padding,out_height,out_width} : Access.Pool}
  lined(~Element,~Logical,~original,~operation,lines_of(pool,channels),logical,pad,pool,channels,input,middle,output,middle_shape,out_shape,limits,actual,input_ready,middle_ready,ready,values,middle_cells,middle_exact,cells,length_exact,{==},accepted)

# The public result follows the outcome of the checked pool.
def pooled_ready(~Element: Data,+name: String,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,+shape: List<&2,U32>,+out_shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  {Dense.tensor_ready(~Element,Window.pooled(~Element,name,shape,out_shape,fill,workers,outcome)) == GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome)) : Bool}:
  match outcome:
    case Operations.Accepted{(input,output)}: {==}
    case Operations.Rejected{(input,output)}: {==}

def pooled_shaped(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+name: String,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,
  +context: Context,+shape: List<&2,U32>,+out_shape: List<&2,U32>,+fill: Element,+workers: U32,
  facts: GatherValues.Facts(~Element,~Context,~reader,context,out_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome))),
  accepted: {Dense.tensor_ready(~Element,Window.pooled(~Element,name,shape,out_shape,fill,workers,outcome)) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Context,reader,context,out_shape,Window.pooled(~Element,name,shape,out_shape,fill,workers,outcome)>:
  match outcome:
    case Operations.Accepted{(input,Storage.Buffer{storage,length,certificate})}:
      GatherValues.shaped_observed(~Element,~Context,~reader,Storage.Buffer{storage,length,certificate},context,out_shape,fill,workers,
        Buffer.observe(~Element,Storage.Buffer{storage,length,certificate}),facts)
    case Operations.Rejected{(input,output)}:
      Empty.absurd(GatherValues.Shaped<Element,Context,reader,context,out_shape,Window.pooled(~Element,name,shape,out_shape,fill,workers,Operations.Rejected{(input,output)})>,WordEncoding.false_true(accepted))

# A fresh buffer, observed as a model, has the length it was allocated with.
def fresh_length(~Element: Data,+fill: Element,+count: U32,+model: Buffer.Model<Element>,
  equation: {Buffer.pack(~Element,model) == Tensors.buffer(~Element,fill,count) : Storage.Buffer<Element>}) ->
  {Buffer.length(~Element,model) == U32.to_nat(count) : Nat}:
  +fresh = {Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)} : Buffer.Content<Element>}
  +allocated = Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,model),Buffer.contents(~Element,Buffer.pack(~Element,model)),fresh,
    Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.content(~Element,model),Buffer.packed_contents(~Element,model)),
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,model),Tensors.buffer(~Element,fill,count),equation) : {Buffer.contents(~Element,_) == fresh : Buffer.Content<Element>}
    AllocationLaws.contents(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)))
  Equal.trans(Nat,Buffer.length(~Element,model),Buffer.content_length(~Element,Buffer.content(~Element,model)),U32.to_nat(count),
    Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,model)),Buffer.length(~Element,model),Buffer.content_length_of(~Element,model)),
    Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.content(~Element,model),fresh,allocated))

# The fresh planes and the fresh output of the public call, observed as models.
def allocated(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +logical: Logical,+pad: Element,+pool: Access.Pool,+channels: U32,+count: U32,+input: Buffer.Model<Element>,+shape: List<&2,U32>,+out_shape: List<&2,U32>,+fill: Element,+workers: U32,+limits: Machine.Limits,+middle: U32,
  middle_observation: Buffer.Observation<Element,Tensors.buffer(~Element,fill,middle)>,observation: Buffer.Observation<Element,Tensors.buffer(~Element,fill,count)>,
  middle_evidence: Count.Count<ShapeLaws.elements(Window.middle_shape(shape,pool)),middle>,evidence: Count.Count<ShapeLaws.elements(out_shape),count>,
  +actual: {limits == Default.limits() : Machine.Limits},+input_ready: Buffer.Sized<Element,input>,+values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +middle_cells: {swept_cells(pool,channels) == ShapeLaws.elements(Window.middle_shape(shape,pool)) : Nat},
  +cells: {pooled_cells(pool,channels) == ShapeLaws.elements(out_shape) : Nat},
  accepted: {Dense.tensor_ready(~Element,Window.pooled(~Element,name,shape,out_shape,fill,workers,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),pool,channels,Tensors.buffer(~Element,fill,middle),Tensors.buffer(~Element,fill,count)))) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,pad),out_shape,Window.pooled(~Element,name,shape,out_shape,fill,workers,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),pool,channels,Tensors.buffer(~Element,fill,middle),Tensors.buffer(~Element,fill,count)))>:
  Buffer.Observation{+middle_model,+middle_equation} = middle_observation
  Buffer.Observation{+output,+equation} = observation
  Count.Count{+middle_count,+middle_bounded} = middle_evidence
  Count.Count{+exact,+bounded} = evidence
  +middle_shape = Window.middle_shape(shape,pool)
  +middle_length = fresh_length(~Element,fill,middle,middle_model,middle_equation)
  +length = fresh_length(~Element,fill,count,output,equation)
  +accepted_models = {
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,middle_model),Tensors.buffer(~Element,fill,middle),middle_equation) : {Dense.tensor_ready(~Element,Window.pooled(~Element,name,shape,out_shape,fill,workers,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),pool,channels,_,Buffer.pack(~Element,output)))) == True{} : Bool}
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,output),Tensors.buffer(~Element,fill,count),equation) : {Dense.tensor_ready(~Element,Window.pooled(~Element,name,shape,out_shape,fill,workers,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),pool,channels,Tensors.buffer(~Element,fill,middle),_))) == True{} : Bool}
    accepted : {Dense.tensor_ready(~Element,Window.pooled(~Element,name,shape,out_shape,fill,workers,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),pool,channels,Buffer.pack(~Element,middle_model),Buffer.pack(~Element,output)))) == True{} : Bool}}
  %middle_equation : GatherValues.Shaped<Element,Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,pad),out_shape,Window.pooled(~Element,name,shape,out_shape,fill,workers,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),pool,channels,_,Tensors.buffer(~Element,fill,count)))>
  %equation : GatherValues.Shaped<Element,Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,pad),out_shape,Window.pooled(~Element,name,shape,out_shape,fill,workers,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),pool,channels,Buffer.pack(~Element,middle_model),_))>
  pooled_shaped(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),name,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),pool,channels,Buffer.pack(~Element,middle_model),Buffer.pack(~Element,output)),context(~Element,~Logical,logical,pool,pad),shape,out_shape,fill,workers,
    pooled_facts(~Element,~Logical,~original,~operation,logical,pad,pool,channels,input,middle_model,output,middle_shape,out_shape,limits,actual,input_ready,
      Buffer.ready_observed(~Element,Tensors.buffer(~Element,fill,middle),middle_model,middle_equation,BandCount.allocated_ready(~Element,middle,fill,Depth.bounded(middle,middle_bounded))),
      Buffer.ready_observed(~Element,Tensors.buffer(~Element,fill,count),output,equation,BandCount.allocated_ready(~Element,count,fill,Depth.bounded(count,bounded))),
      values,
      Equal.trans(Nat,swept_cells(pool,channels),ShapeLaws.elements(middle_shape),Buffer.length(~Element,middle_model),middle_cells,
        Equal.trans(Nat,ShapeLaws.elements(middle_shape),U32.to_nat(middle),Buffer.length(~Element,middle_model),middle_count,
          Equal.sym(Nat,Buffer.length(~Element,middle_model),U32.to_nat(middle),middle_length))),
      Equal.trans(Nat,Buffer.length(~Element,middle_model),U32.to_nat(middle),ShapeLaws.elements(middle_shape),middle_length,
        Equal.sym(Nat,ShapeLaws.elements(middle_shape),U32.to_nat(middle),middle_count)),
      Equal.trans(Nat,pooled_cells(pool,channels),ShapeLaws.elements(out_shape),Buffer.length(~Element,output),cells,
        Equal.trans(Nat,ShapeLaws.elements(out_shape),U32.to_nat(count),Buffer.length(~Element,output),exact,
          Equal.sym(Nat,Buffer.length(~Element,output),U32.to_nat(count),length))),
      Equal.trans(Nat,Buffer.length(~Element,output),U32.to_nat(count),ShapeLaws.elements(out_shape),length,
        Equal.sym(Nat,ShapeLaws.elements(out_shape),U32.to_nat(count),exact)),
      %pooled_ready(~Element,name,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),pool,channels,Buffer.pack(~Element,middle_model),Buffer.pack(~Element,output)),shape,out_shape,fill,workers) : {_ == True{} : Bool}
      accepted_models),
    accepted_models)

def middled(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,middle: Maybe<&2,U32>,
  +logical: Logical,+pad: Element,+pool: Access.Pool,+channels: U32,+count: U32,+input: Buffer.Model<Element>,+shape: List<&2,U32>,+out_shape: List<&2,U32>,+fill: Element,+workers: U32,+limits: Machine.Limits,
  +actual: {limits == Default.limits() : Machine.Limits},+input_ready: Buffer.Sized<Element,input>,+values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +middle_cells: {swept_cells(pool,channels) == ShapeLaws.elements(Window.middle_shape(shape,pool)) : Nat},
  +cells: {pooled_cells(pool,channels) == ShapeLaws.elements(out_shape) : Nat},
  +count_exact: {Tensors.count_of(out_shape) == Some{count} : Maybe<&2,U32>},
  +middle_exact: {Tensors.count_of(Window.middle_shape(shape,pool)) == middle : Maybe<&2,U32>},
  accepted: {Dense.tensor_ready(~Element,Window.pool_middle(~Element,~operation,name,pad,middle,count,channels,pool,Buffer.pack(~Element,input),shape,out_shape,fill,workers)) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,pad),out_shape,Window.pool_middle(~Element,~operation,name,pad,middle,count,channels,pool,Buffer.pack(~Element,input),shape,out_shape,fill,workers)>:
  match middle:
    case Some{+middle}:
      allocated(~Element,~Logical,~original,~operation,name,logical,pad,pool,channels,count,input,shape,out_shape,fill,workers,limits,middle,
        Buffer.observe(~Element,Tensors.buffer(~Element,fill,middle)),Buffer.observe(~Element,Tensors.buffer(~Element,fill,count)),
        Count.count_of(Window.middle_shape(shape,pool),middle,middle_exact),Count.count_of(out_shape,count,count_exact),
        actual,input_ready,values,middle_cells,cells,accepted)
    case None{}:
      Empty.absurd(GatherValues.Shaped<Element,Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,pool,pad),out_shape,Window.pool_middle(~Element,~operation,name,pad,None{},count,channels,pool,Buffer.pack(~Element,input),shape,out_shape,fill,workers)>,WordEncoding.false_true(accepted))

# The elements of the shape of the planes and of the output.
def appended(+left: List<&2,U32>,+right: List<&2,U32>) ->
  {ShapeLaws.elements(List.append(&2,U32,left,right)) == Nat.mul(ShapeLaws.elements(left),ShapeLaws.elements(right)) : Nat}:
  match left:
    case Nil{}: Equal.sym(Nat,Nat.add(ShapeLaws.elements(right),0n),ShapeLaws.elements(right),Addition.zero_right(ShapeLaws.elements(right)))
    case Con{+head,+tail}:
      Equal.trans(Nat,Nat.mul(U32.to_nat(head),ShapeLaws.elements(List.append(&2,U32,tail,right))),Nat.mul(U32.to_nat(head),Nat.mul(ShapeLaws.elements(tail),ShapeLaws.elements(right))),
        Nat.mul(Nat.mul(U32.to_nat(head),ShapeLaws.elements(tail)),ShapeLaws.elements(right)),
        Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(head),value),ShapeLaws.elements(List.append(&2,U32,tail,right)),Nat.mul(ShapeLaws.elements(tail),ShapeLaws.elements(right)),appended(tail,right)),
        Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(head),ShapeLaws.elements(tail)),ShapeLaws.elements(right)),Nat.mul(U32.to_nat(head),Nat.mul(ShapeLaws.elements(tail),ShapeLaws.elements(right))),
          RowMajor.assoc(U32.to_nat(head),ShapeLaws.elements(tail),ShapeLaws.elements(right))))

def leading_elements(ranked: Bool,+shape: List<&2,U32>,+planes: U32,accepted: {Window.leading(ranked,shape) == Some{planes} : Maybe<&2,U32>}) ->
  {ShapeLaws.elements(List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n))) == U32.to_nat(planes) : Nat}:
  match ranked:
    case False{}:
      Empty.absurd({ShapeLaws.elements(List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n))) == U32.to_nat(planes) : Nat},
        WordEncoding.false_true(Equal.cong(Maybe<&2,U32>,Bool,result => Checked.present(result),None{},Some{planes},accepted)))
    case True{}:
      %accepted : ShapeLaws.Count(_,ShapeLaws.elements(List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n))))
      ShapeLaws.nonempty_count(List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n)))

def output_shape(+shape: List<&2,U32>,out_height: U32,out_width: U32) -> List<&2,U32>:
  List.append(&2,U32,List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n)),[out_height,out_width])

def planes_cells(+shape: List<&2,U32>,+out_height: U32,+out_width: U32,+planes: U32,accepted: {Window.planes(shape) == Some{planes} : Maybe<&2,U32>}) ->
  {Nat.mul(U32.to_nat(planes),Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width))) == ShapeLaws.elements(output_shape(shape,out_height,out_width)) : Nat}:
  +leading = List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n))
  %leading_elements(Nat.is_le(2n,List.length(&2,U32,shape)),shape,planes,accepted) :
    {Nat.mul(_,Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width))) == ShapeLaws.elements(output_shape(shape,out_height,out_width)) : Nat}
  %ShapeLaws.times_one(U32.to_nat(out_width)) :
    {Nat.mul(ShapeLaws.elements(leading),Nat.mul(U32.to_nat(out_height),_)) == ShapeLaws.elements(output_shape(shape,out_height,out_width)) : Nat}
  Equal.sym(Nat,ShapeLaws.elements(List.append(&2,U32,leading,[out_height,out_width])),Nat.mul(ShapeLaws.elements(leading),ShapeLaws.elements([out_height,out_width])),
    appended(leading,[out_height,out_width]))

# The public count selection: a Ready result has its planes and its counts.
def counts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,planes: Maybe<&2,U32>,count: Maybe<&2,U32>,
  +logical: Logical,+pad: Element,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+out_height: U32,+out_width: U32,
  +input: Buffer.Model<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,+limits: Machine.Limits,
  +actual: {limits == Default.limits() : Machine.Limits},+input_ready: Buffer.Sized<Element,input>,+values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  +planes_exact: {Window.planes(shape) == planes : Maybe<&2,U32>},
  +count_exact: {Tensors.count_of(output_shape(shape,out_height,out_width)) == count : Maybe<&2,U32>},
  accepted: {Dense.tensor_ready(~Element,Window.pool_counts(~Element,~operation,name,pad,planes,count,Access.Pool{height,width,kernel,stride,padding,out_height,out_width},Buffer.pack(~Element,input),shape,output_shape(shape,out_height,out_width),fill,workers)) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,Access.Pool{height,width,kernel,stride,padding,out_height,out_width},pad),output_shape(shape,out_height,out_width),Window.pool_counts(~Element,~operation,name,pad,planes,count,Access.Pool{height,width,kernel,stride,padding,out_height,out_width},Buffer.pack(~Element,input),shape,output_shape(shape,out_height,out_width),fill,workers)>:
  match planes count:
    case Some{+planes} Some{+count}:
      middled(~Element,~Logical,~original,~operation,name,Tensors.count_of(Window.middle_shape(shape,Access.Pool{height,width,kernel,stride,padding,out_height,out_width})),logical,pad,Access.Pool{height,width,kernel,stride,padding,out_height,out_width},planes,count,input,shape,output_shape(shape,out_height,out_width),fill,workers,limits,
        actual,input_ready,values,planes_cells(shape,height,out_width,planes,planes_exact),planes_cells(shape,out_height,out_width,planes,planes_exact),count_exact,{==},accepted)
    case Some{planes} None{}: Empty.absurd(GatherValues.Shaped<Element,Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,Access.Pool{height,width,kernel,stride,padding,out_height,out_width},pad),output_shape(shape,out_height,out_width),Window.pool_counts(~Element,~operation,name,pad,Some{planes},None{},Access.Pool{height,width,kernel,stride,padding,out_height,out_width},Buffer.pack(~Element,input),shape,output_shape(shape,out_height,out_width),fill,workers)>,WordEncoding.false_true(accepted))
    case None{} Some{count}: Empty.absurd(GatherValues.Shaped<Element,Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,Access.Pool{height,width,kernel,stride,padding,out_height,out_width},pad),output_shape(shape,out_height,out_width),Window.pool_counts(~Element,~operation,name,pad,None{},Some{count},Access.Pool{height,width,kernel,stride,padding,out_height,out_width},Buffer.pack(~Element,input),shape,output_shape(shape,out_height,out_width),fill,workers)>,WordEncoding.false_true(accepted))
    case None{} None{}: Empty.absurd(GatherValues.Shaped<Element,Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,Access.Pool{height,width,kernel,stride,padding,out_height,out_width},pad),output_shape(shape,out_height,out_width),Window.pool_counts(~Element,~operation,name,pad,None{},None{},Access.Pool{height,width,kernel,stride,padding,out_height,out_width},Buffer.pack(~Element,input),shape,output_shape(shape,out_height,out_width),fill,workers)>,WordEncoding.false_true(accepted))

# The shape of a pooled tensor.
def result_shape(+shape: List<&2,U32>,+kernel: U32,+stride: U32,+padding: U32) -> List<&2,U32>:
  +rank = List.length(&2,U32,shape)
  output_shape(shape,Window.out_extent(Layout.extent_at(shape,Nat.sub(rank,2n)),kernel,stride,padding),Window.out_extent(Layout.extent_at(shape,Nat.sub(rank,1n)),kernel,stride,padding))

def modeled(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +logical: Logical,+pad: Element,+kernel: U32,+stride: U32,+padding: U32,+input: Buffer.Model<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +input_ready: Buffer.Sized<Element,input>,+values: Dense.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),
  accepted: {Dense.tensor_ready(~Element,Window.pool2d(~Element,~operation,name,pad,Tensors.Tensor{Buffer.pack(~Element,input),shape,fill,Tensors.Usable{},workers},kernel,stride,padding)) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,Window.geometry(shape,kernel,stride,padding),pad),result_shape(shape,kernel,stride,padding),Window.pool2d(~Element,~operation,name,pad,Tensors.Tensor{Buffer.pack(~Element,input),shape,fill,Tensors.Usable{},workers},kernel,stride,padding)>:
  +rank = List.length(&2,U32,shape)
  +height = Layout.extent_at(shape,Nat.sub(rank,2n))
  +width = Layout.extent_at(shape,Nat.sub(rank,1n))
  +out_height = Window.out_extent(height,kernel,stride,padding)
  +out_width = Window.out_extent(width,kernel,stride,padding)
  counts(~Element,~Logical,~original,~operation,name,Window.planes(shape),Tensors.count_of(output_shape(shape,out_height,out_width)),logical,pad,height,width,kernel,stride,padding,out_height,out_width,
    input,shape,fill,workers,Default.limits(),{==},input_ready,values,{==},{==},accepted)

# A Ready dense tensor with complete values, pooled by the public call: when
# the call returns Ready, its result is a Ready dense tensor of the pooled
# shape that holds the pooled value of the logical input at every index. The
# Ready result is the only premise about the call; its guards are read from
# the call itself.
def shaped(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,-tensor: Tensors.Tensor<Element>,
  +logical: Logical,+pad: Element,+kernel: U32,+stride: U32,+padding: U32,+shape: List<&2,U32>,
  evidence: GatherValues.Shaped<Element,Logical,original,logical,shape,tensor>,
  accepted: {Dense.tensor_ready(~Element,Window.pool2d(~Element,~operation,name,pad,tensor,kernel,stride,padding)) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,Window.geometry(shape,kernel,stride,padding),pad),result_shape(shape,kernel,stride,padding),Window.pool2d(~Element,~operation,name,pad,tensor,kernel,stride,padding)>:
  GatherValues.Shaped{+model,+fill,+workers,+equation,Shape.Both{Dense.WellFormed{length_exact,content_ready},source_values}} = evidence
  %Equal.sym(Tensors.Tensor<Element>,tensor,Tensors.Tensor{Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers},equation) : GatherValues.Shaped<Element,Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,Window.geometry(shape,kernel,stride,padding),pad),result_shape(shape,kernel,stride,padding),Window.pool2d(~Element,~operation,name,pad,_,kernel,stride,padding)>
  modeled(~Element,~Logical,~original,~operation,name,logical,pad,kernel,stride,padding,model,shape,fill,workers,Buffer.ready_content(~Element,model,content_ready),source_values,
    %equation : {Dense.tensor_ready(~Element,Window.pool2d(~Element,~operation,name,pad,_,kernel,stride,padding)) == True{} : Bool}
    accepted)

# The pool of a Ready dense tensor with a contract, as a shaped result.
def dense_shaped(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +logical: Logical,+pad: Element,+kernel: U32,+stride: U32,+padding: U32,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Dense.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers}),
  accepted: {Dense.tensor_ready(~Element,Window.pool2d(~Element,~operation,name,pad,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)) == True{} : Bool}) ->
  GatherValues.Shaped<Element,Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,Window.geometry(shape,kernel,stride,padding),pad),result_shape(shape,kernel,stride,padding),Window.pool2d(~Element,~operation,name,pad,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)>:
  Shape.Both{Shape.Both{well_formed,source_ready},source_values} = contract
  shaped(~Element,~Logical,~original,~operation,name,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers},logical,pad,kernel,stride,padding,shape,
    GatherValues.shaped_observed(~Element,~Logical,~original,storage,logical,shape,fill,workers,Buffer.observe(~Element,storage),Shape.Both{well_formed,source_values}),
    accepted)

def dense(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +logical: Logical,+pad: Element,+kernel: U32,+stride: U32,+padding: U32,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Dense.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers}),
  accepted: {Dense.tensor_ready(~Element,Window.pool2d(~Element,~operation,name,pad,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)) == True{} : Bool}) ->
  Dense.Contract(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,Window.geometry(shape,kernel,stride,padding),pad),Window.pool2d(~Element,~operation,name,pad,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)):
  GatherValues.shaped_contract(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),context(~Element,~Logical,logical,Window.geometry(shape,kernel,stride,padding),pad),result_shape(shape,kernel,stride,padding),Window.pool2d(~Element,~operation,name,pad,Tensors.Tensor{storage,shape,fill,Tensors.Usable{},workers},kernel,stride,padding),
    dense_shaped(~Element,~Logical,~original,~operation,name,logical,pad,kernel,stride,padding,storage,shape,fill,workers,contract,accepted))
