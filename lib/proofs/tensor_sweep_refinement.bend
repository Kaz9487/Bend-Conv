# Ordered window reduction along one axis: each output folds the samples of
# its line in increasing coordinate, starting from pad. A sample outside the
# axis contributes pad. No associativity law is required.
import Base
import ../tensor_access.bend as Access
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./traversal_array_refinement.bend as ArrayLaws
import ./tensor_access_refinement.bend as AccessLaws
import ../storage_buffer.bend as Storage
import ../tensor_operations.bend as Operations
import ./storage_certificate.bend as Certificate
import ./tensor_access_certificate.bend as AccessCertificate
import ./tensor_operations_refinement.bend as OperationsLaws
import ./ordered_fold_refinement.bend as Fold

def sample_when(~Element: Data,pad: Element,valid: Bool,tree: Tree.Tree<Element>,index: U32) -> Element:
  match valid:
    case False{}: pad
    case True{}: Tree.read(Element,tree,index)

def read_when(~Element: Data,+pad: Element,valid: Bool,+tree: Tree.Tree<Element>,+index: U32) ->
  {Access.window_read(~Element,pad,valid,Tree.pack(Element,tree),index) ==
    (Tree.pack(Element,tree),sample_when(~Element,pad,valid,tree,index)) : Array<Element> & Element}:
  match valid:
    case False{}: {==}
    case True{}: StorageLaws.read(Element,tree,index)

def sample_line(~Element: Data,tree: Tree.Tree<Element>,line: Access.Line<Element>,position: U32) -> Element:
  Access.Line{+base,+first,+extent,+step,+padding,pad} = line
  +coordinate = (first + position - padding : U32)
  sample_when(~Element,pad,U32.is_lt(coordinate,extent),tree,(base + coordinate * step : U32))

def read_line(~Element: Data,+tree: Tree.Tree<Element>,+line: Access.Line<Element>,+position: U32) ->
  {Access.read_line(~Element,Tree.pack(Element,tree),line,position) ==
    (Tree.pack(Element,tree),sample_line(~Element,tree,line,position)) : Array<Element> & Element}:
  Access.Line{+base,+first,+extent,+step,+padding,pad} = line
  +coordinate = (first + position - padding : U32)
  read_when(~Element,pad,U32.is_lt(coordinate,extent),tree,(base + coordinate * step : U32))

def line_values(~Element: Data,~operation: Element -> Element -> Element,count: Nat,+position: U32,+tree: Tree.Tree<Element>,+line: Access.Line<Element>,accumulator: Element) -> Element:
  Fold.values(~Tree.Tree<Element>,~Element,~Element,~Access.Line<Element>,~sample_line(~Element),~operation,count,position,tree,line,accumulator)

def lines(~Element: Data,~operation: Element -> Element -> Element,+count: Nat,+position: U32,+tree: Tree.Tree<Element>,+line: Access.Line<Element>,+initial: Element) ->
  {Traversal.fold(~Array<Element>,~Element,~Element,~Access.Line<Element>,~Access.read_line(~Element),~operation,
      count,position,Tree.pack(Element,tree),initial,line) ==
    (Tree.pack(Element,tree),line_values(~Element,~operation,count,position,tree,line,initial)) : Array<Element> & Element}:
  Fold.fold(~Array<Element>,~Tree.Tree<Element>,~Element,~Element,~Access.Line<Element>,~ArrayLaws.pack_tree(~Element),
    ~Access.read_line(~Element),~sample_line(~Element),~operation,~read_line(~Element),count,position,tree,line,initial)

def sample_swept(~Element: Data,~operation: Element -> Element -> Element,tree: Tree.Tree<Element>,context: Access.SweepContext<Element>,position: U32) -> Element:
  Access.SweepContext{+sweep,+pad} = context
  Access.Sweep{extent,step,kernel,stride,padding,out_extent,origin} = sweep
  line_values(~Element,~operation,U32.to_nat(kernel),0,tree,Access.line_at(~Element,sweep,pad,position),pad)

def read_swept(~Element: Data,~operation: Element -> Element -> Element,+tree: Tree.Tree<Element>,+context: Access.SweepContext<Element>,+position: U32) ->
  {Access.read_swept(~Element,~operation,Tree.pack(Element,tree),context,position) ==
    (Tree.pack(Element,tree),sample_swept(~Element,~operation,tree,context,position)) : Array<Element> & Element}:
  Access.SweepContext{+sweep,+pad} = context
  Access.Sweep{extent,step,kernel,stride,padding,out_extent,origin} = sweep
  lines(~Element,~operation,U32.to_nat(kernel),0,tree,Access.line_at(~Element,sweep,pad,position),pad)

def sweep_model(~Element: Data,~operation: Element -> Element -> Element,count: Nat,context: Access.SweepContext<Element>,input: Tree.Tree<Element>,output: Tree.Tree<Element>) -> AccessLaws.ModelResult<Tree.Tree<Element>,Element>:
  AccessLaws.finish_model(~Tree.Tree<Element>,~Element,~Access.SweepContext<Element>,
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Access.SweepContext<Element>>,
      ~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Access.SweepContext<Element>,~sample_swept(~Element,~operation),~Access.swept_index(~Element)),count,
      Traversal.State{input,output,0,context}))

def sweep(~Element: Data,~operation: Element -> Element -> Element,+count: Nat,+context: Access.SweepContext<Element>,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) ->
  {Access.sweep_run(~Element,~operation,count,context,Tree.pack(Element,input),Tree.pack(Element,output)) ==
    AccessLaws.pack_result(~Array<Element>,~Tree.Tree<Element>,~Element,~ArrayLaws.pack_tree(~Element),sweep_model(~Element,~operation,count,context,input,output)) :
    Array<Element> & Array<Element>}:
  AccessLaws.finish_relation(~Array<Element>,~Tree.Tree<Element>,~Element,~Access.SweepContext<Element>,~Access.SweepContext<Element>,~ArrayLaws.pack_tree(~Element),
    ~ArrayLaws.represent(~Array<Element>,~Tree.Tree<Element>,~Element,~Access.SweepContext<Element>,~ArrayLaws.pack_tree(~Element)),
    ~AccessLaws.indexed_finished(~Array<Element>,~Tree.Tree<Element>,~Element,~Access.SweepContext<Element>,~ArrayLaws.pack_tree(~Element)),
    Traversal.run(~Array<Element>,~Element,~Access.SweepContext<Element>,~Access.read_swept(~Element,~operation),~Access.swept_index(~Element),count,
      Traversal.State{Tree.pack(Element,input),Tree.pack(Element,output),0,context}),
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<Element>,Tree.Tree<Element>,Access.SweepContext<Element>>,
      ~ArrayLaws.model_step(~Tree.Tree<Element>,~Element,~Access.SweepContext<Element>,~sample_swept(~Element,~operation),~Access.swept_index(~Element)),count,
      Traversal.State{input,output,0,context}),
    ArrayLaws.traversal_refines_model(~Array<Element>,~Tree.Tree<Element>,~Element,~Access.SweepContext<Element>,~ArrayLaws.pack_tree(~Element),
      ~Access.read_swept(~Element,~operation),~sample_swept(~Element,~operation),~Access.swept_index(~Element),~read_swept(~Element,~operation),
      count,Traversal.State{input,output,0,context}))

type Call<-Element: Data> is Data:
  Call{valid: Bool,count: Maybe<&2,U32>,context: Access.SweepContext<Element>,input_length: Nat,output_length: Nat}

def checked_model(~Element: Data,~operation: Element -> Element -> Element,call: Call<Element>,input: Tree.Tree<Element>,output: Tree.Tree<Element>) -> Operations.Outcome<OperationsLaws.PairOwners(Element)>:
  Call{valid,count,context,input_length,output_length} = call
  match valid count:
    case True{} Some{count}:
      Operations.Accepted{OperationsLaws.modeled_pair(~Element,input_length,output_length,sweep_model(~Element,~operation,U32.to_nat(count),context,input,output))}
    case valid count:
      Operations.Rejected{(OperationsLaws.Owner{Tree.pack(Element,input),input_length},OperationsLaws.Owner{Tree.pack(Element,output),output_length})}

def proposition(~Element: Data,~operation: Element -> Element -> Element,call: Call<Element>,-input: Array<Element>,-output: Array<Element>,-input_tree: Tree.Tree<Element>,-output_tree: Tree.Tree<Element>) -> Type:
  Call{valid,count,context,input_length,output_length} = call
  @input_certificate: Certificate.Certificate<Element,input> -> @output_certificate: Certificate.Certificate<Element,output> ->
  {OperationsLaws.observe_pair_outcome(~Element,Operations.sweep_when(~Element,~operation,valid,count,context,
      Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate})) ==
    checked_model(~Element,~operation,Call{valid,count,context,input_length,output_length},input_tree,output_tree) : Operations.Outcome<OperationsLaws.PairOwners(Element)>}

def packed_result(~Element: Data,~operation: Element -> Element -> Element,+valid: Bool,+count: Maybe<&2,U32>,+context: Access.SweepContext<Element>,+input_length: Nat,+output_length: Nat,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,
  +input_certificate: Certificate.Certificate<Element,Tree.pack(Element,input)>,+output_certificate: Certificate.Certificate<Element,Tree.pack(Element,output)>) ->
  {OperationsLaws.observe_pair_outcome(~Element,Operations.sweep_when(~Element,~operation,valid,count,context,
      Storage.Buffer{Tree.pack(Element,input),input_length,input_certificate},Storage.Buffer{Tree.pack(Element,output),output_length,output_certificate})) ==
    checked_model(~Element,~operation,Call{valid,count,context,input_length,output_length},input,output) : Operations.Outcome<OperationsLaws.PairOwners(Element)>}:
    match valid count:
      case False{} None{}: {==}
      case False{} Some{count}: {==}
      case True{} None{}: {==}
      case True{} Some{+count}:
        Equal.cong(OperationsLaws.PairOwners(Element),Operations.Outcome<OperationsLaws.PairOwners(Element)>,value => Operations.Accepted{value},
          OperationsLaws.observe_pair(~Element,Operations.pair_buffers(~Element,input_length,output_length,
            Access.sweep_run(~Element,~operation,U32.to_nat(count),context,Tree.pack(Element,input),Tree.pack(Element,output)),
            AccessCertificate.sweep(~Element,~operation,U32.to_nat(count),context,Tree.pack(Element,input),Tree.pack(Element,output),input_certificate,output_certificate))),
          OperationsLaws.modeled_pair(~Element,input_length,output_length,sweep_model(~Element,~operation,U32.to_nat(count),context,input,output)),
          OperationsLaws.pair_finished(~Element,input_length,output_length,
            Access.sweep_run(~Element,~operation,U32.to_nat(count),context,Tree.pack(Element,input),Tree.pack(Element,output)),
            AccessCertificate.sweep(~Element,~operation,U32.to_nat(count),context,Tree.pack(Element,input),Tree.pack(Element,output),input_certificate,output_certificate),
            sweep_model(~Element,~operation,U32.to_nat(count),context,input,output),sweep(~Element,~operation,U32.to_nat(count),context,input,output)))

def packed(~Element: Data,~operation: Element -> Element -> Element,call: Call<Element>,input: Tree.Tree<Element>,output: Tree.Tree<Element>) ->
  proposition(~Element,~operation,call,Tree.pack(Element,input),Tree.pack(Element,output),input,output):
  Call{valid,count,context,input_length,output_length} = call
  input_certificate => output_certificate =>
    packed_result(~Element,~operation,valid,count,context,input_length,output_length,input,output,input_certificate,output_certificate)

def observed(~Element: Data,~operation: Element -> Element -> Element,+call: Call<Element>,-input: Array<Element>,-output: Array<Element>,
  input_observation: StorageLaws.Observation<Element,input>,output_observation: StorageLaws.Observation<Element,output>) ->
  proposition(~Element,~operation,call,input,output,Tree.reflect(Element,input),Tree.reflect(Element,output)):
  StorageLaws.Observation{+input_tree,input_packed,input_reflected} = input_observation
  StorageLaws.Observation{+output_tree,output_packed,output_reflected} = output_observation
  %input_reflected : proposition(~Element,~operation,call,input,output,_,Tree.reflect(Element,output))
  %output_reflected : proposition(~Element,~operation,call,input,output,input_tree,_)
  %input_packed : proposition(~Element,~operation,call,_,output,input_tree,output_tree)
  %output_packed : proposition(~Element,~operation,call,Tree.pack(Element,input_tree),_,input_tree,output_tree)
  packed(~Element,~operation,call,input_tree,output_tree)

def call(~Element: Data,pad: Element,+sweep: Access.Sweep,+outer: U32,+input_length: Nat,input_capacity: U32,+output_length: Nat,output_capacity: U32) -> Call<Element>:
  Call{Operations.sweep_geometry(sweep) && Operations.holds(Operations.sweep_input(sweep,outer),input_length,input_capacity) &&
    Operations.holds(Operations.sweep_output(sweep,outer),output_length,output_capacity),Operations.sweep_output(sweep,outer),
    Access.SweepContext{sweep,pad},input_length,output_length}

def denotation(~Element: Data,~operation: Element -> Element -> Element,pad: Element,input: Storage.Buffer<Element>,sweep: Access.Sweep,outer: U32,output: Storage.Buffer<Element>) -> Operations.Outcome<OperationsLaws.PairOwners(Element)>:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Storage.Buffer{output,+output_length,output_certificate} = output
  checked_model(~Element,~operation,call(~Element,pad,sweep,outer,input_length,Storage.capacity(~Element,input,input_certificate),
      output_length,Storage.capacity(~Element,output,output_certificate)),Tree.reflect(Element,input),Tree.reflect(Element,output))

# The actual checked entry, including both rejection and successful execution.
# Buffer ownership supplies its certificate; no balance or readback premise is
# delegated to the caller.
def checked(~Element: Data,~operation: Element -> Element -> Element,+pad: Element,input: Storage.Buffer<Element>,+sweep: Access.Sweep,+outer: U32,output: Storage.Buffer<Element>) ->
  {OperationsLaws.observe_pair_outcome(~Element,Operations.sweep(~Element,~operation,pad,input,sweep,outer,output)) ==
    denotation(~Element,~operation,pad,input,sweep,outer,output) : Operations.Outcome<OperationsLaws.PairOwners(Element)>}:
  Storage.Buffer{input,+input_length,Certificate.Certificate{+input_depth,+input_equation}} = input
  Storage.Buffer{output,+output_length,Certificate.Certificate{+output_depth,+output_equation}} = output
  observed(~Element,~operation,call(~Element,pad,sweep,outer,input_length,U32.shln(1,input_depth),output_length,U32.shln(1,output_depth)),
    input,output,StorageLaws.observe(Element,input),StorageLaws.observe(Element,output))(
      Certificate.Certificate{input_depth,input_equation})(Certificate.Certificate{output_depth,output_equation})
