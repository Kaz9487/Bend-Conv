# Discharge the reader premises for the actual tensor traversal kernels.
# Public operations use only the flat certificates returned at their boundary.
import Base
import ../tensor_access.bend as Access
import ../tensor_view.bend as Views
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../tensor_cursor.bend as Cursor
import ./tensor_cursor_witness.bend as CursorWitness
import ./traversal_visit_witness.bend as VisitWitness
import ./storage_certificate.bend as Certificate
import ./storage_tree_model.bend as Tree
import ./storage_witness.bend as Witness
import ./traversal_witness.bend as TraversalWitness
import ./traversal_certificate.bend as TraversalCertificate

def read_index(~Element: Data,-input: Array<Element>,context: Unit,index: U32,evidence: Witness.ArrayWitness<Element,input>) ->
  TraversalWitness.Sample<Array<Element>,Element,input,Traversal.read_index(~Element,input,context,index)>:
  TraversalWitness.array_sample(~Element,input,Array.get(Element,input,index),Witness.observed(Element,input,index,evidence))

type CopyCall is Data:
  CopyCall{fast: Bool,count: Nat,source: Views.View,destination: U32}

def copy_operation(~Element: Data,call: CopyCall,input: Array<Element>,output: Array<Element>) -> Array<Element> & Array<Element>:
  CopyCall{fast,count,source,destination} = call
  Access.copy_run(~Element,fast,count,source,destination,input,output)

def combine_operation(~Element: Data,~operation: Element -> Element -> Element,call: CopyCall,input: Array<Element>,output: Array<Element>) -> Array<Element> & Array<Element>:
  CopyCall{fast,count,source,destination} = call
  Access.combine_run(~Element,~operation,fast,count,source,destination,input,output)

# A strided operation keeps its owners through each part of a run: the stride
# steps, the state handed to the cursor, and the cursor step that ends the run.
# The lemmas are generic in the two element steps and in the evidence that
# each of them keeps.
def last_of_run(~Element: Data,+depth: Nat,extent: Nat,stride: U32,rewind: U32,rest: Cursor.Depth1(),-input: Array<Element>,
  -state: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,
  evidence: TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Stride,input,depth,state>) ->
  TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Cursor,input,depth,Access.last_of_run(~Element,extent,stride,rewind,rest,state)>:
  match evidence:
    case TraversalWitness.StateEvidence{output,position,Cursor.Stride{address,step},equation,output_evidence}:
      %Equal.sym(Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,state,Traversal.State{input,output,position,Cursor.Stride{address,step}},equation) :
        TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Cursor,input,depth,Access.last_of_run(~Element,extent,stride,rewind,rest,_)>
      TraversalWitness.StateEvidence{output,position,Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,1n,stride,rewind},rest},address},{==},output_evidence}

# A step keeps the evidence of its state.
def Keeps(~Element: Data,~Context: Data,~step: Traversal.State<&1,&1,Array<Element>,Array<Element>,Context> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Context>) -> Type:
  @+depth: Nat -> @-input: Array<Element> -> @-state: Traversal.State<&1,&1,Array<Element>,Array<Element>,Context> -> Witness.ArrayWitness<Element,input> ->
    TraversalWitness.StateEvidence<Array<Element>,Element,Context,input,depth,state> -> TraversalWitness.StateEvidence<Array<Element>,Element,Context,input,depth,step(state)>

def loop(~Element: Data,~Context: Data,~step: Traversal.State<&1,&1,Array<Element>,Array<Element>,Context> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Context>,
  ~keeps: Keeps(~Element,~Context,~step),
  +depth: Nat,count: Nat,-input: Array<Element>,-state: Traversal.State<&1,&1,Array<Element>,Array<Element>,Context>,+input_evidence: Witness.ArrayWitness<Element,input>,
  evidence: TraversalWitness.StateEvidence<Array<Element>,Element,Context,input,depth,state>) ->
  TraversalWitness.StateEvidence<Array<Element>,Element,Context,input,depth,Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Context>,~step,count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      loop(~Element,~Context,~step,~keeps,depth,rest,input,step(state),input_evidence,keeps(depth,input,state,input_evidence,evidence))

def run_axis(~Element: Data,~inside: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  ~inside_keeps: Keeps(~Element,~Cursor.Stride,~inside),
  ~last_keeps: Keeps(~Element,~Cursor.Cursor,~last),
  +depth: Nat,remaining: Nat,extent: Nat,+stride: U32,rewind: U32,rest: Cursor.Depth1(),+address: U32,
  -input: Array<Element>,-output: Array<Element>,position: U32,+input_evidence: Witness.ArrayWitness<Element,input>,
  output_evidence: Witness.Witness(Element,depth,output)) ->
  TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Cursor,input,depth,Access.run_axis(~Element,~inside,~last,remaining,extent,stride,rewind,rest,address,input,output,position)>:
  match remaining:
    case 0n:
      last_keeps(depth,input,Traversal.State{input,output,position,Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,0n,stride,rewind},rest},address}},input_evidence,
        TraversalWitness.StateEvidence{output,position,Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,0n,stride,rewind},rest},address},{==},output_evidence})
    case 1n+count:
      last_keeps(depth,input,
        Access.last_of_run(~Element,extent,stride,rewind,rest,Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~inside,count,Traversal.State{input,output,position,Cursor.Stride{address,stride}})),
        input_evidence,
        last_of_run(~Element,depth,extent,stride,rewind,rest,input,Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~inside,count,Traversal.State{input,output,position,Cursor.Stride{address,stride}}),
          loop(~Element,~Cursor.Stride,~inside,~inside_keeps,depth,count,input,Traversal.State{input,output,position,Cursor.Stride{address,stride}},input_evidence,
            TraversalWitness.StateEvidence{output,position,Cursor.Stride{address,stride},{==},output_evidence})))

def run_step(~Element: Data,~inside: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  ~inside_keeps: Keeps(~Element,~Cursor.Stride,~inside),
  ~last_keeps: Keeps(~Element,~Cursor.Cursor,~last),
  +depth: Nat,-input: Array<Element>,-state: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  +input_evidence: Witness.ArrayWitness<Element,input>,evidence: TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Cursor,input,depth,state>) ->
  TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Cursor,input,depth,Access.run_step(~Element,~inside,~last,state)>:
  match evidence:
    case TraversalWitness.StateEvidence{output,position,Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,remaining,stride,rewind},rest},address},equation,output_evidence}:
      %Equal.sym(Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,state,
        Traversal.State{input,output,position,Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,remaining,stride,rewind},rest},address}},equation) :
        TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Cursor,input,depth,Access.run_step(~Element,~inside,~last,_)>
      run_axis(~Element,~inside,~last,~inside_keeps,~last_keeps,depth,remaining,extent,stride,rewind,rest,address,input,output,position,input_evidence,output_evidence)

def run_loop(~Element: Data,~inside: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  ~inside_keeps: Keeps(~Element,~Cursor.Stride,~inside),
  ~last_keeps: Keeps(~Element,~Cursor.Cursor,~last),
  +depth: Nat,count: Nat,-input: Array<Element>,-state: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  +input_evidence: Witness.ArrayWitness<Element,input>,evidence: TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Cursor,input,depth,state>) ->
  TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Cursor,input,depth,Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,~Access.run_step(~Element,~inside,~last),count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      run_loop(~Element,~inside,~last,~inside_keeps,~last_keeps,depth,rest,input,Access.run_step(~Element,~inside,~last,state),input_evidence,
        run_step(~Element,~inside,~last,~inside_keeps,~last_keeps,depth,input,state,input_evidence,evidence))

def last_loop(~Element: Data,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  ~last_keeps: Keeps(~Element,~Cursor.Cursor,~last),
  +depth: Nat,count: Nat,-input: Array<Element>,-state: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  +input_evidence: Witness.ArrayWitness<Element,input>,evidence: TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Cursor,input,depth,state>) ->
  TraversalWitness.StateEvidence<Array<Element>,Element,Cursor.Cursor,input,depth,Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,~last,count,state)>:
  loop(~Element,~Cursor.Cursor,~last,~last_keeps,depth,count,input,state,input_evidence,evidence)

def stepped(~Element: Data,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  ~last_keeps: Keeps(~Element,~Cursor.Cursor,~last),
  +depth: Nat,count: Nat,cursor: Cursor.Cursor,destination: U32,-input: Array<Element>,-output: Array<Element>,
  input_evidence: Witness.ArrayWitness<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,
    Traversal.finish(~Array<Element>,~Element,~Cursor.Cursor,Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,~last,count,
      Traversal.State{input,output,destination,cursor}))>:
  TraversalCertificate.completed(~Array<Element>,~Element,~Cursor.Cursor,~TraversalWitness.array_evidence(~Element),
    ~Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,~last),~last_loop(~Element,~last,~last_keeps),
    count,destination,cursor,input,output,input_evidence,depth,equation)

def guarded(~Element: Data,~inside: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  ~inside_keeps: Keeps(~Element,~Cursor.Stride,~inside),
  ~last_keeps: Keeps(~Element,~Cursor.Cursor,~last),
  +depth: Nat,full_run: Bool,whole_runs: Bool,runs: Nat,count: Nat,cursor: Cursor.Cursor,destination: U32,
  -input: Array<Element>,-output: Array<Element>,input_evidence: Witness.ArrayWitness<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,Access.run_guarded(~Element,~inside,~last,full_run,whole_runs,runs,count,cursor,destination,input,output)>:
  match full_run whole_runs:
    case True{} True{}:
      TraversalCertificate.completed(~Array<Element>,~Element,~Cursor.Cursor,~TraversalWitness.array_evidence(~Element),
        ~Loop.run(~Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,~Access.run_step(~Element,~inside,~last)),~run_loop(~Element,~inside,~last,~inside_keeps,~last_keeps),
        runs,destination,cursor,input,output,input_evidence,depth,equation)
    case True{} False{}: stepped(~Element,~last,~last_keeps,depth,count,cursor,destination,input,output,input_evidence,equation)
    case False{} whole_runs: stepped(~Element,~last,~last_keeps,depth,count,cursor,destination,input,output,input_evidence,equation)

def remaining_run(~Element: Data,~inside: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  ~inside_keeps: Keeps(~Element,~Cursor.Stride,~inside),
  ~last_keeps: Keeps(~Element,~Cursor.Cursor,~last),
  +depth: Nat,remaining: Nat,+extent: Nat,stride: U32,rewind: U32,rest: Cursor.Depth1(),address: U32,+count: Nat,destination: U32,
  -input: Array<Element>,-output: Array<Element>,input_evidence: Witness.ArrayWitness<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,Access.run_remaining(~Element,~inside,~last,remaining,extent,stride,rewind,rest,address,count,destination,input,output)>:
  match remaining:
    case 0n:
      stepped(~Element,~last,~last_keeps,depth,count,Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,0n,stride,rewind},rest},address},destination,input,output,input_evidence,equation)
    case 1n+ +inner_count:
      guarded(~Element,~inside,~last,~inside_keeps,~last_keeps,depth,Nat.is_eq(extent,1n+inner_count),Nat.is_eq(Nat.mul(Nat.div(count,1n+inner_count),1n+inner_count),count),Nat.div(count,1n+inner_count),count,
        Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,1n+inner_count,stride,rewind},rest},address},destination,input,output,input_evidence,equation)

def strided(~Element: Data,~inside: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Stride>,~last: Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor> -> Traversal.State<&1,&1,Array<Element>,Array<Element>,Cursor.Cursor>,
  ~inside_keeps: Keeps(~Element,~Cursor.Stride,~inside),
  ~last_keeps: Keeps(~Element,~Cursor.Cursor,~last),
  +depth: Nat,count: Nat,cursor: Cursor.Cursor,destination: U32,-input: Array<Element>,-output: Array<Element>,
  input_evidence: Witness.ArrayWitness<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,Access.run_strided(~Element,~inside,~last,count,cursor,destination,input,output)>:
  Cursor.Cursor{Cursor.Level{Cursor.Axis{extent,remaining,stride,rewind},rest},address} = cursor
  remaining_run(~Element,~inside,~last,~inside_keeps,~last_keeps,depth,remaining,extent,stride,rewind,rest,address,count,destination,input,output,input_evidence,equation)

def copied_strided(~Element: Data,+depth: Nat,count: Nat,cursor: Cursor.Cursor,destination: U32,-input: Array<Element>,-output: Array<Element>,
  input_evidence: Witness.ArrayWitness<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,Access.copy_strided(~Element,count,cursor,destination,input,output)>:
  strided(~Element,~Traversal.visit_step(~Array<Element>,~Element,~Cursor.Stride,~Cursor.read_stride(~Element)),
    ~Traversal.visit_step(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element)),
    ~VisitWitness.step(~Array<Element>,~Element,~Cursor.Stride,~TraversalWitness.array_evidence(~Element),~Cursor.read_stride(~Element),~CursorWitness.read_stride(~Element)),
    ~VisitWitness.step(~Array<Element>,~Element,~Cursor.Cursor,~TraversalWitness.array_evidence(~Element),~Cursor.read(~Element),~CursorWitness.read(~Element)),
    depth,count,cursor,destination,input,output,input_evidence,equation)

def updated_strided(~Element: Data,~operation: Element -> Element -> Element,+depth: Nat,count: Nat,cursor: Cursor.Cursor,destination: U32,
  -input: Array<Element>,-output: Array<Element>,input_evidence: Witness.ArrayWitness<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,Access.update_strided(~Element,~operation,count,cursor,destination,input,output)>:
  strided(~Element,~Traversal.visit_update_step(~Array<Element>,~Element,~Cursor.Stride,~Cursor.read_stride(~Element),~operation),
    ~Traversal.visit_update_step(~Array<Element>,~Element,~Cursor.Cursor,~Cursor.read(~Element),~operation),
    ~VisitWitness.update_step(~Array<Element>,~Element,~Cursor.Stride,~TraversalWitness.array_evidence(~Element),~Cursor.read_stride(~Element),~operation,~CursorWitness.read_stride(~Element)),
    ~VisitWitness.update_step(~Array<Element>,~Element,~Cursor.Cursor,~TraversalWitness.array_evidence(~Element),~Cursor.read(~Element),~operation,~CursorWitness.read(~Element)),
    depth,count,cursor,destination,input,output,input_evidence,equation)

type ScatterCall is Data:
  ScatterCall{count: Nat,target: Views.View,source: U32}

def scatter_operation(~Element: Data,call: ScatterCall,input: Array<Element>,output: Array<Element>) -> Array<Element> & Array<Element>:
  ScatterCall{count,target,source} = call
  Access.scatter_run(~Element,count,target,source,input,output)

def scattered(~Element: Data,+depth: Nat,call: ScatterCall,-input: Array<Element>,-output: Array<Element>,input_certificate: Certificate.Certificate<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,scatter_operation(~Element,call,input,output)>:
  match call:
    case ScatterCall{count,target,source}:
      strided(~Element,~Traversal.spread_step(~Element,~Cursor.Stride,~Cursor.target_stride),~Traversal.spread_step(~Element,~Cursor.Cursor,~Cursor.target),
        ~VisitWitness.spread_step(~Element,~Cursor.Stride,~Cursor.target_stride),~VisitWitness.spread_step(~Element,~Cursor.Cursor,~Cursor.target),
        depth,count,Cursor.prepare(target),source,input,output,Certificate.witness(Element,input,input_certificate),equation)

def copied(~Element: Data,+depth: Nat,call: CopyCall,-input: Array<Element>,-output: Array<Element>,input_certificate: Certificate.Certificate<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,copy_operation(~Element,call,input,output)>:
  match call:
    case CopyCall{True{},count,Views.View{axes,offset},destination}:
      TraversalCertificate.completed(~Array<Element>,~Element,~Traversal.Offsets,~TraversalWitness.array_evidence(~Element),
        ~Traversal.run(~Array<Element>,~Element,~Traversal.Offsets,~Traversal.source_offset(~Element),~Traversal.destination_offset),
        ~TraversalWitness.run(~Array<Element>,~Element,~Traversal.Offsets,~TraversalWitness.array_evidence(~Element),~Traversal.source_offset(~Element),~Traversal.destination_offset,~TraversalWitness.offset_read(~Element)),
        count,0,Traversal.Offsets{offset,destination},input,output,Certificate.witness(Element,input,input_certificate),depth,equation)
    case CopyCall{False{},count,source,destination}:
      copied_strided(~Element,depth,count,Cursor.prepare(source),destination,input,output,Certificate.witness(Element,input,input_certificate),equation)

def combined(~Element: Data,~operation: Element -> Element -> Element,+depth: Nat,call: CopyCall,-input: Array<Element>,-output: Array<Element>,
  input_certificate: Certificate.Certificate<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,combine_operation(~Element,~operation,call,input,output)>:
  match call:
    case CopyCall{True{},count,source,destination}:
      TraversalCertificate.completed(~Array<Element>,~Element,~Unit,~TraversalWitness.array_evidence(~Element),
        ~Traversal.update(~Array<Element>,~Element,~Element,~Unit,~Traversal.read_index(~Element),~operation,~Traversal.index),
        ~TraversalWitness.update(~Array<Element>,~Element,~Element,~Unit,~TraversalWitness.array_evidence(~Element),~Traversal.read_index(~Element),~operation,~Traversal.index,~read_index(~Element)),
        count,0,Unit{},input,output,Certificate.witness(Element,input,input_certificate),depth,equation)
    case CopyCall{False{},count,source,destination}:
      updated_strided(~Element,~operation,depth,count,Cursor.prepare(source),destination,input,output,Certificate.witness(Element,input,input_certificate),equation)

def copy(~Element: Data,fast: Bool,count: Nat,source: Views.View,destination: U32,-input: Array<Element>,-output: Array<Element>,
  input_certificate: Certificate.Certificate<Element,input>,output_certificate: Certificate.Certificate<Element,output>) ->
  TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),Access.copy_run(~Element,fast,count,source,destination,input,output)>:
  Certificate.Certificate{+depth,+equation} = input_certificate
  TraversalCertificate.operation(~Array<Element>,~Element,~CopyCall,~Certificate.family(~Element),~copy_operation(~Element),~copied(~Element),
    CopyCall{fast,count,source,destination},input,output,Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},output_certificate)

def scatter(~Element: Data,count: Nat,target: Views.View,source: U32,-input: Array<Element>,-output: Array<Element>,
  input_certificate: Certificate.Certificate<Element,input>,output_certificate: Certificate.Certificate<Element,output>) ->
  TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),Access.scatter_run(~Element,count,target,source,input,output)>:
  Certificate.Certificate{+depth,+equation} = input_certificate
  TraversalCertificate.operation(~Array<Element>,~Element,~ScatterCall,~Certificate.family(~Element),~scatter_operation(~Element),~scattered(~Element),
    ScatterCall{count,target,source},input,output,Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},output_certificate)

def combine(~Element: Data,~operation: Element -> Element -> Element,fast: Bool,count: Nat,source: Views.View,destination: U32,
  -input: Array<Element>,-output: Array<Element>,input_certificate: Certificate.Certificate<Element,input>,output_certificate: Certificate.Certificate<Element,output>) ->
  TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),Access.combine_run(~Element,~operation,fast,count,source,destination,input,output)>:
  Certificate.Certificate{+depth,+equation} = input_certificate
  TraversalCertificate.operation(~Array<Element>,~Element,~CopyCall,~Certificate.family(~Element),~combine_operation(~Element,~operation),~combined(~Element,~operation),
    CopyCall{fast,count,source,destination},input,output,Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},output_certificate)

type ReduceCall is Data:
  ReduceCall{fast: Bool,count: Nat,view: Views.View}

def reduce_operation(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,
  call: ReduceCall,storage: Array<Element>,initial: Accumulator) -> Array<Element> & Accumulator:
  ReduceCall{fast,count,view} = call
  Access.reduce_run(~Element,~Accumulator,~operation,fast,count,view,storage,initial)

def reduced(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,
  call: ReduceCall,-storage: Array<Element>,-initial: Accumulator,certificate: Certificate.Certificate<Element,storage>) ->
  {TraversalWitness.first(~Array<Element>,~Accumulator,reduce_operation(~Element,~Accumulator,~operation,call,storage,initial)) == storage : Array<Element>}:
  match call:
    case ReduceCall{True{},count,Views.View{axes,offset}}:
      TraversalWitness.fold(~Array<Element>,~Element,~Accumulator,~Traversal.Offsets,~TraversalWitness.array_evidence(~Element),~Traversal.source_offset(~Element),~operation,~TraversalWitness.offset_read(~Element),
        count,storage,0,Traversal.Offsets{offset,0},initial,Certificate.witness(Element,storage,certificate))
    case ReduceCall{False{},count,view}:
      TraversalWitness.fold_finished(~Array<Element>,~Accumulator,~Cursor.Cursor,storage,
        Traversal.visit_fold(~Array<Element>,~Element,~Accumulator,~Cursor.Cursor,~Cursor.read(~Element),~operation,count,Traversal.State{storage,initial,0,Cursor.prepare(view)}),
        VisitWitness.fold_run(~Array<Element>,~Element,~Accumulator,~Cursor.Cursor,~TraversalWitness.array_evidence(~Element),~Cursor.read(~Element),~operation,~CursorWitness.read(~Element),
          count,storage,Traversal.State{storage,initial,0,Cursor.prepare(view)},Certificate.witness(Element,storage,certificate),
          TraversalWitness.FoldEvidence{initial,0,Cursor.prepare(view),{==}}))

def reduce(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,fast: Bool,count: Nat,view: Views.View,
  -storage: Array<Element>,-initial: Accumulator,certificate: Certificate.Certificate<Element,storage>) ->
  Certificate.Certificate<Element,TraversalWitness.first(~Array<Element>,~Accumulator,Access.reduce_run(~Element,~Accumulator,~operation,fast,count,view,storage,initial))>:
  Certificate.Certificate{+depth,+equation} = certificate
  TraversalCertificate.fold_operation(~Array<Element>,~Accumulator,~ReduceCall,~Certificate.family(~Element),~reduce_operation(~Element,~Accumulator,~operation),~reduced(~Element,~Accumulator,~operation),
    ReduceCall{fast,count,view},storage,initial,Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation})

type IndexedCall is Data:
  IndexedCall{count: Nat,view: Views.View}

def indexed_operation(~Element: Data,~operation: U32 -> Element -> Element,call: IndexedCall,input: Array<Element>,output: Array<Element>) ->
  Array<Element> & Array<Element>:
  IndexedCall{count,view} = call
  Access.apply_indexed_run(~Element,~operation,count,view,input,output)

def indexed_sample(~Element: Data,~operation: U32 -> Element -> Element,-storage: Array<Element>,+position: U32,-result: Array<Element> & Element,
  sample: TraversalWitness.Sample<Array<Element>,Element,storage,result>) ->
  TraversalWitness.Sample<Array<Element>,Element,storage,Access.indexed_value(~Element,~operation,position,result)>:
  match sample:
    case TraversalWitness.Sample{value,equation}:
      %Equal.sym(Array<Element> & Element,result,(storage,value),equation) :
        TraversalWitness.Sample<Array<Element>,Element,storage,Access.indexed_value(~Element,~operation,position,_)>
      TraversalWitness.Sample{operation(position,value),{==}}

def indexed_read(~Element: Data,~operation: U32 -> Element -> Element,-storage: Array<Element>,context: Traversal.Offsets,index: U32,
  evidence: Witness.ArrayWitness<Element,storage>) ->
  TraversalWitness.Sample<Array<Element>,Element,storage,Access.read_indexed(~Element,~operation,storage,context,index)>:
  +index = index
  indexed_sample(~Element,~operation,storage,index,Traversal.source_offset(~Element,storage,context,index),
    TraversalWitness.offset_read(~Element,storage,context,index,evidence))

def indexed(~Element: Data,~operation: U32 -> Element -> Element,+depth: Nat,call: IndexedCall,-input: Array<Element>,-output: Array<Element>,
  input_certificate: Certificate.Certificate<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,indexed_operation(~Element,~operation,call,input,output)>:
  match call:
    case IndexedCall{count,Views.View{axes,offset}}:
      TraversalCertificate.completed(~Array<Element>,~Element,~Traversal.Offsets,~TraversalWitness.array_evidence(~Element),
        ~Traversal.run(~Array<Element>,~Element,~Traversal.Offsets,~Access.read_indexed(~Element,~operation),~Traversal.destination_offset),
        ~TraversalWitness.run(~Array<Element>,~Element,~Traversal.Offsets,~TraversalWitness.array_evidence(~Element),~Access.read_indexed(~Element,~operation),~Traversal.destination_offset,~indexed_read(~Element,~operation)),
        count,0,Traversal.Offsets{offset,0},input,output,Certificate.witness(Element,input,input_certificate),depth,equation)

def apply_indexed(~Element: Data,~operation: U32 -> Element -> Element,count: Nat,view: Views.View,-input: Array<Element>,-output: Array<Element>,
  input_certificate: Certificate.Certificate<Element,input>,output_certificate: Certificate.Certificate<Element,output>) ->
  TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),Access.apply_indexed_run(~Element,~operation,count,view,input,output)>:
  Certificate.Certificate{+depth,+equation} = input_certificate
  TraversalCertificate.operation(~Array<Element>,~Element,~IndexedCall,~Certificate.family(~Element),~indexed_operation(~Element,~operation),~indexed(~Element,~operation),
    IndexedCall{count,view},input,output,Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},output_certificate)

def window_read_sample(~Element: Data,+pad: Element,valid: Bool,-input: Array<Element>,+index: U32,
  evidence: Witness.ArrayWitness<Element,input>) ->
  TraversalWitness.Sample<Array<Element>,Element,input,Access.window_read(~Element,pad,valid,input,index)>:
  match valid:
    case False{}: TraversalWitness.Sample{pad,{==}}
    case True{}: TraversalWitness.array_sample(~Element,input,Array.get(Element,input,index),Witness.observed(Element,input,index,evidence))

# The one-axis window reduction: every sample is an element of the input or
# the pad, so the sweep returns its input and writes only its output.
def read_line_sample(~Element: Data,-input: Array<Element>,line: Access.Line<Element>,position: U32,
  evidence: Witness.ArrayWitness<Element,input>) ->
  TraversalWitness.Sample<Array<Element>,Element,input,Access.read_line(~Element,input,line,position)>:
  Access.Line{+base,+first,+extent,+step,+padding,+pad} = line
  +coordinate = (first + position - padding : U32)
  window_read_sample(~Element,pad,U32.is_lt(coordinate,extent),input,(base + coordinate * step : U32),evidence)

def line_sample(~Element: Data,-input: Array<Element>,
  -state: Traversal.State<&1,&2,Array<Element>,Element,Access.Line<Element>>,
  evidence: TraversalWitness.FoldEvidence<Array<Element>,Element,Access.Line<Element>,input,state>) ->
  TraversalWitness.Sample<Array<Element>,Element,input,Traversal.finish_fold(~Array<Element>,~Element,~Access.Line<Element>,state)>:
  match evidence:
    case TraversalWitness.FoldEvidence{accumulator,position,context,equation}:
      %Equal.sym(Traversal.State<&1,&2,Array<Element>,Element,Access.Line<Element>>,state,Traversal.State{input,accumulator,position,context},equation) :
        TraversalWitness.Sample<Array<Element>,Element,input,Traversal.finish_fold(~Array<Element>,~Element,~Access.Line<Element>,_)>
      TraversalWitness.Sample{accumulator,{==}}

def swept_sample(~Element: Data,~operation: Element -> Element -> Element,-input: Array<Element>,context: Access.SweepContext<Element>,position: U32,
  evidence: Witness.ArrayWitness<Element,input>) ->
  TraversalWitness.Sample<Array<Element>,Element,input,Access.read_swept(~Element,~operation,input,context,position)>:
  Access.SweepContext{+sweep,+pad} = context
  Access.Sweep{extent,step,+kernel,stride,padding,out_extent,origin} = sweep
  +position = position
  line_sample(~Element,input,
    Loop.run(~Traversal.State<&1,&2,Array<Element>,Element,Access.Line<Element>>,
      ~Traversal.fold_step(~Array<Element>,~Element,~Element,~Access.Line<Element>,~Access.read_line(~Element),~operation),
      U32.to_nat(kernel),Traversal.State{input,pad,0,Access.line_at(~Element,sweep,pad,position)}),
    TraversalWitness.fold_run(~Array<Element>,~Element,~Element,~Access.Line<Element>,~TraversalWitness.array_evidence(~Element),~Access.read_line(~Element),~operation,
      ~read_line_sample(~Element),U32.to_nat(kernel),input,Traversal.State{input,pad,0,Access.line_at(~Element,sweep,pad,position)},evidence,
      TraversalWitness.FoldEvidence{pad,0,Access.line_at(~Element,sweep,pad,position),{==}}))

type SweepCall<-Element: Data> is Data:
  SweepCall{count: Nat,context: Access.SweepContext<Element>}

def sweep_operation(~Element: Data,~operation: Element -> Element -> Element,call: SweepCall<Element>,input: Array<Element>,output: Array<Element>) ->
  Array<Element> & Array<Element>:
  SweepCall{count,context} = call
  Access.sweep_run(~Element,~operation,count,context,input,output)

def swept(~Element: Data,~operation: Element -> Element -> Element,+depth: Nat,call: SweepCall<Element>,-input: Array<Element>,-output: Array<Element>,
  input_certificate: Certificate.Certificate<Element,input>,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Array<Element>,Element,input,depth,sweep_operation(~Element,~operation,call,input,output)>:
  match call:
    case SweepCall{count,context}:
      TraversalCertificate.completed(~Array<Element>,~Element,~Access.SweepContext<Element>,~TraversalWitness.array_evidence(~Element),
        ~Traversal.run(~Array<Element>,~Element,~Access.SweepContext<Element>,~Access.read_swept(~Element,~operation),~Access.swept_index(~Element)),
        ~TraversalWitness.run(~Array<Element>,~Element,~Access.SweepContext<Element>,~TraversalWitness.array_evidence(~Element),~Access.read_swept(~Element,~operation),
          ~Access.swept_index(~Element),~swept_sample(~Element,~operation)),
        count,0,context,input,output,Certificate.witness(Element,input,input_certificate),depth,equation)

def sweep(~Element: Data,~operation: Element -> Element -> Element,count: Nat,context: Access.SweepContext<Element>,-input: Array<Element>,-output: Array<Element>,
  input_certificate: Certificate.Certificate<Element,input>,output_certificate: Certificate.Certificate<Element,output>) ->
  TraversalCertificate.Certificates<Array<Element>,Element,Certificate.family(~Element),Access.sweep_run(~Element,~operation,count,context,input,output)>:
  Certificate.Certificate{+depth,+equation} = input_certificate
  TraversalCertificate.operation(~Array<Element>,~Element,~SweepCall<Element>,~Certificate.family(~Element),~sweep_operation(~Element,~operation),~swept(~Element,~operation),
    SweepCall{count,context},input,output,Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},Certificate.Certificate{depth,equation},output_certificate)
