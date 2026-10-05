# Shape and owner preservation for traversals with consumed cursor state.
import Base
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./storage_witness.bend as Witness
import ./traversal_witness.bend as TraversalWitness

type Sample<-Reader: Type,-Element: Data,-Context: Data,-reader: Reader,-result: Reader & Traversal.Visited<Context,Element>> is Data:
  Sample{context: Context,-value: Element,equation: {result == (reader,Traversal.Visited{context,value}) : Reader & Traversal.Visited<Context,Element>}}

def store(~Reader: Type,~Element: Data,~Context: Data,+depth: Nat,-reader: Reader,+position: U32,-output: Array<Element>,
  -result: Reader & Traversal.Visited<Context,Element>,output_evidence: Witness.Witness(Element,depth,output),sample: Sample<Reader,Element,Context,reader,result>) ->
  TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_store(~Reader,~Element,~Context,position,output,result)>:
  match sample:
    case Sample{context,value,equation}:
      %Equal.sym(Reader & Traversal.Visited<Context,Element>,result,(reader,Traversal.Visited{context,value}),equation) :
        TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_store(~Reader,~Element,~Context,position,output,_)>
      TraversalWitness.StateEvidence{Array.set(Element,output,position,value),(position + 1 : U32),context,{==},Witness.write(Element,depth,output,position,value,output_evidence)}

def step(~Reader: Type,~Element: Data,~Context: Data,~ReaderEvidence: @-reader: Reader -> Data,
  ~read: Reader -> Context -> Reader & Traversal.Visited<Context,Element>,
  ~read_preserves: @-reader: Reader -> @context: Context -> ReaderEvidence(reader) -> Sample<Reader,Element,Context,reader,read(reader,context)>,
  +depth: Nat,-reader: Reader,-state: Traversal.State<&1,&1,Reader,Array<Element>,Context>,reader_evidence: ReaderEvidence(reader),
  evidence: TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,state>) ->
  TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_step(~Reader,~Element,~Context,~read,state)>:
  match evidence:
    case TraversalWitness.StateEvidence{output,position,context,equation,output_evidence}:
      %Equal.sym(Traversal.State<&1,&1,Reader,Array<Element>,Context>,state,Traversal.State{reader,output,position,context},equation) :
        TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_step(~Reader,~Element,~Context,~read,_)>
      store(~Reader,~Element,~Context,depth,reader,position,output,read(reader,context),output_evidence,read_preserves(reader,context,reader_evidence))

def spread_store(~Element: Data,~Context: Data,+depth: Nat,-input: Array<Element>,+position: U32,-output: Array<Element>,target: Traversal.Target<Context>,
  -result: Array<Element> & Element,output_evidence: Witness.Witness(Element,depth,output),observed: Witness.Read<Element,input,result>) ->
  TraversalWitness.StateEvidence<Array<Element>,Element,Context,input,depth,Traversal.spread_store(~Element,~Context,position,output,target,result)>:
  match target observed:
    case Traversal.Target{context,+address} Witness.Read{value,equation}:
      %Equal.sym(Array<Element> & Element,result,(input,value),equation) :
        TraversalWitness.StateEvidence<Array<Element>,Element,Context,input,depth,Traversal.spread_store(~Element,~Context,position,output,Traversal.Target{context,address},_)>
      TraversalWitness.StateEvidence{Array.set(Element,output,address,value),(position + 1 : U32),context,{==},Witness.write(Element,depth,output,address,value,output_evidence)}

def spread_step(~Element: Data,~Context: Data,~target: Context -> Traversal.Target<Context>,
  +depth: Nat,-input: Array<Element>,-state: Traversal.State<&1,&1,Array<Element>,Array<Element>,Context>,input_evidence: Witness.ArrayWitness<Element,input>,
  evidence: TraversalWitness.StateEvidence<Array<Element>,Element,Context,input,depth,state>) ->
  TraversalWitness.StateEvidence<Array<Element>,Element,Context,input,depth,Traversal.spread_step(~Element,~Context,~target,state)>:
  match evidence:
    case TraversalWitness.StateEvidence{output,+position,context,equation,output_evidence}:
      %Equal.sym(Traversal.State<&1,&1,Array<Element>,Array<Element>,Context>,state,Traversal.State{input,output,position,context},equation) :
        TraversalWitness.StateEvidence<Array<Element>,Element,Context,input,depth,Traversal.spread_step(~Element,~Context,~target,_)>
      spread_store(~Element,~Context,depth,input,position,output,target(context),Array.get(Element,input,position),output_evidence,
        Witness.observed(Element,input,position,input_evidence))

def update_sample(~Reader: Type,~Element: Data,~Context: Data,~combine: Element -> Element -> Element,
  +depth: Nat,-reader: Reader,+position: U32,-output: Array<Element>,-current: Element,-result: Reader & Traversal.Visited<Context,Element>,
  output_evidence: Witness.Witness(Element,depth,output),sample: Sample<Reader,Element,Context,reader,result>) ->
  TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_update_sample(~Reader,~Element,~Context,~combine,position,output,current,result)>:
  match sample:
    case Sample{context,value,equation}:
      %Equal.sym(Reader & Traversal.Visited<Context,Element>,result,(reader,Traversal.Visited{context,value}),equation) :
        TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_update_sample(~Reader,~Element,~Context,~combine,position,output,current,_)>
      TraversalWitness.StateEvidence{Array.set(Element,output,position,combine(current,value)),(position + 1 : U32),context,{==},
        Witness.write(Element,depth,output,position,combine(current,value),output_evidence)}

def update_current(~Reader: Type,~Element: Data,~Context: Data,~read: Reader -> Context -> Reader & Traversal.Visited<Context,Element>,
  ~combine: Element -> Element -> Element,+depth: Nat,-reader: Reader,position: U32,context: Context,-output: Array<Element>,-result: Array<Element> & Element,
  output_evidence: Witness.Witness(Element,depth,output),observed: Witness.Read<Element,output,result>,sample: Sample<Reader,Element,Context,reader,read(reader,context)>) ->
  TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_update_current(~Reader,~Element,~Context,~read,~combine,position,context,reader,result)>:
  match observed:
    case Witness.Read{value,equation}:
      %Equal.sym(Array<Element> & Element,result,(output,value),equation) :
        TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_update_current(~Reader,~Element,~Context,~read,~combine,position,context,reader,_)>
      update_sample(~Reader,~Element,~Context,~combine,depth,reader,position,output,value,read(reader,context),output_evidence,sample)

def update_step(~Reader: Type,~Element: Data,~Context: Data,~ReaderEvidence: @-reader: Reader -> Data,
  ~read: Reader -> Context -> Reader & Traversal.Visited<Context,Element>,~combine: Element -> Element -> Element,
  ~read_preserves: @-reader: Reader -> @context: Context -> ReaderEvidence(reader) -> Sample<Reader,Element,Context,reader,read(reader,context)>,
  +depth: Nat,-reader: Reader,-state: Traversal.State<&1,&1,Reader,Array<Element>,Context>,reader_evidence: ReaderEvidence(reader),
  evidence: TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,state>) ->
  TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_update_step(~Reader,~Element,~Context,~read,~combine,state)>:
  match evidence:
    case TraversalWitness.StateEvidence{output,+position,+context,equation,+output_evidence}:
      %Equal.sym(Traversal.State<&1,&1,Reader,Array<Element>,Context>,state,Traversal.State{reader,output,position,context},equation) :
        TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_update_step(~Reader,~Element,~Context,~read,~combine,_)>
      update_current(~Reader,~Element,~Context,~read,~combine,depth,reader,position,context,output,Array.get(Element,output,position),output_evidence,
        Witness.read(Element,depth,output,position,output_evidence),read_preserves(reader,context,reader_evidence))

def update(~Reader: Type,~Element: Data,~Context: Data,~ReaderEvidence: @-reader: Reader -> Data,
  ~read: Reader -> Context -> Reader & Traversal.Visited<Context,Element>,~combine: Element -> Element -> Element,
  ~read_preserves: @-reader: Reader -> @context: Context -> ReaderEvidence(reader) -> Sample<Reader,Element,Context,reader,read(reader,context)>,
  +depth: Nat,count: Nat,-reader: Reader,-state: Traversal.State<&1,&1,Reader,Array<Element>,Context>,+reader_evidence: ReaderEvidence(reader),
  evidence: TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,state>) ->
  TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,Traversal.visit_update(~Reader,~Element,~Context,~read,~combine,count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      update(~Reader,~Element,~Context,~ReaderEvidence,~read,~combine,~read_preserves,depth,rest,reader,
        Traversal.visit_update_step(~Reader,~Element,~Context,~read,~combine,state),reader_evidence,
        update_step(~Reader,~Element,~Context,~ReaderEvidence,~read,~combine,~read_preserves,depth,reader,state,reader_evidence,evidence))

def fold_value(~Reader: Type,~Element: Data,~Accumulator: Data,~Context: Data,~combine: Accumulator -> Element -> Accumulator,
  -reader: Reader,+position: U32,-accumulator: Accumulator,-result: Reader & Traversal.Visited<Context,Element>,sample: Sample<Reader,Element,Context,reader,result>) ->
  TraversalWitness.FoldEvidence<Reader,Accumulator,Context,reader,Traversal.visit_fold_value(~Reader,~Element,~Accumulator,~Context,~combine,position,accumulator,result)>:
  match sample:
    case Sample{context,value,equation}:
      %Equal.sym(Reader & Traversal.Visited<Context,Element>,result,(reader,Traversal.Visited{context,value}),equation) :
        TraversalWitness.FoldEvidence<Reader,Accumulator,Context,reader,Traversal.visit_fold_value(~Reader,~Element,~Accumulator,~Context,~combine,position,accumulator,_)>
      TraversalWitness.FoldEvidence{combine(accumulator,value),(position + 1 : U32),context,{==}}

def fold_step(~Reader: Type,~Element: Data,~Accumulator: Data,~Context: Data,~ReaderEvidence: @-reader: Reader -> Data,
  ~read: Reader -> Context -> Reader & Traversal.Visited<Context,Element>,~combine: Accumulator -> Element -> Accumulator,
  ~read_preserves: @-reader: Reader -> @context: Context -> ReaderEvidence(reader) -> Sample<Reader,Element,Context,reader,read(reader,context)>,
  -reader: Reader,-state: Traversal.State<&1,&2,Reader,Accumulator,Context>,reader_evidence: ReaderEvidence(reader),
  evidence: TraversalWitness.FoldEvidence<Reader,Accumulator,Context,reader,state>) ->
  TraversalWitness.FoldEvidence<Reader,Accumulator,Context,reader,Traversal.visit_fold_step(~Reader,~Element,~Accumulator,~Context,~read,~combine,state)>:
  match evidence:
    case TraversalWitness.FoldEvidence{accumulator,position,context,equation}:
      %Equal.sym(Traversal.State<&1,&2,Reader,Accumulator,Context>,state,Traversal.State{reader,accumulator,position,context},equation) :
        TraversalWitness.FoldEvidence<Reader,Accumulator,Context,reader,Traversal.visit_fold_step(~Reader,~Element,~Accumulator,~Context,~read,~combine,_)>
      fold_value(~Reader,~Element,~Accumulator,~Context,~combine,reader,position,accumulator,read(reader,context),read_preserves(reader,context,reader_evidence))

def fold_run(~Reader: Type,~Element: Data,~Accumulator: Data,~Context: Data,~ReaderEvidence: @-reader: Reader -> Data,
  ~read: Reader -> Context -> Reader & Traversal.Visited<Context,Element>,~combine: Accumulator -> Element -> Accumulator,
  ~read_preserves: @-reader: Reader -> @context: Context -> ReaderEvidence(reader) -> Sample<Reader,Element,Context,reader,read(reader,context)>,
  count: Nat,-reader: Reader,-state: Traversal.State<&1,&2,Reader,Accumulator,Context>,+reader_evidence: ReaderEvidence(reader),
  evidence: TraversalWitness.FoldEvidence<Reader,Accumulator,Context,reader,state>) ->
  TraversalWitness.FoldEvidence<Reader,Accumulator,Context,reader,Traversal.visit_fold(~Reader,~Element,~Accumulator,~Context,~read,~combine,count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      fold_run(~Reader,~Element,~Accumulator,~Context,~ReaderEvidence,~read,~combine,~read_preserves,rest,reader,
        Traversal.visit_fold_step(~Reader,~Element,~Accumulator,~Context,~read,~combine,state),reader_evidence,
        fold_step(~Reader,~Element,~Accumulator,~Context,~ReaderEvidence,~read,~combine,~read_preserves,reader,state,reader_evidence,evidence))
