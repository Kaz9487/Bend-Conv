# Proof-only preservation of affine traversal owners. Reader values are handed
# back unchanged; output witnesses keep their original depth. Numerical loops
# call none of these functions: certificate equations use them under rewrites.
import Base
import ../traversal_loop.bend as Loop
import ../traversal_array.bend as Traversal
import ./storage_witness.bend as Witness

type Sample<-Reader: Type,-Element: Data,-reader: Reader,-result: Reader & Element> is Data:
  Sample{-value: Element,equation: {result == (reader,value) : Reader & Element}}

type StateEvidence<-Reader: Type,-Element: Data,-Context: Data,-reader: Reader,-depth: Nat,
  -state: Traversal.State<&1,&1,Reader,Array<Element>,Context>> is Data:
  StateEvidence{-output: Array<Element>,position: U32,context: Context,
    equation: {state == Traversal.State{reader,output,position,context} : Traversal.State<&1,&1,Reader,Array<Element>,Context>},
    output_evidence: Witness.Witness(Element,depth,output)}

def store(~Reader: Type,~Element: Data,~Context: Data,~destination: Context -> U32 -> U32,
  +depth: Nat,-reader: Reader,+position: U32,+context: Context,-output: Array<Element>,-result: Reader & Element,
  output_evidence: Witness.Witness(Element,depth,output),sample: Sample<Reader,Element,reader,result>) ->
  StateEvidence<Reader,Element,Context,reader,depth,Traversal.store(~Reader,~Element,~Context,~destination,position,context,output,result)>:
  match sample:
    case Sample{value,equation}:
      %Equal.sym(Reader & Element,result,(reader,value),equation) :
        StateEvidence<Reader,Element,Context,reader,depth,Traversal.store(~Reader,~Element,~Context,~destination,position,context,output,_)>
      StateEvidence{Array.set(Element,output,destination(context,position),value),(position + 1 : U32),context,{==},
        Witness.write(Element,depth,output,destination(context,position),value,output_evidence)}

def step(~Reader: Type,~Element: Data,~Context: Data,~ReaderEvidence: @-reader: Reader -> Data,
  ~read: Reader -> Context -> U32 -> Reader & Element,~destination: Context -> U32 -> U32,
  ~read_preserves: @-reader: Reader -> @context: Context -> @index: U32 -> ReaderEvidence(reader) ->
    Sample<Reader,Element,reader,read(reader,context,index)>,
  +depth: Nat,-reader: Reader,-state: Traversal.State<&1,&1,Reader,Array<Element>,Context>,
  reader_evidence: ReaderEvidence(reader),evidence: StateEvidence<Reader,Element,Context,reader,depth,state>) ->
  StateEvidence<Reader,Element,Context,reader,depth,Traversal.step(~Reader,~Element,~Context,~read,~destination,state)>:
  match evidence:
    case StateEvidence{output,+position,+context,equation,output_evidence}:
      %Equal.sym(Traversal.State<&1,&1,Reader,Array<Element>,Context>,state,Traversal.State{reader,output,position,context},equation) :
        StateEvidence<Reader,Element,Context,reader,depth,Traversal.step(~Reader,~Element,~Context,~read,~destination,_)>
      store(~Reader,~Element,~Context,~destination,depth,reader,position,context,output,read(reader,context,position),output_evidence,
        read_preserves(reader,context,position,reader_evidence))

def run(~Reader: Type,~Element: Data,~Context: Data,~ReaderEvidence: @-reader: Reader -> Data,
  ~read: Reader -> Context -> U32 -> Reader & Element,~destination: Context -> U32 -> U32,
  ~read_preserves: @-reader: Reader -> @context: Context -> @index: U32 -> ReaderEvidence(reader) ->
    Sample<Reader,Element,reader,read(reader,context,index)>,
  +depth: Nat,count: Nat,-reader: Reader,-state: Traversal.State<&1,&1,Reader,Array<Element>,Context>,
  +reader_evidence: ReaderEvidence(reader),evidence: StateEvidence<Reader,Element,Context,reader,depth,state>) ->
  StateEvidence<Reader,Element,Context,reader,depth,Traversal.run(~Reader,~Element,~Context,~read,~destination,count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      run(~Reader,~Element,~Context,~ReaderEvidence,~read,~destination,~read_preserves,depth,rest,reader,
        Traversal.step(~Reader,~Element,~Context,~read,~destination,state),reader_evidence,
        step(~Reader,~Element,~Context,~ReaderEvidence,~read,~destination,~read_preserves,depth,reader,state,reader_evidence,evidence))

type ResultEvidence<-Reader: Type,-Element: Data,-reader: Reader,-depth: Nat,-result: Reader & Array<Element>> is Data:
  ResultEvidence{-output: Array<Element>,equation: {result == (reader,output) : Reader & Array<Element>},
    output_evidence: Witness.Witness(Element,depth,output)}

def finish(~Reader: Type,~Element: Data,~Context: Data,-reader: Reader,-depth: Nat,
  -state: Traversal.State<&1,&1,Reader,Array<Element>,Context>,evidence: StateEvidence<Reader,Element,Context,reader,depth,state>) ->
  ResultEvidence<Reader,Element,reader,depth,Traversal.finish(~Reader,~Element,~Context,state)>:
  match evidence:
    case StateEvidence{output,position,context,equation,output_evidence}:
      %Equal.sym(Traversal.State<&1,&1,Reader,Array<Element>,Context>,state,Traversal.State{reader,output,position,context},equation) :
        ResultEvidence<Reader,Element,reader,depth,Traversal.finish(~Reader,~Element,~Context,_)>
      ResultEvidence{output,{==},output_evidence}

def array_evidence(~Element: Data,-storage: Array<Element>) -> Data:
  Witness.ArrayWitness<Element,storage>

def array_sample(~Element: Data,-storage: Array<Element>,-result: Array<Element> & Element,
  read: Witness.Read<Element,storage,result>) -> Sample<Array<Element>,Element,storage,result>:
  match read:
    case Witness.Read{value,equation}: Sample{value,equation}

def offset_read(~Element: Data,-storage: Array<Element>,context: Traversal.Offsets,index: U32,
  evidence: Witness.ArrayWitness<Element,storage>) ->
  Sample<Array<Element>,Element,storage,Traversal.source_offset(~Element,storage,context,index)>:
  Traversal.Offsets{source,destination} = context
  array_sample(~Element,storage,Array.get(Element,storage,(source + index : U32)),Witness.observed(Element,storage,(source + index : U32),evidence))

def first(~Left: Type,~Right: Type,pair: Left & Right) -> Left:
  (left,right) = pair
  left

def second(~Left: Type,~Right: Type,pair: Left & Right) -> Right:
  (left,right) = pair
  right

def source_unchanged(~Reader: Type,~Element: Data,-reader: Reader,-depth: Nat,-result: Reader & Array<Element>,
  evidence: ResultEvidence<Reader,Element,reader,depth,result>) -> {first(~Reader,~Array<Element>,result) == reader : Reader}:
  match evidence:
    case ResultEvidence{output,equation,output_evidence}:
      Equal.cong(Reader & Array<Element>,Reader,pair => first(~Reader,~Array<Element>,pair),result,(reader,output),equation)

def output_witness(~Reader: Type,~Element: Data,-reader: Reader,-depth: Nat,-result: Reader & Array<Element>,
  evidence: ResultEvidence<Reader,Element,reader,depth,result>) -> Witness.Witness(Element,depth,second(~Reader,~Array<Element>,result)):
  match evidence:
    case ResultEvidence{output,equation,output_evidence}:
      %Equal.sym(Reader & Array<Element>,result,(reader,output),equation) : Witness.Witness(Element,depth,second(~Reader,~Array<Element>,_))
      output_evidence

def update_sample(~Reader: Type,~Element: Data,~Value: Data,~Context: Data,
  ~combine: Element -> Value -> Element,~destination: Context -> U32 -> U32,
  +depth: Nat,-reader: Reader,+position: U32,+context: Context,-output: Array<Element>,-current: Element,
  -result: Reader & Value,output_evidence: Witness.Witness(Element,depth,output),
  sample: Sample<Reader,Value,reader,result>) ->
  StateEvidence<Reader,Element,Context,reader,depth,
    Traversal.update_sample(~Reader,~Element,~Value,~Context,~combine,~destination,position,context,output,current,result)>:
  match sample:
    case Sample{value,equation}:
      %Equal.sym(Reader & Value,result,(reader,value),equation) :
        StateEvidence<Reader,Element,Context,reader,depth,
          Traversal.update_sample(~Reader,~Element,~Value,~Context,~combine,~destination,position,context,output,current,_)>
      StateEvidence{Array.set(Element,output,destination(context,position),combine(current,value)),(position + 1 : U32),context,{==},
        Witness.write(Element,depth,output,destination(context,position),combine(current,value),output_evidence)}

def update_current(~Reader: Type,~Element: Data,~Value: Data,~Context: Data,
  ~read: Reader -> Context -> U32 -> Reader & Value,~combine: Element -> Value -> Element,
  ~destination: Context -> U32 -> U32,+depth: Nat,-reader: Reader,+position: U32,+context: Context,
  -output: Array<Element>,-result: Array<Element> & Element,
  output_evidence: Witness.Witness(Element,depth,output),observed: Witness.Read<Element,output,result>,
  sample: Sample<Reader,Value,reader,read(reader,context,position)>) ->
  StateEvidence<Reader,Element,Context,reader,depth,
    Traversal.update_current(~Reader,~Element,~Value,~Context,~read,~combine,~destination,reader,position,context,result)>:
  match observed:
    case Witness.Read{value,equation}:
      %Equal.sym(Array<Element> & Element,result,(output,value),equation) :
        StateEvidence<Reader,Element,Context,reader,depth,
          Traversal.update_current(~Reader,~Element,~Value,~Context,~read,~combine,~destination,reader,position,context,_)>
      update_sample(~Reader,~Element,~Value,~Context,~combine,~destination,depth,reader,position,context,output,value,
        read(reader,context,position),output_evidence,sample)

def update_step(~Reader: Type,~Element: Data,~Value: Data,~Context: Data,~ReaderEvidence: @-reader: Reader -> Data,
  ~read: Reader -> Context -> U32 -> Reader & Value,~combine: Element -> Value -> Element,~destination: Context -> U32 -> U32,
  ~read_preserves: @-reader: Reader -> @context: Context -> @index: U32 -> ReaderEvidence(reader) ->
    Sample<Reader,Value,reader,read(reader,context,index)>,
  +depth: Nat,-reader: Reader,-state: Traversal.State<&1,&1,Reader,Array<Element>,Context>,
  reader_evidence: ReaderEvidence(reader),evidence: StateEvidence<Reader,Element,Context,reader,depth,state>) ->
  StateEvidence<Reader,Element,Context,reader,depth,
    Traversal.update_step(~Reader,~Element,~Value,~Context,~read,~combine,~destination,state)>:
  match evidence:
    case StateEvidence{output,+position,+context,equation,+output_evidence}:
      %Equal.sym(Traversal.State<&1,&1,Reader,Array<Element>,Context>,state,Traversal.State{reader,output,position,context},equation) :
        StateEvidence<Reader,Element,Context,reader,depth,
          Traversal.update_step(~Reader,~Element,~Value,~Context,~read,~combine,~destination,_)>
      update_current(~Reader,~Element,~Value,~Context,~read,~combine,~destination,depth,reader,position,context,output,
        Array.get(Element,output,destination(context,position)),output_evidence,
        Witness.read(Element,depth,output,destination(context,position),output_evidence),
        read_preserves(reader,context,position,reader_evidence))

def update(~Reader: Type,~Element: Data,~Value: Data,~Context: Data,~ReaderEvidence: @-reader: Reader -> Data,
  ~read: Reader -> Context -> U32 -> Reader & Value,~combine: Element -> Value -> Element,~destination: Context -> U32 -> U32,
  ~read_preserves: @-reader: Reader -> @context: Context -> @index: U32 -> ReaderEvidence(reader) ->
    Sample<Reader,Value,reader,read(reader,context,index)>,
  +depth: Nat,count: Nat,-reader: Reader,-state: Traversal.State<&1,&1,Reader,Array<Element>,Context>,
  +reader_evidence: ReaderEvidence(reader),evidence: StateEvidence<Reader,Element,Context,reader,depth,state>) ->
  StateEvidence<Reader,Element,Context,reader,depth,
    Traversal.update(~Reader,~Element,~Value,~Context,~read,~combine,~destination,count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      update(~Reader,~Element,~Value,~Context,~ReaderEvidence,~read,~combine,~destination,~read_preserves,depth,rest,reader,
        Traversal.update_step(~Reader,~Element,~Value,~Context,~read,~combine,~destination,state),reader_evidence,
        update_step(~Reader,~Element,~Value,~Context,~ReaderEvidence,~read,~combine,~destination,~read_preserves,
          depth,reader,state,reader_evidence,evidence))

type MapEvidence<-Element: Data,-Context: Data,-depth: Nat,
  -state: Traversal.State<&2,&1,Unit,Array<Element>,Context>> is Data:
  MapEvidence{-output: Array<Element>,position: U32,context: Context,
    equation: {state == Traversal.State{Unit{},output,position,context} : Traversal.State<&2,&1,Unit,Array<Element>,Context>},
    output_evidence: Witness.Witness(Element,depth,output)}

def mapped_value(~Element: Data,~Context: Data,~map: Context -> Element -> Element,
  +depth: Nat,+position: U32,+context: Context,-output: Array<Element>,-result: Array<Element> & Element,
  output_evidence: Witness.Witness(Element,depth,output),observed: Witness.Read<Element,output,result>) ->
  MapEvidence<Element,Context,depth,Traversal.mapped_value(~Element,~Context,~map,position,context,result)>:
  match observed:
    case Witness.Read{value,equation}:
      %Equal.sym(Array<Element> & Element,result,(output,value),equation) :
        MapEvidence<Element,Context,depth,Traversal.mapped_value(~Element,~Context,~map,position,context,_)>
      MapEvidence{Array.set(Element,output,position,map(context,value)),(position + 1 : U32),context,{==},
        Witness.write(Element,depth,output,position,map(context,value),output_evidence)}

def map_step(~Element: Data,~Context: Data,~map: Context -> Element -> Element,+depth: Nat,
  -state: Traversal.State<&2,&1,Unit,Array<Element>,Context>,evidence: MapEvidence<Element,Context,depth,state>) ->
  MapEvidence<Element,Context,depth,Traversal.map_step(~Element,~Context,~map,state)>:
  match evidence:
    case MapEvidence{output,+position,+context,equation,+output_evidence}:
      %Equal.sym(Traversal.State<&2,&1,Unit,Array<Element>,Context>,state,Traversal.State{Unit{},output,position,context},equation) :
        MapEvidence<Element,Context,depth,Traversal.map_step(~Element,~Context,~map,_)>
      mapped_value(~Element,~Context,~map,depth,position,context,output,Array.get(Element,output,position),output_evidence,
        Witness.read(Element,depth,output,position,output_evidence))

def map_run(~Element: Data,~Context: Data,~map: Context -> Element -> Element,+depth: Nat,count: Nat,
  -state: Traversal.State<&2,&1,Unit,Array<Element>,Context>,evidence: MapEvidence<Element,Context,depth,state>) ->
  MapEvidence<Element,Context,depth,Loop.run(~Traversal.State<&2,&1,Unit,Array<Element>,Context>,~Traversal.map_step(~Element,~Context,~map),count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      map_run(~Element,~Context,~map,depth,rest,Traversal.map_step(~Element,~Context,~map,state),
        map_step(~Element,~Context,~map,depth,state,evidence))

def mapped_output(~Element: Data,~Context: Data,-depth: Nat,-state: Traversal.State<&2,&1,Unit,Array<Element>,Context>,
  evidence: MapEvidence<Element,Context,depth,state>) -> Witness.Witness(Element,depth,Traversal.mapped_output(~Element,~Context,state)):
  match evidence:
    case MapEvidence{output,position,context,equation,output_evidence}:
      %Equal.sym(Traversal.State<&2,&1,Unit,Array<Element>,Context>,state,Traversal.State{Unit{},output,position,context},equation) :
        Witness.Witness(Element,depth,Traversal.mapped_output(~Element,~Context,_))
      output_evidence

def map_in_place(~Element: Data,~Context: Data,~map: Context -> Element -> Element,+depth: Nat,count: Nat,
  position: U32,context: Context,-output: Array<Element>,evidence: Witness.Witness(Element,depth,output)) ->
  Witness.Witness(Element,depth,Traversal.map_in_place(~Element,~Context,~map,count,position,context,output)):
  mapped_output(~Element,~Context,depth,
    Loop.run(~Traversal.State<&2,&1,Unit,Array<Element>,Context>,~Traversal.map_step(~Element,~Context,~map),count,
      Traversal.State{Unit{},output,position,context}),
    map_run(~Element,~Context,~map,depth,count,Traversal.State{Unit{},output,position,context},MapEvidence{output,position,context,{==},evidence}))

type FoldEvidence<-Reader: Type,-Accumulator: Data,-Context: Data,-reader: Reader,
  -state: Traversal.State<&1,&2,Reader,Accumulator,Context>> is Data:
  FoldEvidence{-accumulator: Accumulator,position: U32,context: Context,
    equation: {state == Traversal.State{reader,accumulator,position,context} : Traversal.State<&1,&2,Reader,Accumulator,Context>}}

def fold_value(~Reader: Type,~Element: Data,~Accumulator: Data,~Context: Data,
  ~combine: Accumulator -> Element -> Accumulator,-reader: Reader,-accumulator: Accumulator,+position: U32,context: Context,
  -result: Reader & Element,sample: Sample<Reader,Element,reader,result>) ->
  FoldEvidence<Reader,Accumulator,Context,reader,
    Traversal.fold_value(~Reader,~Element,~Accumulator,~Context,~combine,accumulator,position,context,result)>:
  match sample:
    case Sample{value,equation}:
      %Equal.sym(Reader & Element,result,(reader,value),equation) :
        FoldEvidence<Reader,Accumulator,Context,reader,
          Traversal.fold_value(~Reader,~Element,~Accumulator,~Context,~combine,accumulator,position,context,_)>
      FoldEvidence{combine(accumulator,value),(position + 1 : U32),context,{==}}

def fold_step(~Reader: Type,~Element: Data,~Accumulator: Data,~Context: Data,~ReaderEvidence: @-reader: Reader -> Data,
  ~read: Reader -> Context -> U32 -> Reader & Element,~combine: Accumulator -> Element -> Accumulator,
  ~read_preserves: @-reader: Reader -> @context: Context -> @index: U32 -> ReaderEvidence(reader) ->
    Sample<Reader,Element,reader,read(reader,context,index)>,
  -reader: Reader,-state: Traversal.State<&1,&2,Reader,Accumulator,Context>,reader_evidence: ReaderEvidence(reader),
  evidence: FoldEvidence<Reader,Accumulator,Context,reader,state>) ->
  FoldEvidence<Reader,Accumulator,Context,reader,Traversal.fold_step(~Reader,~Element,~Accumulator,~Context,~read,~combine,state)>:
  match evidence:
    case FoldEvidence{accumulator,+position,+context,equation}:
      %Equal.sym(Traversal.State<&1,&2,Reader,Accumulator,Context>,state,Traversal.State{reader,accumulator,position,context},equation) :
        FoldEvidence<Reader,Accumulator,Context,reader,Traversal.fold_step(~Reader,~Element,~Accumulator,~Context,~read,~combine,_)>
      fold_value(~Reader,~Element,~Accumulator,~Context,~combine,reader,accumulator,position,context,read(reader,context,position),
        read_preserves(reader,context,position,reader_evidence))

def fold_run(~Reader: Type,~Element: Data,~Accumulator: Data,~Context: Data,~ReaderEvidence: @-reader: Reader -> Data,
  ~read: Reader -> Context -> U32 -> Reader & Element,~combine: Accumulator -> Element -> Accumulator,
  ~read_preserves: @-reader: Reader -> @context: Context -> @index: U32 -> ReaderEvidence(reader) ->
    Sample<Reader,Element,reader,read(reader,context,index)>,
  count: Nat,-reader: Reader,-state: Traversal.State<&1,&2,Reader,Accumulator,Context>,+reader_evidence: ReaderEvidence(reader),
  evidence: FoldEvidence<Reader,Accumulator,Context,reader,state>) ->
  FoldEvidence<Reader,Accumulator,Context,reader,
    Loop.run(~Traversal.State<&1,&2,Reader,Accumulator,Context>,~Traversal.fold_step(~Reader,~Element,~Accumulator,~Context,~read,~combine),count,state)>:
  match count:
    case 0n: evidence
    case 1n+rest:
      fold_run(~Reader,~Element,~Accumulator,~Context,~ReaderEvidence,~read,~combine,~read_preserves,rest,reader,
        Traversal.fold_step(~Reader,~Element,~Accumulator,~Context,~read,~combine,state),reader_evidence,
        fold_step(~Reader,~Element,~Accumulator,~Context,~ReaderEvidence,~read,~combine,~read_preserves,reader,state,reader_evidence,evidence))

def fold_finished(~Reader: Type,~Accumulator: Data,~Context: Data,-reader: Reader,
  -state: Traversal.State<&1,&2,Reader,Accumulator,Context>,evidence: FoldEvidence<Reader,Accumulator,Context,reader,state>) ->
  {first(~Reader,~Accumulator,Traversal.finish_fold(~Reader,~Accumulator,~Context,state)) == reader : Reader}:
  match evidence:
    case FoldEvidence{accumulator,position,context,equation}:
      %Equal.sym(Traversal.State<&1,&2,Reader,Accumulator,Context>,state,Traversal.State{reader,accumulator,position,context},equation) :
        {first(~Reader,~Accumulator,Traversal.finish_fold(~Reader,~Accumulator,~Context,_)) == reader : Reader}
      {==}

def fold(~Reader: Type,~Element: Data,~Accumulator: Data,~Context: Data,~ReaderEvidence: @-reader: Reader -> Data,
  ~read: Reader -> Context -> U32 -> Reader & Element,~combine: Accumulator -> Element -> Accumulator,
  ~read_preserves: @-reader: Reader -> @context: Context -> @index: U32 -> ReaderEvidence(reader) ->
    Sample<Reader,Element,reader,read(reader,context,index)>,
  count: Nat,-reader: Reader,position: U32,context: Context,-initial: Accumulator,reader_evidence: ReaderEvidence(reader)) ->
  {first(~Reader,~Accumulator,Traversal.fold(~Reader,~Element,~Accumulator,~Context,~read,~combine,count,position,reader,initial,context)) == reader : Reader}:
  fold_finished(~Reader,~Accumulator,~Context,reader,
    Loop.run(~Traversal.State<&1,&2,Reader,Accumulator,Context>,~Traversal.fold_step(~Reader,~Element,~Accumulator,~Context,~read,~combine),count,
      Traversal.State{reader,initial,position,context}),
    fold_run(~Reader,~Element,~Accumulator,~Context,~ReaderEvidence,~read,~combine,~read_preserves,count,reader,
      Traversal.State{reader,initial,position,context},reader_evidence,FoldEvidence{initial,position,context,{==}}))
