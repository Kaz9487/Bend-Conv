# One owned read/transform/write traversal. Reader and address mapping are static
# templates, so composition does not create a closure for each element.
import Base
import ./traversal_loop.bend as Loop

def write_list(~Element: Data,values: List<&2,Element>,+offset: U32,storage: Array<Element>) -> Array<Element>:
  match values:
    case []: storage
    case head <> tail:
      write_list(~Element,tail,(offset + 1 : U32),Array.set(Element,storage,offset,head))

type State<reader_kind,output_kind,-Reader: Kind(reader_kind),-Output: Kind(output_kind),-Context: Data> is Kind(reader_kind <&> output_kind):
  State{reader: Reader,output: Output,position: U32,context: Context}

# Indexed state updates share the traversal counter without requiring an array
# destination. The caller establishes any no-wrap bound on the U32 position.
def indexed_step(~Value: Type,~Context: Data,~update: U32 -> Context -> Value -> Value,
  state: State<&2,&1,Unit,Value,Context>) -> State<&2,&1,Unit,Value,Context>:
  State{Unit{},value,+position,+context} = state
  State{Unit{},update(position,context,value),(position + 1 : U32),context}

def indexed_output(~Value: Type,~Context: Data,state: State<&2,&1,Unit,Value,Context>) -> Value:
  State{Unit{},value,position,context} = state
  value

def indexed_fold(~Value: Type,~Context: Data,~update: U32 -> Context -> Value -> Value,
  count: Nat,position: U32,context: Context,value: Value) -> Value:
  indexed_output(~Value,~Context,
    Loop.run(~State<&2,&1,Unit,Value,Context>,~indexed_step(~Value,~Context,~update),count,
      State{Unit{},value,position,context}))

def write_state(~Reader: Type,~Element: Data,~Context: Data,reader: Reader,output: Array<Element>,position: U32,context: Context,
  destination: U32,value: Element) -> State<&1,&1,Reader,Array<Element>,Context>:
  State{reader,Array.set(Element,output,destination,value),(position + 1 : U32),context}

def store(~Reader: Type,~Element: Data,~Context: Data,~destination: Context -> U32 -> U32,
  +position: U32,+context: Context,output: Array<Element>,result: Reader & Element) -> State<&1,&1,Reader,Array<Element>,Context>:
  (reader,value) = result
  write_state(~Reader,~Element,~Context,reader,output,position,context,destination(context,position),value)

def step(~Reader: Type,~Element: Data,~Context: Data,
  ~read: Reader -> Context -> U32 -> Reader & Element,~destination: Context -> U32 -> U32,
  state: State<&1,&1,Reader,Array<Element>,Context>) -> State<&1,&1,Reader,Array<Element>,Context>:
  State{reader,output,+position,+context} = state
  store(~Reader,~Element,~Context,~destination,position,context,output,read(reader,context,position))

def run(~Reader: Type,~Element: Data,~Context: Data,
  ~read: Reader -> Context -> U32 -> Reader & Element,~destination: Context -> U32 -> U32,
  count: Nat,state: State<&1,&1,Reader,Array<Element>,Context>) -> State<&1,&1,Reader,Array<Element>,Context>:
  Loop.run(~State<&1,&1,Reader,Array<Element>,Context>,~step(~Reader,~Element,~Context,~read,~destination),count,state)

# Return both owners when the caller can reuse the source allocation later.
def finish(~Reader: Type,~Element: Data,~Context: Data,state: State<&1,&1,Reader,Array<Element>,Context>) -> Reader & Array<Element>:
  State{reader,output,position,context} = state
  (reader,output)

def index(context: Unit,position: U32) -> U32:
  position

def read_index(~Element: Data,storage: Array<Element>,context: Unit,position: U32) -> Array<Element> & Element:
  Array.get(Element,storage,position)

type Offsets is Data:
  Offsets{source: U32,destination: U32}

def destination_offset(context: Offsets,position: U32) -> U32:
  Offsets{source,destination} = context
  (destination + position : U32)

def source_offset(~Element: Data,input: Array<Element>,context: Offsets,position: U32) -> Array<Element> & Element:
  Offsets{source,destination} = context
  Array.get(Element,input,(source + position : U32))

def copy(~Element: Data,count: Nat,source: U32,destination: U32,input: Array<Element>,target: Array<Element>) -> Array<Element> & Array<Element>:
  finish(~Array<Element>,~Element,~Offsets,
    run(~Array<Element>,~Element,~Offsets,~source_offset(~Element),~destination_offset,count,
      State{input,target,0,Offsets{source,destination}}))

# Ordered folds share the same state/counter and preserve the combine order.
# Element and Accumulator can differ; no associativity assumption is required.
def fold_value(~Reader: Type,~Element: Data,~Accumulator: Data,~Context: Data,
  ~combine: Accumulator -> Element -> Accumulator,accumulator: Accumulator,
  position: U32,context: Context,result: Reader & Element) -> State<&1,&2,Reader,Accumulator,Context>:
  (reader,value) = result
  State{reader,combine(accumulator,value),(position + 1 : U32),context}

def fold_step(~Reader: Type,~Element: Data,~Accumulator: Data,~Context: Data,
  ~read: Reader -> Context -> U32 -> Reader & Element,~combine: Accumulator -> Element -> Accumulator,
  state: State<&1,&2,Reader,Accumulator,Context>) -> State<&1,&2,Reader,Accumulator,Context>:
  State{reader,accumulator,+position,+context} = state
  fold_value(~Reader,~Element,~Accumulator,~Context,~combine,accumulator,position,context,read(reader,context,position))

def finish_fold(~Reader: Type,~Accumulator: Data,~Context: Data,state: State<&1,&2,Reader,Accumulator,Context>) -> Reader & Accumulator:
  State{reader,accumulator,position,context} = state
  (reader,accumulator)

def fold(~Reader: Type,~Element: Data,~Accumulator: Data,~Context: Data,
  ~read: Reader -> Context -> U32 -> Reader & Element,~combine: Accumulator -> Element -> Accumulator,
  count: Nat,position: U32,reader: Reader,initial: Accumulator,context: Context) -> Reader & Accumulator:
  finish_fold(~Reader,~Accumulator,~Context,
    Loop.run(~State<&1,&2,Reader,Accumulator,Context>,~fold_step(~Reader,~Element,~Accumulator,~Context,~read,~combine),
      count,State{reader,initial,position,context}))

# Reuse the output owner for each read/write; no second array or clone.
def mapped_value(~Element: Data,~Context: Data,~map: Context -> Element -> Element,
  +position: U32,+context: Context,result: Array<Element> & Element) -> State<&2,&1,Unit,Array<Element>,Context>:
  (output,value) = result
  State{Unit{},Array.set(Element,output,position,map(context,value)),(position + 1 : U32),context}

def map_step(~Element: Data,~Context: Data,~map: Context -> Element -> Element,
  state: State<&2,&1,Unit,Array<Element>,Context>) -> State<&2,&1,Unit,Array<Element>,Context>:
  State{Unit{},output,+position,context} = state
  mapped_value(~Element,~Context,~map,position,context,Array.get(Element,output,position))

def mapped_output(~Element: Data,~Context: Data,state: State<&2,&1,Unit,Array<Element>,Context>) -> Array<Element>:
  State{Unit{},output,position,context} = state
  output

def map_in_place(~Element: Data,~Context: Data,~map: Context -> Element -> Element,
  count: Nat,position: U32,context: Context,output: Array<Element>) -> Array<Element>:
  mapped_output(~Element,~Context,Loop.run(~State<&2,&1,Unit,Array<Element>,Context>,
    ~map_step(~Element,~Context,~map),count,State{Unit{},output,position,context}))

# Read the destination's current element, combine it with an owned reader, then
# store through the same traversal state. Distinct Array owners cannot alias.
def update_sample(~Reader: Type,~Element: Data,~Sample: Data,~Context: Data,
  ~combine: Element -> Sample -> Element,~destination: Context -> U32 -> U32,
  position: U32,context: Context,output: Array<Element>,current: Element,result: Reader & Sample) -> State<&1,&1,Reader,Array<Element>,Context>:
  (reader,sample) = result
  store(~Reader,~Element,~Context,~destination,position,context,output,(reader,combine(current,sample)))

def update_current(~Reader: Type,~Element: Data,~Sample: Data,~Context: Data,
  ~read: Reader -> Context -> U32 -> Reader & Sample,~combine: Element -> Sample -> Element,
  ~destination: Context -> U32 -> U32,reader: Reader,+position: U32,+context: Context,result: Array<Element> & Element) -> State<&1,&1,Reader,Array<Element>,Context>:
  (output,current) = result
  update_sample(~Reader,~Element,~Sample,~Context,~combine,~destination,position,context,output,current,read(reader,context,position))

def update_step(~Reader: Type,~Element: Data,~Sample: Data,~Context: Data,
  ~read: Reader -> Context -> U32 -> Reader & Sample,~combine: Element -> Sample -> Element,
  ~destination: Context -> U32 -> U32,state: State<&1,&1,Reader,Array<Element>,Context>) -> State<&1,&1,Reader,Array<Element>,Context>:
  State{reader,output,+position,+context} = state
  update_current(~Reader,~Element,~Sample,~Context,~read,~combine,~destination,reader,position,context,
    Array.get(Element,output,destination(context,position)))

def update(~Reader: Type,~Element: Data,~Sample: Data,~Context: Data,
  ~read: Reader -> Context -> U32 -> Reader & Sample,~combine: Element -> Sample -> Element,
  ~destination: Context -> U32 -> U32,count: Nat,state: State<&1,&1,Reader,Array<Element>,Context>) -> State<&1,&1,Reader,Array<Element>,Context>:
  Loop.run(~State<&1,&1,Reader,Array<Element>,Context>,
    ~update_step(~Reader,~Element,~Sample,~Context,~read,~combine,~destination),count,state)

# A cursor is consumed by read and returned advanced. The state position is the
# absolute contiguous destination, so it need not retain a second cursor copy.
type Visited<-Context: Data,-Element: Data> is Data:
  Visited{context: Context,value: Element}

def visit_store(~Reader: Type,~Element: Data,~Context: Data,+position: U32,output: Array<Element>,result: Reader & Visited<Context,Element>) ->
  State<&1,&1,Reader,Array<Element>,Context>:
  (reader,Visited{context,value}) = result
  write_state(~Reader,~Element,~Context,reader,output,position,context,position,value)

def visit_step(~Reader: Type,~Element: Data,~Context: Data,~read: Reader -> Context -> Reader & Visited<Context,Element>,
  state: State<&1,&1,Reader,Array<Element>,Context>) -> State<&1,&1,Reader,Array<Element>,Context>:
  State{reader,output,position,context} = state
  visit_store(~Reader,~Element,~Context,position,output,read(reader,context))

def visit(~Reader: Type,~Element: Data,~Context: Data,~read: Reader -> Context -> Reader & Visited<Context,Element>,
  count: Nat,state: State<&1,&1,Reader,Array<Element>,Context>) -> State<&1,&1,Reader,Array<Element>,Context>:
  Loop.run(~State<&1,&1,Reader,Array<Element>,Context>,~visit_step(~Reader,~Element,~Context,~read),count,state)

# A scatter reads the source at the state position, which counts up, and
# stores at the address its context names; target yields that address with
# the context of the next element.
type Target<-Context: Data> is Data:
  Target{context: Context,address: U32}

def spread_store(~Element: Data,~Context: Data,position: U32,output: Array<Element>,target: Target<Context>,result: Array<Element> & Element) ->
  State<&1,&1,Array<Element>,Array<Element>,Context>:
  Target{context,address} = target
  (input,value) = result
  State{input,Array.set(Element,output,address,value),(position + 1 : U32),context}

def spread_step(~Element: Data,~Context: Data,~target: Context -> Target<Context>,
  state: State<&1,&1,Array<Element>,Array<Element>,Context>) -> State<&1,&1,Array<Element>,Array<Element>,Context>:
  State{input,output,+position,context} = state
  spread_store(~Element,~Context,position,output,target(context),Array.get(Element,input,position))

def visit_update_sample(~Reader: Type,~Element: Data,~Context: Data,~combine: Element -> Element -> Element,
  +position: U32,output: Array<Element>,current: Element,result: Reader & Visited<Context,Element>) -> State<&1,&1,Reader,Array<Element>,Context>:
  (reader,Visited{context,value}) = result
  write_state(~Reader,~Element,~Context,reader,output,position,context,position,combine(current,value))

def visit_update_current(~Reader: Type,~Element: Data,~Context: Data,~read: Reader -> Context -> Reader & Visited<Context,Element>,
  ~combine: Element -> Element -> Element,position: U32,context: Context,reader: Reader,result: Array<Element> & Element) -> State<&1,&1,Reader,Array<Element>,Context>:
  (output,current) = result
  visit_update_sample(~Reader,~Element,~Context,~combine,position,output,current,read(reader,context))

def visit_update_step(~Reader: Type,~Element: Data,~Context: Data,~read: Reader -> Context -> Reader & Visited<Context,Element>,
  ~combine: Element -> Element -> Element,state: State<&1,&1,Reader,Array<Element>,Context>) -> State<&1,&1,Reader,Array<Element>,Context>:
  State{reader,output,+position,context} = state
  visit_update_current(~Reader,~Element,~Context,~read,~combine,position,context,reader,Array.get(Element,output,position))

def visit_update(~Reader: Type,~Element: Data,~Context: Data,~read: Reader -> Context -> Reader & Visited<Context,Element>,
  ~combine: Element -> Element -> Element,count: Nat,state: State<&1,&1,Reader,Array<Element>,Context>) -> State<&1,&1,Reader,Array<Element>,Context>:
  Loop.run(~State<&1,&1,Reader,Array<Element>,Context>,~visit_update_step(~Reader,~Element,~Context,~read,~combine),count,state)

def visit_fold_value(~Reader: Type,~Element: Data,~Accumulator: Data,~Context: Data,~combine: Accumulator -> Element -> Accumulator,
  position: U32,accumulator: Accumulator,result: Reader & Visited<Context,Element>) -> State<&1,&2,Reader,Accumulator,Context>:
  (reader,Visited{context,value}) = result
  State{reader,combine(accumulator,value),(position + 1 : U32),context}

def visit_fold_step(~Reader: Type,~Element: Data,~Accumulator: Data,~Context: Data,~read: Reader -> Context -> Reader & Visited<Context,Element>,
  ~combine: Accumulator -> Element -> Accumulator,state: State<&1,&2,Reader,Accumulator,Context>) -> State<&1,&2,Reader,Accumulator,Context>:
  State{reader,accumulator,position,context} = state
  visit_fold_value(~Reader,~Element,~Accumulator,~Context,~combine,position,accumulator,read(reader,context))

def visit_fold(~Reader: Type,~Element: Data,~Accumulator: Data,~Context: Data,~read: Reader -> Context -> Reader & Visited<Context,Element>,
  ~combine: Accumulator -> Element -> Accumulator,count: Nat,state: State<&1,&2,Reader,Accumulator,Context>) -> State<&1,&2,Reader,Accumulator,Context>:
  Loop.run(~State<&1,&2,Reader,Accumulator,Context>,~visit_fold_step(~Reader,~Element,~Accumulator,~Context,~read,~combine),count,state)
