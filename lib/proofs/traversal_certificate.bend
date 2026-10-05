# Flat certificates are updated once after a complete traversal. The witness
# traversal is evaluated only by the checker, beneath erased equality rewrites.
import Base
import ../traversal_array.bend as Traversal
import ./storage_tree_model.bend as Tree
import ./storage_certificate.bend as Certificate
import ./traversal_witness.bend as TraversalWitness

type Certificates<-Reader: Type,-Element: Data,-ReaderCertificate: @-reader: Reader -> Data,
  -result: Reader & Array<Element>> is Data:
  Certificates{source: ReaderCertificate(TraversalWitness.first(~Reader,~Array<Element>,result)),
    output: Certificate.Certificate<Element,TraversalWitness.second(~Reader,~Array<Element>,result)>}

def completed(~Reader: Type,~Element: Data,~Context: Data,~ReaderEvidence: @-reader: Reader -> Data,
  ~run: Nat -> Traversal.State<&1,&1,Reader,Array<Element>,Context> -> Traversal.State<&1,&1,Reader,Array<Element>,Context>,
  ~preserves: @+depth: Nat -> @count: Nat -> @-reader: Reader -> @-state: Traversal.State<&1,&1,Reader,Array<Element>,Context> ->
    @+reader_evidence: ReaderEvidence(reader) -> TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,state> ->
    TraversalWitness.StateEvidence<Reader,Element,Context,reader,depth,run(count,state)>,
  count: Nat,position: U32,context: Context,-reader: Reader,-output: Array<Element>,reader_evidence: ReaderEvidence(reader),+depth: Nat,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  TraversalWitness.ResultEvidence<Reader,Element,reader,depth,
    Traversal.finish(~Reader,~Element,~Context,run(count,Traversal.State{reader,output,position,context}))>:
  TraversalWitness.finish(~Reader,~Element,~Context,reader,depth,run(count,Traversal.State{reader,output,position,context}),
    preserves(depth,count,reader,Traversal.State{reader,output,position,context},reader_evidence,
      TraversalWitness.StateEvidence{output,position,context,{==},Certificate.witness_at(Element,depth,output,equation)}))

def map_equation(~Element: Data,~Context: Data,~map: Context -> Element -> Element,count: Nat,position: U32,context: Context,
  -output: Array<Element>,+depth: Nat,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  {Traversal.map_in_place(~Element,~Context,~map,count,position,context,output) ==
    Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,Traversal.map_in_place(~Element,~Context,~map,count,position,context,output)))) : Array<Element>}:
  %Certificate.restored(Element,depth,Traversal.map_in_place(~Element,~Context,~map,count,position,context,output),
    TraversalWitness.map_in_place(~Element,~Context,~map,depth,count,position,context,output,Certificate.witness_at(Element,depth,output,equation))) :
    {_ == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,Traversal.map_in_place(~Element,~Context,~map,count,position,context,output)))) : Array<Element>}
  {==}

def map_in_place(~Element: Data,~Context: Data,~map: Context -> Element -> Element,count: Nat,position: U32,context: Context,
  -output: Array<Element>,certificate: Certificate.Certificate<Element,output>) ->
  Certificate.Certificate<Element,Traversal.map_in_place(~Element,~Context,~map,count,position,context,output)>:
  Certificate.Certificate{+depth,equation} = certificate
  Certificate.Certificate{depth,map_equation(~Element,~Context,~map,count,position,context,output,depth,equation)}

def fold(~Reader: Type,~Element: Data,~Accumulator: Data,~Context: Data,~ReaderCertificate: @-reader: Reader -> Data,
  ~ReaderEvidence: @-reader: Reader -> Data,~witness: @-reader: Reader -> ReaderCertificate(reader) -> ReaderEvidence(reader),
  ~read: Reader -> Context -> U32 -> Reader & Element,~combine: Accumulator -> Element -> Accumulator,
  ~read_preserves: @-reader: Reader -> @context: Context -> @index: U32 -> ReaderEvidence(reader) ->
    TraversalWitness.Sample<Reader,Element,reader,read(reader,context,index)>,
  count: Nat,position: U32,context: Context,-reader: Reader,-initial: Accumulator,+certificate: ReaderCertificate(reader)) ->
  ReaderCertificate(TraversalWitness.first(~Reader,~Accumulator,Traversal.fold(~Reader,~Element,~Accumulator,~Context,~read,~combine,count,position,reader,initial,context))):
  %Equal.sym(Reader,TraversalWitness.first(~Reader,~Accumulator,Traversal.fold(~Reader,~Element,~Accumulator,~Context,~read,~combine,count,position,reader,initial,context)),reader,
    TraversalWitness.fold(~Reader,~Element,~Accumulator,~Context,~ReaderEvidence,~read,~combine,~read_preserves,
      count,reader,position,context,initial,witness(reader,certificate))) : ReaderCertificate(_)
  certificate

# Metadata preparation belongs inside the erased preservation proof. Passing a
# prepared cursor to a live certificate helper would retain or rebuild it.
def operation_source(~Reader: Type,~Element: Data,~Metadata: Data,~ReaderCertificate: @-reader: Reader -> Data,
  ~operation: Metadata -> Reader -> Array<Element> -> Reader & Array<Element>,
  ~preserves: @+depth: Nat -> @metadata: Metadata -> @-reader: Reader -> @-output: Array<Element> -> ReaderCertificate(reader) ->
    {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>} ->
    TraversalWitness.ResultEvidence<Reader,Element,reader,depth,operation(metadata,reader,output)>,
  metadata: Metadata,-reader: Reader,-output: Array<Element>,proof_certificate: ReaderCertificate(reader),reader_certificate: ReaderCertificate(reader),depth: Nat,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  ReaderCertificate(TraversalWitness.first(~Reader,~Array<Element>,operation(metadata,reader,output))):
  %Equal.sym(Reader,TraversalWitness.first(~Reader,~Array<Element>,operation(metadata,reader,output)),reader,
    TraversalWitness.source_unchanged(~Reader,~Element,reader,depth,operation(metadata,reader,output),
      preserves(depth,metadata,reader,output,proof_certificate,equation))) : ReaderCertificate(_)
  reader_certificate

def operation_equation(~Reader: Type,~Element: Data,~Metadata: Data,~ReaderCertificate: @-reader: Reader -> Data,
  ~operation: Metadata -> Reader -> Array<Element> -> Reader & Array<Element>,
  ~preserves: @+depth: Nat -> @metadata: Metadata -> @-reader: Reader -> @-output: Array<Element> -> ReaderCertificate(reader) ->
    {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>} ->
    TraversalWitness.ResultEvidence<Reader,Element,reader,depth,operation(metadata,reader,output)>,
  metadata: Metadata,-reader: Reader,-output: Array<Element>,reader_certificate: ReaderCertificate(reader),+depth: Nat,
  equation: {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>}) ->
  {TraversalWitness.second(~Reader,~Array<Element>,operation(metadata,reader,output)) ==
    Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,TraversalWitness.second(~Reader,~Array<Element>,operation(metadata,reader,output))))) : Array<Element>}:
  %Certificate.restored(Element,depth,TraversalWitness.second(~Reader,~Array<Element>,operation(metadata,reader,output)),
    TraversalWitness.output_witness(~Reader,~Element,reader,depth,operation(metadata,reader,output),
      preserves(depth,metadata,reader,output,reader_certificate,equation))) :
    {_ == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,TraversalWitness.second(~Reader,~Array<Element>,operation(metadata,reader,output))))) : Array<Element>}
  {==}

def operation(~Reader: Type,~Element: Data,~Metadata: Data,~ReaderCertificate: @-reader: Reader -> Data,
  ~operation: Metadata -> Reader -> Array<Element> -> Reader & Array<Element>,
  ~preserves: @+depth: Nat -> @metadata: Metadata -> @-reader: Reader -> @-output: Array<Element> -> ReaderCertificate(reader) ->
    {output == Tree.pack(Element,Certificate.normalized(Element,depth,Tree.reflect(Element,output))) : Array<Element>} ->
    TraversalWitness.ResultEvidence<Reader,Element,reader,depth,operation(metadata,reader,output)>,
  +metadata: Metadata,-reader: Reader,-output: Array<Element>,source_certificate: ReaderCertificate(reader),equation_certificate: ReaderCertificate(reader),
  reader_certificate: ReaderCertificate(reader),output_certificate: Certificate.Certificate<Element,output>) ->
  Certificates<Reader,Element,ReaderCertificate,operation(metadata,reader,output)>:
  Certificate.Certificate{+depth,+equation} = output_certificate
  Certificates{operation_source(~Reader,~Element,~Metadata,~ReaderCertificate,~operation,~preserves,metadata,reader,output,source_certificate,reader_certificate,depth,equation),
    Certificate.Certificate{depth,operation_equation(~Reader,~Element,~Metadata,~ReaderCertificate,~operation,~preserves,metadata,reader,output,equation_certificate,depth,equation)}}

def fold_operation(~Reader: Type,~Accumulator: Data,~Metadata: Data,~ReaderCertificate: @-reader: Reader -> Data,
  ~operation: Metadata -> Reader -> Accumulator -> Reader & Accumulator,
  ~preserves: @metadata: Metadata -> @-reader: Reader -> @-initial: Accumulator -> ReaderCertificate(reader) ->
    {TraversalWitness.first(~Reader,~Accumulator,operation(metadata,reader,initial)) == reader : Reader},
  metadata: Metadata,-reader: Reader,-initial: Accumulator,proof_certificate: ReaderCertificate(reader),certificate: ReaderCertificate(reader)) ->
  ReaderCertificate(TraversalWitness.first(~Reader,~Accumulator,operation(metadata,reader,initial))):
  %Equal.sym(Reader,TraversalWitness.first(~Reader,~Accumulator,operation(metadata,reader,initial)),reader,
    preserves(metadata,reader,initial,proof_certificate)) : ReaderCertificate(_)
  certificate
