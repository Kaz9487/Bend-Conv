# Cloning preserves both buffer contents and logical lengths. The runtime
# certificates may differ; callers reason about contents, not certificate identity.
import Base
import ../storage_buffer.bend as Storage
import ../storage_clone.bend as Clone
import ./storage_clone_certificate.bend as CertificateClone
import ./storage_certificate.bend as Certificate
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./traversal_witness.bend as TraversalWitness

def contents(~Element: Data,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Buffer.Content<Element> & Buffer.Content<Element>:
  (left,right) = owners
  (Buffer.contents(~Element,left),Buffer.contents(~Element,right))

def array_contents(~Element: Data,length: Nat,owners: Array<Element> & Array<Element>) -> Buffer.Content<Element> & Buffer.Content<Element>:
  (left,right) = owners
  +length = length
  (Buffer.Content{Tree.reflect(Element,left),length},Buffer.Content{Tree.reflect(Element,right),length})

def buffers(~Element: Data,+length: Nat,owners: Array<Element> & Array<Element>,certificates: CertificateClone.Copies<Element,owners>) ->
  {contents(~Element,Clone.buffers(~Element,length,owners,certificates)) ==
    array_contents(~Element,length,owners) : Buffer.Content<Element> & Buffer.Content<Element>}:
  (left,right) = owners
  CertificateClone.Copies{left_certificate,right_certificate} = certificates
  {==}

def cloned_trees(~Element: Data,+tree: Tree.Tree<Element>,+length: Nat) ->
  {array_contents(~Element,length,Array.clone(Element,Tree.pack(Element,tree))) ==
    (Buffer.Content{tree,length},Buffer.Content{tree,length}) : Buffer.Content<Element> & Buffer.Content<Element>}:
  Equal.trans(Buffer.Content<Element> & Buffer.Content<Element>,
    array_contents(~Element,length,Array.clone(Element,Tree.pack(Element,tree))),
    array_contents(~Element,length,(Tree.pack(Element,tree),Tree.pack(Element,tree))),
    (Buffer.Content{tree,length},Buffer.Content{tree,length}),
    Equal.cong(Array<Element> & Array<Element>,Buffer.Content<Element> & Buffer.Content<Element>,
      owners => array_contents(~Element,length,owners),Array.clone(Element,Tree.pack(Element,tree)),(Tree.pack(Element,tree),Tree.pack(Element,tree)),StorageLaws.clone(Element,tree)),
    Equal.cong(Tree.Tree<Element>,Buffer.Content<Element> & Buffer.Content<Element>,
      tree => (Buffer.Content{tree,length},Buffer.Content{tree,length}),Tree.reflect(Element,Tree.pack(Element,tree)),tree,StorageLaws.reflect_packed(Element,tree)))

def packed(~Element: Data,+model: Buffer.Model<Element>) ->
  {contents(~Element,Clone.clone(~Element,Buffer.pack(~Element,model))) ==
    (Buffer.content(~Element,model),Buffer.content(~Element,model)) : Buffer.Content<Element> & Buffer.Content<Element>}:
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = model
  +certificate = {Certificate.Certificate{depth,equation} : Certificate.Certificate<Element,Tree.pack(Element,tree)>}
  Equal.trans(Buffer.Content<Element> & Buffer.Content<Element>,
    contents(~Element,Clone.clone(~Element,Buffer.pack(~Element,Buffer.Model{tree,length,certificate}))),
    array_contents(~Element,length,Array.clone(Element,Tree.pack(Element,tree))),
    (Buffer.Content{tree,length},Buffer.Content{tree,length}),
    buffers(~Element,length,Array.clone(Element,Tree.pack(Element,tree)),CertificateClone.clone(~Element,Tree.pack(Element,tree),certificate,certificate,certificate)),
    cloned_trees(~Element,tree,length))

def packed_contents(~Element: Data,+model: Buffer.Model<Element>) ->
  {contents(~Element,Clone.clone(~Element,Buffer.pack(~Element,model))) ==
    (Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.contents(~Element,Buffer.pack(~Element,model))) : Buffer.Content<Element> & Buffer.Content<Element>}:
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.content(~Element,model),Buffer.packed_contents(~Element,model)) :
    {contents(~Element,Clone.clone(~Element,Buffer.pack(~Element,model))) == (_,_) : Buffer.Content<Element> & Buffer.Content<Element>}
  packed(~Element,model)

def observed(~Element: Data,-input: Storage.Buffer<Element>,observation: Buffer.Observation<Element,input>) ->
  {contents(~Element,Clone.clone(~Element,input)) == (Buffer.contents(~Element,input),Buffer.contents(~Element,input)) : Buffer.Content<Element> & Buffer.Content<Element>}:
  Buffer.Observation{+model,equation} = observation
  %equation : {contents(~Element,Clone.clone(~Element,_)) == (Buffer.contents(~Element,_),Buffer.contents(~Element,_)) : Buffer.Content<Element> & Buffer.Content<Element>}
  packed_contents(~Element,model)

def clone(~Element: Data,input: Storage.Buffer<Element>) ->
  {contents(~Element,Clone.clone(~Element,input)) == (Buffer.contents(~Element,input),Buffer.contents(~Element,input)) : Buffer.Content<Element> & Buffer.Content<Element>}:
  observed(~Element,input,Buffer.observe(~Element,input))

# Observe a real affine pair once, then let induction use canonical models
# carrying the certificates actually returned by the runtime.
type Observation<-Element: Data,-owners: Storage.Buffer<Element> & Storage.Buffer<Element>> is Data:
  Observation{left: Buffer.Model<Element>,right: Buffer.Model<Element>,
    equation: {(Buffer.pack(~Element,left),Buffer.pack(~Element,right)) == owners : Storage.Buffer<Element> & Storage.Buffer<Element>}}

def pair_observed(~Element: Data,-left: Storage.Buffer<Element>,-right: Storage.Buffer<Element>,
  left_observation: Buffer.Observation<Element,left>,right_observation: Buffer.Observation<Element,right>) -> Observation<Element,(left,right)>:
  Buffer.Observation{+left_model,left_equal} = left_observation
  Buffer.Observation{+right_model,right_equal} = right_observation
  Observation{left_model,right_model,
    %left_equal : {(Buffer.pack(~Element,left_model),Buffer.pack(~Element,right_model)) == (_,right) : Storage.Buffer<Element> & Storage.Buffer<Element>}
    %right_equal : {(Buffer.pack(~Element,left_model),Buffer.pack(~Element,right_model)) == (Buffer.pack(~Element,left_model),_) : Storage.Buffer<Element> & Storage.Buffer<Element>}
    {==}}

def observe_pair(~Element: Data,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Observation<Element,owners>:
  (left,right) = owners
  pair_observed(~Element,left,right,Buffer.observe(~Element,left),Buffer.observe(~Element,right))

def lift_pair(~Element: Data,~Proposition: Storage.Buffer<Element> & Storage.Buffer<Element> -> Type,
  owners: Storage.Buffer<Element> & Storage.Buffer<Element>,observation: Observation<Element,owners>,
  canonical: @+left: Buffer.Model<Element> -> @+right: Buffer.Model<Element> ->
    @equation: {(Buffer.pack(~Element,left),Buffer.pack(~Element,right)) == owners : Storage.Buffer<Element> & Storage.Buffer<Element>} ->
    Proposition((Buffer.pack(~Element,left),Buffer.pack(~Element,right)))) -> Proposition(owners):
  Observation{+left,+right,+equation} = observation
  %equation : Proposition(_)
  canonical(left,right,equation)

def packed_pair_contents(~Element: Data,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>) ->
  {contents(~Element,(Buffer.pack(~Element,left),Buffer.pack(~Element,right))) ==
    (Buffer.content(~Element,left),Buffer.content(~Element,right)) : Buffer.Content<Element> & Buffer.Content<Element>}:
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,left)),Buffer.content(~Element,left),Buffer.packed_contents(~Element,left)) :
    {(_,Buffer.contents(~Element,Buffer.pack(~Element,right))) == (Buffer.content(~Element,left),Buffer.content(~Element,right)) : Buffer.Content<Element> & Buffer.Content<Element>}
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,right)),Buffer.content(~Element,right),Buffer.packed_contents(~Element,right)) :
    {(Buffer.content(~Element,left),_) == (Buffer.content(~Element,left),Buffer.content(~Element,right)) : Buffer.Content<Element> & Buffer.Content<Element>}
  {==}

def pair_contents(~Element: Data,+source: Buffer.Model<Element>,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>,
  equation: {(Buffer.pack(~Element,left),Buffer.pack(~Element,right)) == Clone.clone(~Element,Buffer.pack(~Element,source)) : Storage.Buffer<Element> & Storage.Buffer<Element>}) ->
  {(Buffer.content(~Element,left),Buffer.content(~Element,right)) ==
    (Buffer.content(~Element,source),Buffer.content(~Element,source)) : Buffer.Content<Element> & Buffer.Content<Element>}:
  Equal.trans(Buffer.Content<Element> & Buffer.Content<Element>,
    (Buffer.content(~Element,left),Buffer.content(~Element,right)),contents(~Element,(Buffer.pack(~Element,left),Buffer.pack(~Element,right))),
    (Buffer.content(~Element,source),Buffer.content(~Element,source)),
    Equal.sym(Buffer.Content<Element> & Buffer.Content<Element>,contents(~Element,(Buffer.pack(~Element,left),Buffer.pack(~Element,right))),
      (Buffer.content(~Element,left),Buffer.content(~Element,right)),packed_pair_contents(~Element,left,right)),
    Equal.trans(Buffer.Content<Element> & Buffer.Content<Element>,contents(~Element,(Buffer.pack(~Element,left),Buffer.pack(~Element,right))),
      contents(~Element,Clone.clone(~Element,Buffer.pack(~Element,source))),(Buffer.content(~Element,source),Buffer.content(~Element,source)),
      Equal.cong(Storage.Buffer<Element> & Storage.Buffer<Element>,Buffer.Content<Element> & Buffer.Content<Element>,
        owners => contents(~Element,owners),(Buffer.pack(~Element,left),Buffer.pack(~Element,right)),Clone.clone(~Element,Buffer.pack(~Element,source)),equation),packed(~Element,source)))

def left_contents(~Element: Data,+source: Buffer.Model<Element>,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>,
  equation: {(Buffer.pack(~Element,left),Buffer.pack(~Element,right)) == Clone.clone(~Element,Buffer.pack(~Element,source)) : Storage.Buffer<Element> & Storage.Buffer<Element>}) ->
  {Buffer.content(~Element,left) == Buffer.content(~Element,source) : Buffer.Content<Element>}:
  Equal.cong(Buffer.Content<Element> & Buffer.Content<Element>,Buffer.Content<Element>,
    owners => TraversalWitness.first(~Buffer.Content<Element>,~Buffer.Content<Element>,owners),
    (Buffer.content(~Element,left),Buffer.content(~Element,right)),(Buffer.content(~Element,source),Buffer.content(~Element,source)),pair_contents(~Element,source,left,right,equation))

def right_contents(~Element: Data,+source: Buffer.Model<Element>,+left: Buffer.Model<Element>,+right: Buffer.Model<Element>,
  equation: {(Buffer.pack(~Element,left),Buffer.pack(~Element,right)) == Clone.clone(~Element,Buffer.pack(~Element,source)) : Storage.Buffer<Element> & Storage.Buffer<Element>}) ->
  {Buffer.content(~Element,right) == Buffer.content(~Element,source) : Buffer.Content<Element>}:
  Equal.cong(Buffer.Content<Element> & Buffer.Content<Element>,Buffer.Content<Element>,
    owners => TraversalWitness.second(~Buffer.Content<Element>,~Buffer.Content<Element>,owners),
    (Buffer.content(~Element,left),Buffer.content(~Element,right)),(Buffer.content(~Element,source),Buffer.content(~Element,source)),pair_contents(~Element,source,left,right,equation))
