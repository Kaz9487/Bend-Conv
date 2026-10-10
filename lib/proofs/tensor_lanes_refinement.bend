# Shared gather/scatter facts for the owned binary and ternary lanes.
import Base
import ../traversal_cells.bend as Cells
import ./traversal_cells_refinement.bend as CellLaws
import ./traversal_batch_refinement.bend as Batch
import ./traversal_map_readback.bend as MapReadback
import ./storage_shape.bend as Shape
import ../tensor_lanes.bend as Lanes
import ../traversal_lanes.bend as Unary
import ../storage_buffer.bend as Storage
import ../kernel_tile.bend as Tile
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./storage_certificate.bend as Certificate
import ./traversal_lanes_refinement.bend as UnaryLaws
import ./traversal_lanes_certificate.bend as LaneCertificate
import ./traversal_witness.bend as Witness
import ../tensor_operations.bend as Operations
import ../tensor_access.bend as Access
import ../tensor_view.bend as Views
import ./tensor_access_refinement.bend as AccessLaws
import ./tensor_access_certificate.bend as AccessCertificate
import ./traversal_certificate.bend as TraversalCertificate
import ./traversal_array_refinement.bend as ArrayLaws
import ./storage_clone_refinement.bend as CloneLaws
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./loop_projection.bend as Projection
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_division.bend as Division
import ./word_encoding.bend as WordEncoding
import ./tensor_cursor_refinement.bend as CursorLaws
import ./tensor_visit_refinement.bend as TensorVisit
import ./traversal_visit_refinement.bend as Visit
import ./tensor_cursor_model.bend as CursorModel
import ./boolean_logic.bend as Boolean
import ./tensor_shape_refinement.bend as ShapeLaws
import ./nat_order.bend as Order
import ./word_power_of_two.bend as Power
import ./tensor_operation_bounds.bend as OperationBounds
import ./tensor_contiguous_bounds.bend as ContiguousBounds
import ./u32_full_range.bend as FullRange
import ../tensor_shape.bend as Shapes

# The scalar kernel's existing traversal model, projected through Buffer.
def raw_pair_contents(left_length: Nat,right_length: Nat,result: Array<F32> & Array<F32>) -> Buffer.Content<F32> & Buffer.Content<F32>:
  (right,left) = result
  (Buffer.Content{Tree.reflect(F32,left),left_length},Buffer.Content{Tree.reflect(F32,right),right_length})

def combined_contents(+left_length: Nat,+right_length: Nat,result: Array<F32> & Array<F32>,
  certificates: TraversalCertificate.Certificates<Array<F32>,F32,Certificate.family(~F32),result>) ->
  {CloneLaws.contents(~F32,Operations.combine_buffers(~F32,left_length,right_length,result,certificates)) ==
    raw_pair_contents(left_length,right_length,result) : Buffer.Content<F32> & Buffer.Content<F32>}:
  (right,left) = result
  TraversalCertificate.Certificates{right_certificate,left_certificate} = certificates
  {==}

def pair_model_contents(left_length: Nat,right_length: Nat,result: AccessLaws.ModelResult<Tree.Tree<F32>,F32>) -> Buffer.Content<F32> & Buffer.Content<F32>:
  AccessLaws.ModelResult{right,left} = result
  (Buffer.Content{left,left_length},Buffer.Content{right,right_length})

def packed_pair_contents(+left_length: Nat,+right_length: Nat,+result: AccessLaws.ModelResult<Tree.Tree<F32>,F32>) ->
  {raw_pair_contents(left_length,right_length,AccessLaws.pack_result(~Array<F32>,~Tree.Tree<F32>,~F32,~ArrayLaws.pack_tree(~F32),result)) ==
    pair_model_contents(left_length,right_length,result) : Buffer.Content<F32> & Buffer.Content<F32>}:
  AccessLaws.ModelResult{+right,+left} = result
  %Equal.sym(Tree.Tree<F32>,Tree.reflect(F32,Tree.pack(F32,left)),left,StorageLaws.reflect_packed(F32,left)) :
    {(Buffer.Content{_,left_length},Buffer.Content{Tree.reflect(F32,Tree.pack(F32,right)),right_length}) ==
      (Buffer.Content{left,left_length},Buffer.Content{right,right_length}) : Buffer.Content<F32> & Buffer.Content<F32>}
  Equal.cong(Tree.Tree<F32>,Buffer.Content<F32> & Buffer.Content<F32>,tree => (Buffer.Content{left,left_length},Buffer.Content{tree,right_length}),
    Tree.reflect(F32,Tree.pack(F32,right)),right,StorageLaws.reflect_packed(F32,right))

def scalar_core(~operation: F32 -> F32 -> F32,+count: Nat,+source: Views.View,+position: U32,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>) ->
  {CloneLaws.contents(~F32,Operations.combine_kernel(~F32,~operation,count,source,position,Buffer.pack(~F32,left),Buffer.pack(~F32,right))) ==
    pair_model_contents(Buffer.length(~F32,left),Buffer.length(~F32,right),AccessLaws.combine_model(~F32,~operation,
      Access.zero_contiguous(source) && U32.is_zero(position),count,source,position,Buffer.tree(~F32,right),Buffer.tree(~F32,left))) : Buffer.Content<F32> & Buffer.Content<F32>}:
  Buffer.Model{+a,+al,+ac} = left
  Buffer.Model{+b,+bl,+bc} = right
  +fast = Access.zero_contiguous(source) && U32.is_zero(position)
  +model = AccessLaws.combine_model(~F32,~operation,fast,count,source,position,b,a)
  Equal.trans(Buffer.Content<F32> & Buffer.Content<F32>,
    CloneLaws.contents(~F32,Operations.combine_kernel(~F32,~operation,count,source,position,Storage.Buffer{Tree.pack(F32,a),al,ac},Storage.Buffer{Tree.pack(F32,b),bl,bc})),
    raw_pair_contents(al,bl,Access.combine_run(~F32,~operation,fast,count,source,position,Tree.pack(F32,b),Tree.pack(F32,a))),pair_model_contents(al,bl,model),
    combined_contents(al,bl,Access.combine_run(~F32,~operation,fast,count,source,position,Tree.pack(F32,b),Tree.pack(F32,a)),
      AccessCertificate.combine(~F32,~operation,fast,count,source,position,Tree.pack(F32,b),Tree.pack(F32,a),bc,ac)),
    Equal.trans(Buffer.Content<F32> & Buffer.Content<F32>,
      raw_pair_contents(al,bl,Access.combine_run(~F32,~operation,fast,count,source,position,Tree.pack(F32,b),Tree.pack(F32,a))),
      raw_pair_contents(al,bl,AccessLaws.pack_result(~Array<F32>,~Tree.Tree<F32>,~F32,~ArrayLaws.pack_tree(~F32),model)),pair_model_contents(al,bl,model),
      Equal.cong(Array<F32> & Array<F32>,Buffer.Content<F32> & Buffer.Content<F32>,result => raw_pair_contents(al,bl,result),
        Access.combine_run(~F32,~operation,fast,count,source,position,Tree.pack(F32,b),Tree.pack(F32,a)),
        AccessLaws.pack_result(~Array<F32>,~Tree.Tree<F32>,~F32,~ArrayLaws.pack_tree(~F32),model),AccessLaws.combine(~F32,~operation,fast,count,source,position,b,a)),
      packed_pair_contents(al,bl,model)))

def observed_read(result: Storage.Buffer<F32> & Tile.Tile8) -> Buffer.Content<F32> & Tile.Tile8:
  (buffer,tile) = result
  (Buffer.contents(~F32,buffer),tile)

def raw_read(length: Nat,result: Array<F32> & Tile.Tile8) -> Buffer.Content<F32> & Tile.Tile8:
  (values,tile) = result
  (Buffer.Content{Tree.reflect(F32,values),length},tile)

def read_contents(+length: Nat,result: Array<F32> & Tile.Tile8,
  certificate: Certificate.Certificate<F32,Witness.first(~Array<F32>,~Tile.Tile8,result)>) ->
  {observed_read(Lanes.read_result(length,result,certificate)) ==
    raw_read(length,result) : Buffer.Content<F32> & Tile.Tile8}:
  (values,tile) = result
  {==}


def read(+model: Buffer.Model<F32>,+position: U32) ->
  {observed_read(Lanes.read(Buffer.pack(~F32,model),position)) ==
    (Buffer.content(~F32,model),UnaryLaws.cell(~Unit,~Lanes.identity,Buffer.tree(~F32,model),position,Unit{})) : Buffer.Content<F32> & Tile.Tile8}:
  Buffer.Model{+tree,+length,+certificate} = model
  +tile = UnaryLaws.cell(~Unit,~Lanes.identity,tree,position,Unit{})
  Equal.trans(Buffer.Content<F32> & Tile.Tile8,
    observed_read(Lanes.read(Storage.Buffer{Tree.pack(F32,tree),length,certificate},position)),
    raw_read(length,Unary.gather(~Unit,~Lanes.identity,Tree.pack(F32,tree),position,Unit{})),(Buffer.Content{tree,length},tile),
    read_contents(length,Unary.gather(~Unit,~Lanes.identity,Tree.pack(F32,tree),position,Unit{}),
      LaneCertificate.gather(~Unit,~Lanes.identity,Tree.pack(F32,tree),position,Unit{},certificate)),
    %Equal.sym(Array<F32> & Tile.Tile8,Unary.gather(~Unit,~Lanes.identity,Tree.pack(F32,tree),position,Unit{}),(Tree.pack(F32,tree),tile),UnaryLaws.gathered(~Unit,~Lanes.identity,tree,position,Unit{})) :
      {raw_read(length,_) == (Buffer.Content{tree,length},tile) : Buffer.Content<F32> & Tile.Tile8}
    %Equal.sym(Tree.Tree<F32>,Tree.reflect(F32,Tree.pack(F32,tree)),tree,StorageLaws.reflect_packed(F32,tree)) :
      {(Buffer.Content{_,length},tile) == (Buffer.Content{tree,length},tile) : Buffer.Content<F32> & Tile.Tile8}
    {==})

def written(+tree: Tree.Tree<F32>,+position: U32,tile: Tile.Tile8) -> Tree.Tree<F32>:
  Tile.Tile{a,b,c,d,e,f,g,h} = tile
  UnaryLaws.written(tree,position,a,b,c,d,e,f,g,h)

def scattered(+tree: Tree.Tree<F32>,+position: U32,+tile: Tile.Tile8) ->
  {Unary.scatter(Tree.pack(F32,tree),position,tile) == Tree.pack(F32,written(tree,position,tile)) : Array<F32>}:
  Tile.Tile{a,b,c,d,e,f,g,h} = tile
  UnaryLaws.scattered(tree,position,a,b,c,d,e,f,g,h)

def write(+model: Buffer.Model<F32>,+position: U32,+tile: Tile.Tile8) ->
  {Buffer.contents(~F32,Lanes.write(Buffer.pack(~F32,model),position,tile)) ==
    Buffer.Content{written(Buffer.tree(~F32,model),position,tile),Buffer.length(~F32,model)} : Buffer.Content<F32>}:
  Buffer.Model{+tree,+length,certificate} = model
  Equal.cong(Tree.Tree<F32>,Buffer.Content<F32>,tree => Buffer.Content{tree,length},
    Tree.reflect(F32,Unary.scatter(Tree.pack(F32,tree),position,tile)),written(tree,position,tile),
    Equal.trans(Tree.Tree<F32>,Tree.reflect(F32,Unary.scatter(Tree.pack(F32,tree),position,tile)),Tree.reflect(F32,Tree.pack(F32,written(tree,position,tile))),written(tree,position,tile),
      Equal.cong(Array<F32>,Tree.Tree<F32>,values => Tree.reflect(F32,values),Unary.scatter(Tree.pack(F32,tree),position,tile),Tree.pack(F32,written(tree,position,tile)),scattered(tree,position,tile)),
      StorageLaws.reflect_packed(F32,written(tree,position,tile))))

def pair_contents(owners: Lanes.Binary) -> Buffer.Content<F32> & Buffer.Content<F32>:
  Lanes.Binary{left,right} = owners
  (Buffer.contents(~F32,left),Buffer.contents(~F32,right))

def pair_read_contents(result: Lanes.Binary & Tile.Tile8) -> (Buffer.Content<F32> & Buffer.Content<F32>) & Tile.Tile8:
  CellLaws.observe_gather(~Lanes.Binary,~(Buffer.Content<F32> & Buffer.Content<F32>),~Tile.Tile8,~pair_contents,result)

def pair_read_model(~operation: F32 -> F32 -> F32,left: Buffer.Content<F32> & Tile.Tile8,right: Buffer.Content<F32> & Tile.Tile8) ->
  (Buffer.Content<F32> & Buffer.Content<F32>) & Tile.Tile8:
  (left,a) = left
  (right,b) = right
  ((left,right),Lanes.combine_tile(~operation,a,b))

def pair_reads(~operation: F32 -> F32 -> F32,left: Storage.Buffer<F32> & Tile.Tile8,right: Storage.Buffer<F32> & Tile.Tile8) ->
  {pair_read_contents(Lanes.pair_reads(~operation,left,right)) == pair_read_model(~operation,observed_read(left),observed_read(right)) :
    (Buffer.Content<F32> & Buffer.Content<F32>) & Tile.Tile8}:
  (left,a) = left
  (right,b) = right
  {==}

def pair_cell(~operation: F32 -> F32 -> F32,+left: Tree.Tree<F32>,+right: Tree.Tree<F32>,+position: U32) -> Tile.Tile8:
  Lanes.combine_tile(~operation,UnaryLaws.cell(~Unit,~Lanes.identity,left,position,Unit{}),UnaryLaws.cell(~Unit,~Lanes.identity,right,position,Unit{}))

def pair_gather(~operation: F32 -> F32 -> F32,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,+position: U32) ->
  {pair_read_contents(Lanes.pair_gather(~operation,Lanes.Binary{Buffer.pack(~F32,left),Buffer.pack(~F32,right)},position,Unit{})) ==
    ((Buffer.content(~F32,left),Buffer.content(~F32,right)),pair_cell(~operation,Buffer.tree(~F32,left),Buffer.tree(~F32,right),position)) :
    (Buffer.Content<F32> & Buffer.Content<F32>) & Tile.Tile8}:
  Equal.trans((Buffer.Content<F32> & Buffer.Content<F32>) & Tile.Tile8,
    pair_read_contents(Lanes.pair_gather(~operation,Lanes.Binary{Buffer.pack(~F32,left),Buffer.pack(~F32,right)},position,Unit{})),
    pair_read_model(~operation,observed_read(Lanes.read(Buffer.pack(~F32,left),position)),observed_read(Lanes.read(Buffer.pack(~F32,right),position))),
    ((Buffer.content(~F32,left),Buffer.content(~F32,right)),pair_cell(~operation,Buffer.tree(~F32,left),Buffer.tree(~F32,right),position)),
    pair_reads(~operation,Lanes.read(Buffer.pack(~F32,left),position),Lanes.read(Buffer.pack(~F32,right),position)),
    %Equal.sym(Buffer.Content<F32> & Tile.Tile8,observed_read(Lanes.read(Buffer.pack(~F32,left),position)),
      (Buffer.content(~F32,left),UnaryLaws.cell(~Unit,~Lanes.identity,Buffer.tree(~F32,left),position,Unit{})),read(left,position)) :
      {pair_read_model(~operation,_,observed_read(Lanes.read(Buffer.pack(~F32,right),position))) ==
        ((Buffer.content(~F32,left),Buffer.content(~F32,right)),pair_cell(~operation,Buffer.tree(~F32,left),Buffer.tree(~F32,right),position)) :
        (Buffer.Content<F32> & Buffer.Content<F32>) & Tile.Tile8}
    %Equal.sym(Buffer.Content<F32> & Tile.Tile8,observed_read(Lanes.read(Buffer.pack(~F32,right),position)),
      (Buffer.content(~F32,right),UnaryLaws.cell(~Unit,~Lanes.identity,Buffer.tree(~F32,right),position,Unit{})),read(right,position)) :
      {pair_read_model(~operation,(Buffer.content(~F32,left),UnaryLaws.cell(~Unit,~Lanes.identity,Buffer.tree(~F32,left),position,Unit{})),_) ==
        ((Buffer.content(~F32,left),Buffer.content(~F32,right)),pair_cell(~operation,Buffer.tree(~F32,left),Buffer.tree(~F32,right),position)) :
        (Buffer.Content<F32> & Buffer.Content<F32>) & Tile.Tile8}
    {==})

def pair_scatter(+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,+position: U32,+tile: Tile.Tile8) ->
  {pair_contents(Lanes.pair_scatter(Lanes.Binary{Buffer.pack(~F32,left),Buffer.pack(~F32,right)},position,tile)) ==
    (Buffer.Content{written(Buffer.tree(~F32,left),position,tile),Buffer.length(~F32,left)},Buffer.content(~F32,right)) : Buffer.Content<F32> & Buffer.Content<F32>}:
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Lanes.write(Buffer.pack(~F32,left),position,tile)),
    Buffer.Content{written(Buffer.tree(~F32,left),position,tile),Buffer.length(~F32,left)},write(left,position,tile)) :
    {(_,Buffer.contents(~F32,Buffer.pack(~F32,right))) ==
      (Buffer.Content{written(Buffer.tree(~F32,left),position,tile),Buffer.length(~F32,left)},Buffer.content(~F32,right)) : Buffer.Content<F32> & Buffer.Content<F32>}
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,right)),Buffer.content(~F32,right),Buffer.packed_contents(~F32,right)) :
    {(Buffer.Content{written(Buffer.tree(~F32,left),position,tile),Buffer.length(~F32,left)},_) ==
      (Buffer.Content{written(Buffer.tree(~F32,left),position,tile),Buffer.length(~F32,left)},Buffer.content(~F32,right)) : Buffer.Content<F32> & Buffer.Content<F32>}
  {==}

def pair_scatter_model(owners: Buffer.Content<F32> & Buffer.Content<F32>,position: U32,tile: Tile.Tile8) -> Buffer.Content<F32> & Buffer.Content<F32>:
  (Buffer.Content{left,length},right) = owners
  (Buffer.Content{written(left,position,tile),length},right)

def pair_packed_contents(+left: Buffer.Model<F32>,+right: Buffer.Model<F32>) ->
  {pair_contents(Lanes.Binary{Buffer.pack(~F32,left),Buffer.pack(~F32,right)}) == (Buffer.content(~F32,left),Buffer.content(~F32,right)) : Buffer.Content<F32> & Buffer.Content<F32>}:
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,left)),Buffer.content(~F32,left),Buffer.packed_contents(~F32,left)) :
    {(_,Buffer.contents(~F32,Buffer.pack(~F32,right))) == (Buffer.content(~F32,left),Buffer.content(~F32,right)) : Buffer.Content<F32> & Buffer.Content<F32>}
  Equal.cong(Buffer.Content<F32>,Buffer.Content<F32> & Buffer.Content<F32>,content => (Buffer.content(~F32,left),content),
    Buffer.contents(~F32,Buffer.pack(~F32,right)),Buffer.content(~F32,right),Buffer.packed_contents(~F32,right))

def pair_scatter_seen(-left: Storage.Buffer<F32>,-right: Storage.Buffer<F32>,+position: U32,+tile: Tile.Tile8,
  left_seen: Buffer.Observation<F32,left>,right_seen: Buffer.Observation<F32,right>) ->
  {pair_contents(Lanes.pair_scatter(Lanes.Binary{left,right},position,tile)) ==
    pair_scatter_model(pair_contents(Lanes.Binary{left,right}),position,tile) : Buffer.Content<F32> & Buffer.Content<F32>}:
  Buffer.Observation{Buffer.Model{+at,+al,+ac},left_equal} = left_seen
  Buffer.Observation{Buffer.Model{+bt,+bl,+bc},right_equal} = right_seen
  +a = {Buffer.Model{at,al,ac} : Buffer.Model<F32>}
  +b = {Buffer.Model{bt,bl,bc} : Buffer.Model<F32>}
  %left_equal : {pair_contents(Lanes.pair_scatter(Lanes.Binary{_,right},position,tile)) == pair_scatter_model(pair_contents(Lanes.Binary{_,right}),position,tile) : Buffer.Content<F32> & Buffer.Content<F32>}
  %right_equal : {pair_contents(Lanes.pair_scatter(Lanes.Binary{Buffer.pack(~F32,a),_},position,tile)) == pair_scatter_model(pair_contents(Lanes.Binary{Buffer.pack(~F32,a),_}),position,tile) : Buffer.Content<F32> & Buffer.Content<F32>}
  Equal.trans(Buffer.Content<F32> & Buffer.Content<F32>,
    pair_contents(Lanes.pair_scatter(Lanes.Binary{Buffer.pack(~F32,a),Buffer.pack(~F32,b)},position,tile)),
    pair_scatter_model((Buffer.content(~F32,a),Buffer.content(~F32,b)),position,tile),
    pair_scatter_model(pair_contents(Lanes.Binary{Buffer.pack(~F32,a),Buffer.pack(~F32,b)}),position,tile),
    pair_scatter(a,b,position,tile),
    Equal.cong(Buffer.Content<F32> & Buffer.Content<F32>,Buffer.Content<F32> & Buffer.Content<F32>,owners => pair_scatter_model(owners,position,tile),
      (Buffer.content(~F32,a),Buffer.content(~F32,b)),pair_contents(Lanes.Binary{Buffer.pack(~F32,a),Buffer.pack(~F32,b)}),
      Equal.sym(Buffer.Content<F32> & Buffer.Content<F32>,pair_contents(Lanes.Binary{Buffer.pack(~F32,a),Buffer.pack(~F32,b)}),
        (Buffer.content(~F32,a),Buffer.content(~F32,b)),pair_packed_contents(a,b))))

def pair_scatter_native(owners: Lanes.Binary,+position: U32,+tile: Tile.Tile8) ->
  {pair_contents(Lanes.pair_scatter(owners,position,tile)) == pair_scatter_model(pair_contents(owners),position,tile) : Buffer.Content<F32> & Buffer.Content<F32>}:
  Lanes.Binary{left,right} = owners
  pair_scatter_seen(left,right,position,tile,Buffer.observe(~F32,left),Buffer.observe(~F32,right))

# The binary value is just the indexed unary model with a read-only context.
def pair_value(~operation: F32 -> F32 -> F32,right: Tree.Tree<F32>,position: U32,left: F32) -> F32:
  operation(left,Tree.read(F32,right,position))

def pair_state(state: Batch.ModelState(~F32,~Tree.Tree<F32>)) -> Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,Unit>:
  Traversal.State{Unit{},left,position,right} = state
  Traversal.State{right,left,position,Unit{}}

def pair_step(~operation: F32 -> F32 -> F32,+state: Batch.ModelState(~F32,~Tree.Tree<F32>)) ->
  {ArrayLaws.model_update_step(~Tree.Tree<F32>,~F32,~F32,~Unit,~AccessLaws.sample_index(~F32),~operation,~Traversal.index,pair_state(state)) ==
    pair_state(ArrayLaws.model_map_step(~F32,~Tree.Tree<F32>,~pair_value(~operation),state)) : Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,Unit>}:
  Traversal.State{Unit{},left,position,right} = state
  {==}

def pair_loop(~operation: F32 -> F32 -> F32,+count: Nat,+start: Nat,+right: Tree.Tree<F32>,+left: Tree.Tree<F32>) ->
  {Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,Unit>,
      ~ArrayLaws.model_update_step(~Tree.Tree<F32>,~F32,~F32,~Unit,~AccessLaws.sample_index(~F32),~operation,~Traversal.index),count,
      Traversal.State{right,left,U32.from_nat(start),Unit{}}) ==
    Traversal.State{right,MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,U32.from_nat(start),right,left),U32.from_nat(Nat.add(start,count)),Unit{}} :
    Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,Unit>}:
  +state = {Traversal.State{Unit{},left,U32.from_nat(start),right} : Batch.ModelState(~F32,~Tree.Tree<F32>)}
  %Batch.state_after(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,start,right,left) :
    {Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,Unit>,
      ~ArrayLaws.model_update_step(~Tree.Tree<F32>,~F32,~F32,~Unit,~AccessLaws.sample_index(~F32),~operation,~Traversal.index),count,pair_state(state)) == pair_state(_) :
      Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,Unit>}
  Projection.iteration_refines_model(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,Unit>,~Batch.ModelState(~F32,~Tree.Tree<F32>),
    ~ArrayLaws.model_update_step(~Tree.Tree<F32>,~F32,~F32,~Unit,~AccessLaws.sample_index(~F32),~operation,~Traversal.index),
    ~ArrayLaws.model_map_step(~F32,~Tree.Tree<F32>,~pair_value(~operation)),~pair_state,~pair_step(~operation),count,state)

def scalar_fast(~operation: F32 -> F32 -> F32,+count: Nat,+source: Views.View,+position: U32,+right: Tree.Tree<F32>,+left: Tree.Tree<F32>) ->
  {AccessLaws.combine_model(~F32,~operation,True{},count,source,position,right,left) ==
    AccessLaws.ModelResult{right,MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,0,right,left)} : AccessLaws.ModelResult<Tree.Tree<F32>,F32>}:
  Equal.cong(Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,Unit>,AccessLaws.ModelResult<Tree.Tree<F32>,F32>,
    state => AccessLaws.finish_model(~Tree.Tree<F32>,~F32,~Unit,state),
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,Unit>,
      ~ArrayLaws.model_update_step(~Tree.Tree<F32>,~F32,~F32,~Unit,~AccessLaws.sample_index(~F32),~operation,~Traversal.index),count,Traversal.State{right,left,0,Unit{}}),
    Traversal.State{right,MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,0,right,left),U32.from_nat(count),Unit{}},pair_loop(~operation,count,0n,right,left))

def cursor_state_identity(state: Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,CursorLaws.Coordinates>) -> Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,CursorLaws.Coordinates>:
  state

def increment_cursor_step(~operation: F32 -> F32 -> F32,state: Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,CursorLaws.Coordinates>) -> Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,CursorLaws.Coordinates>:
  Traversal.State{+right,+left,+position,+cursor} = state
  Traversal.State{right,Tree.write(F32,left,position,operation(Tree.read(F32,left,position),TensorVisit.sample(~F32,right,cursor))),U32.inc(position),CursorLaws.next(cursor)}

def increment_cursor_equal(~operation: F32 -> F32 -> F32,+state: Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,CursorLaws.Coordinates>) ->
  {Visit.model_update_step(~Tree.Tree<F32>,~F32,~CursorLaws.Coordinates,~TensorVisit.sample(~F32),~CursorLaws.next,~operation,state) ==
    increment_cursor_step(~operation,state) : Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,CursorLaws.Coordinates>}:
  Traversal.State{+right,+left,+position,+cursor} = state
  Equal.cong(U32,Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,CursorLaws.Coordinates>,next =>
    Traversal.State{right,Tree.write(F32,left,position,operation(Tree.read(F32,left,position),TensorVisit.sample(~F32,right,cursor))),next,CursorLaws.next(cursor)},
    (position + 1 : U32),U32.inc(position),Bounds.add_one(position))

def increment_cursor(~operation: F32 -> F32 -> F32,count: Nat,source: Views.View,position: U32,right: Tree.Tree<F32>,left: Tree.Tree<F32>) -> AccessLaws.ModelResult<Tree.Tree<F32>,F32>:
  AccessLaws.finish_model(~Tree.Tree<F32>,~F32,~CursorLaws.Coordinates,
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,CursorLaws.Coordinates>,~increment_cursor_step(~operation),count,Traversal.State{right,left,position,CursorLaws.initial(source)}))

def scalar_slow_increment(~operation: F32 -> F32 -> F32,+count: Nat,+source: Views.View,+position: U32,+right: Tree.Tree<F32>,+left: Tree.Tree<F32>) ->
  {AccessLaws.combine_model(~F32,~operation,False{},count,source,position,right,left) == increment_cursor(~operation,count,source,position,right,left) : AccessLaws.ModelResult<Tree.Tree<F32>,F32>}:
  +state = {Traversal.State{right,left,position,CursorLaws.initial(source)} : Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,CursorLaws.Coordinates>}
  Equal.cong(Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,CursorLaws.Coordinates>,AccessLaws.ModelResult<Tree.Tree<F32>,F32>,
    state => AccessLaws.finish_model(~Tree.Tree<F32>,~F32,~CursorLaws.Coordinates,state),
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,CursorLaws.Coordinates>,
      ~Visit.model_update_step(~Tree.Tree<F32>,~F32,~CursorLaws.Coordinates,~TensorVisit.sample(~F32),~CursorLaws.next,~operation),count,state),
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,CursorLaws.Coordinates>,~increment_cursor_step(~operation),count,state),
    Projection.iteration_refines_model(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,CursorLaws.Coordinates>,~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,CursorLaws.Coordinates>,
      ~Visit.model_update_step(~Tree.Tree<F32>,~F32,~CursorLaws.Coordinates,~TensorVisit.sample(~F32),~CursorLaws.next,~operation),
      ~increment_cursor_step(~operation),~cursor_state_identity,~increment_cursor_equal(~operation),count,state))

def increment_direct_step(~operation: F32 -> F32 -> F32,state: Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,U32>) -> Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,U32>:
  Traversal.State{+right,+left,+position,+source} = state
  Traversal.State{right,Tree.write(F32,left,position,operation(Tree.read(F32,left,position),Tree.read(F32,right,source))),U32.inc(position),U32.inc(source)}

def increment_direct(~operation: F32 -> F32 -> F32,count: Nat,source: U32,position: U32,right: Tree.Tree<F32>,left: Tree.Tree<F32>) -> AccessLaws.ModelResult<Tree.Tree<F32>,F32>:
  AccessLaws.finish_model(~Tree.Tree<F32>,~F32,~U32,
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,U32>,~increment_direct_step(~operation),count,Traversal.State{right,left,position,source}))

def tail_increment_small(~operation: F32 -> F32 -> F32,+count: Nat,+position: U32,+right: Tree.Tree<F32>,+left: Tree.Tree<F32>,small: {Nat.is_lt(count,8n) == True{} : Bool}) ->
  {increment_cursor(~operation,count,Views.View{[Views.Axis{U32.from_nat(count),1}],position},position,right,left) ==
    increment_direct(~operation,count,U32.from_nat(U32.to_nat(position)),position,right,left) : AccessLaws.ModelResult<Tree.Tree<F32>,F32>}:
  match count:
    case 0n: {==}
    case 1n: {==}
    case 2n: {==}
    case 3n: {==}
    case 4n: {==}
    case 5n: {==}
    case 6n: {==}
    case 7n: {==}
    case 1n+1n+1n+1n+1n+1n+1n+1n+ +rest:
      Empty.absurd({increment_cursor(~operation,8n+rest,Views.View{[Views.Axis{U32.from_nat(8n+rest),1}],position},position,right,left) ==
        increment_direct(~operation,8n+rest,U32.from_nat(U32.to_nat(position)),position,right,left) : AccessLaws.ModelResult<Tree.Tree<F32>,F32>},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(rest,0n),True{},Equal.sym(Bool,Nat.is_lt(rest,0n),False{},Division.lt_zero(rest)),small)))

def direct_state(state: Batch.ModelState(~F32,~Tree.Tree<F32>)) -> Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,U32>:
  Traversal.State{Unit{},left,+position,right} = state
  Traversal.State{right,left,position,position}

def direct_step_equal(~operation: F32 -> F32 -> F32,+state: Batch.ModelState(~F32,~Tree.Tree<F32>)) ->
  {increment_direct_step(~operation,direct_state(state)) == direct_state(ArrayLaws.model_map_step(~F32,~Tree.Tree<F32>,~pair_value(~operation),state)) :
    Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,U32>}:
  Traversal.State{Unit{},+left,+position,+right} = state
  Equal.cong(U32,Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,U32>,next =>
    Traversal.State{right,Tree.write(F32,left,position,operation(Tree.read(F32,left,position),Tree.read(F32,right,position))),next,next},
    U32.inc(position),(position + 1 : U32),Equal.sym(U32,(position + 1 : U32),U32.inc(position),Bounds.add_one(position)))

def direct_loop(~operation: F32 -> F32 -> F32,+count: Nat,+start: Nat,+right: Tree.Tree<F32>,+left: Tree.Tree<F32>) ->
  {Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,U32>,~increment_direct_step(~operation),count,
      Traversal.State{right,left,U32.from_nat(start),U32.from_nat(start)}) ==
    Traversal.State{right,MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,U32.from_nat(start),right,left),U32.from_nat(Nat.add(start,count)),U32.from_nat(Nat.add(start,count))} :
    Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,U32>}:
  +state = {Traversal.State{Unit{},left,U32.from_nat(start),right} : Batch.ModelState(~F32,~Tree.Tree<F32>)}
  %Batch.state_after(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,start,right,left) :
    {Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,U32>,~increment_direct_step(~operation),count,direct_state(state)) == direct_state(_) :
      Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,U32>}
  Projection.iteration_refines_model(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,U32>,~Batch.ModelState(~F32,~Tree.Tree<F32>),
    ~increment_direct_step(~operation),~ArrayLaws.model_map_step(~F32,~Tree.Tree<F32>,~pair_value(~operation)),~direct_state,~direct_step_equal(~operation),count,state)

def direct_mapped(~operation: F32 -> F32 -> F32,+count: Nat,+start: Nat,+right: Tree.Tree<F32>,+left: Tree.Tree<F32>) ->
  {increment_direct(~operation,count,U32.from_nat(start),U32.from_nat(start),right,left) ==
    AccessLaws.ModelResult{right,MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,U32.from_nat(start),right,left)} : AccessLaws.ModelResult<Tree.Tree<F32>,F32>}:
  Equal.cong(Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,U32>,AccessLaws.ModelResult<Tree.Tree<F32>,F32>,state => AccessLaws.finish_model(~Tree.Tree<F32>,~F32,~U32,state),
    Loop.run(~Traversal.State<&2,&2,Tree.Tree<F32>,Tree.Tree<F32>,U32>,~increment_direct_step(~operation),count,Traversal.State{right,left,U32.from_nat(start),U32.from_nat(start)}),
    Traversal.State{right,MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,U32.from_nat(start),right,left),U32.from_nat(Nat.add(start,count)),U32.from_nat(Nat.add(start,count))},
    direct_loop(~operation,count,start,right,left))

def scalar_tail_slow(~operation: F32 -> F32 -> F32,+count: Nat,+start: Nat,+right: Tree.Tree<F32>,+left: Tree.Tree<F32>,small: {Nat.is_lt(count,8n) == True{} : Bool}) ->
  {AccessLaws.combine_model(~F32,~operation,False{},count,Views.View{[Views.Axis{U32.from_nat(count),1}],U32.from_nat(start)},U32.from_nat(start),right,left) ==
    AccessLaws.ModelResult{right,MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,U32.from_nat(start),right,left)} : AccessLaws.ModelResult<Tree.Tree<F32>,F32>}:
  +position = U32.from_nat(start)
  +source = {Views.View{[Views.Axis{U32.from_nat(count),1}],position} : Views.View}
  Equal.trans(AccessLaws.ModelResult<Tree.Tree<F32>,F32>,AccessLaws.combine_model(~F32,~operation,False{},count,source,position,right,left),
    increment_cursor(~operation,count,source,position,right,left),AccessLaws.ModelResult{right,MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,position,right,left)},
    scalar_slow_increment(~operation,count,source,position,right,left),
    Equal.trans(AccessLaws.ModelResult<Tree.Tree<F32>,F32>,increment_cursor(~operation,count,source,position,right,left),
      increment_direct(~operation,count,position,position,right,left),AccessLaws.ModelResult{right,MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,position,right,left)},
      %Arithmetic.native_roundtrip(position) : {increment_cursor(~operation,count,source,position,right,left) == increment_direct(~operation,count,_,position,right,left) : AccessLaws.ModelResult<Tree.Tree<F32>,F32>}
      tail_increment_small(~operation,count,position,right,left,small),direct_mapped(~operation,count,start,right,left)))

def scalar_tail_when(~operation: F32 -> F32 -> F32,+fast: Bool,+count: Nat,+start: Nat,+right: Tree.Tree<F32>,+left: Tree.Tree<F32>,
  equation: {Access.zero_contiguous(Views.View{[Views.Axis{U32.from_nat(count),1}],U32.from_nat(start)}) && U32.is_zero(U32.from_nat(start)) == fast : Bool},
  small: {Nat.is_lt(count,8n) == True{} : Bool}) ->
  {AccessLaws.combine_model(~F32,~operation,fast,count,Views.View{[Views.Axis{U32.from_nat(count),1}],U32.from_nat(start)},U32.from_nat(start),right,left) ==
    AccessLaws.ModelResult{right,MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,U32.from_nat(start),right,left)} : AccessLaws.ModelResult<Tree.Tree<F32>,F32>}:
  match fast:
    case False{}: scalar_tail_slow(~operation,count,start,right,left,small)
    case True{}:
      +position = U32.from_nat(start)
      +source = {Views.View{[Views.Axis{U32.from_nat(count),1}],position} : Views.View}
      +zero = ShapeLaws.native_equal(position,0,Boolean.right(Access.zero_contiguous(source),U32.is_zero(position),equation))
      %Equal.sym(U32,position,0,zero) :
        {AccessLaws.combine_model(~F32,~operation,True{},count,source,position,right,left) ==
          AccessLaws.ModelResult{right,MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,_,right,left)} : AccessLaws.ModelResult<Tree.Tree<F32>,F32>}
      scalar_fast(~operation,count,source,position,right,left)

def pair_result_contents(result: Storage.Buffer<F32> & Storage.Buffer<F32>) ->
  {pair_contents(Lanes.pair_result(result)) == CloneLaws.contents(~F32,result) : Buffer.Content<F32> & Buffer.Content<F32>}:
  (left,right) = result
  {==}

def pair_tail(~operation: F32 -> F32 -> F32,+count: Nat,+start: Nat,+slot: U32,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,+scratch: Tree.Tree<Tile.Tile8>,
  small: {Nat.is_lt(count,8n) == True{} : Bool}) ->
  {pair_contents(Lanes.pair_finish(~operation,count,Cells.State{Lanes.Binary{Buffer.pack(~F32,left),Buffer.pack(~F32,right)},Tree.pack(Tile.Tile8,scratch),U32.from_nat(start),slot,Unit{}})) ==
    (Buffer.Content{MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,U32.from_nat(start),Buffer.tree(~F32,right),Buffer.tree(~F32,left)),Buffer.length(~F32,left)},
      Buffer.content(~F32,right)) : Buffer.Content<F32> & Buffer.Content<F32>}:
  Buffer.Model{+a,+al,+ac} = left
  Buffer.Model{+b,+bl,+bc} = right
  +position = U32.from_nat(start)
  +source = {Views.View{[Views.Axis{U32.from_nat(count),1}],position} : Views.View}
  +fast = Access.zero_contiguous(source) && U32.is_zero(position)
  +model = AccessLaws.combine_model(~F32,~operation,fast,count,source,position,b,a)
  +mapped = MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,position,b,a)
  Equal.trans(Buffer.Content<F32> & Buffer.Content<F32>,
    pair_contents(Lanes.pair_result(Operations.combine_kernel(~F32,~operation,count,source,position,Storage.Buffer{Tree.pack(F32,a),al,ac},Storage.Buffer{Tree.pack(F32,b),bl,bc}))),
    CloneLaws.contents(~F32,Operations.combine_kernel(~F32,~operation,count,source,position,Storage.Buffer{Tree.pack(F32,a),al,ac},Storage.Buffer{Tree.pack(F32,b),bl,bc})),
    (Buffer.Content{mapped,al},Buffer.Content{b,bl}),
    pair_result_contents(Operations.combine_kernel(~F32,~operation,count,source,position,Storage.Buffer{Tree.pack(F32,a),al,ac},Storage.Buffer{Tree.pack(F32,b),bl,bc})),
    Equal.trans(Buffer.Content<F32> & Buffer.Content<F32>,
      CloneLaws.contents(~F32,Operations.combine_kernel(~F32,~operation,count,source,position,Storage.Buffer{Tree.pack(F32,a),al,ac},Storage.Buffer{Tree.pack(F32,b),bl,bc})),
      pair_model_contents(al,bl,model),(Buffer.Content{mapped,al},Buffer.Content{b,bl}),scalar_core(~operation,count,source,position,Buffer.Model{a,al,ac},Buffer.Model{b,bl,bc}),
      Equal.cong(AccessLaws.ModelResult<Tree.Tree<F32>,F32>,Buffer.Content<F32> & Buffer.Content<F32>,result => pair_model_contents(al,bl,result),
        model,AccessLaws.ModelResult{b,mapped},scalar_tail_when(~operation,fast,count,start,b,a,{==},small))))

def pair_batch(~operation: F32 -> F32 -> F32,+left: Tree.Tree<F32>,+right: Tree.Tree<F32>,+position: U32) ->
  {written(left,position,pair_cell(~operation,left,right,position)) ==
    Batch.batch(~F32,~Tree.Tree<F32>,~pair_value(~operation),8n,position,right,left,left) : Tree.Tree<F32>}:
  {==}

# No independent read-over-write induction: the existing batch theorem covers
# the binary round, including preservation of the left values before use.
def pair_batch_mapped(~operation: F32 -> F32 -> F32,+left: Tree.Tree<F32>,+right: Tree.Tree<F32>,+start: Nat,+depth: Nat,+capacity: U32,
  +capacity_equal: {Tree.capacity(F32,left) == capacity : U32},+balanced: Shape.Balanced(F32,depth,left),
  +limit: {Nat.is_le(depth,31n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(start,8n),U32.to_nat(capacity)) == True{} : Bool}) ->
  {written(left,U32.from_nat(start),pair_cell(~operation,left,right,U32.from_nat(start))) ==
    MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),8n,U32.from_nat(start),right,left) : Tree.Tree<F32>}:
  Equal.trans(Tree.Tree<F32>,written(left,U32.from_nat(start),pair_cell(~operation,left,right,U32.from_nat(start))),
    Batch.batch(~F32,~Tree.Tree<F32>,~pair_value(~operation),8n,U32.from_nat(start),right,left,left),
    MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),8n,U32.from_nat(start),right,left),
    pair_batch(~operation,left,right,U32.from_nat(start)),
    Batch.mapped(~F32,~Tree.Tree<F32>,~pair_value(~operation),8n,start,right,left,depth,capacity,capacity_equal,balanced,limit,capacity_ok,room))

def pair_round(~operation: F32 -> F32 -> F32,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,+scratch: Tree.Tree<Tile.Tile8>,+position: U32,+slot: U32) ->
  {CellLaws.observe(~Lanes.Binary,~(Buffer.Content<F32> & Buffer.Content<F32>),~Tile.Tile8,~Unit,~pair_contents,
      Cells.step(~Lanes.Binary,~Tile.Tile8,~Unit,~Lanes.pair_gather(~operation),~Lanes.pair_scatter,~8,
        Cells.State{Lanes.Binary{Buffer.pack(~F32,left),Buffer.pack(~F32,right)},Tree.pack(Tile.Tile8,scratch),position,slot,Unit{}})) ==
    CellLaws.model_stored(~(Buffer.Content<F32> & Buffer.Content<F32>),~Tile.Tile8,~Unit,~pair_scatter_model,~8,position,slot,Unit{},scratch,
      ((Buffer.content(~F32,left),Buffer.content(~F32,right)),pair_cell(~operation,Buffer.tree(~F32,left),Buffer.tree(~F32,right),position))) :
    CellLaws.Observed<(Buffer.Content<F32> & Buffer.Content<F32>),Tile.Tile8,Unit>}:
  Equal.trans(CellLaws.Observed<(Buffer.Content<F32> & Buffer.Content<F32>),Tile.Tile8,Unit>,
    CellLaws.observe(~Lanes.Binary,~(Buffer.Content<F32> & Buffer.Content<F32>),~Tile.Tile8,~Unit,~pair_contents,
      Cells.step(~Lanes.Binary,~Tile.Tile8,~Unit,~Lanes.pair_gather(~operation),~Lanes.pair_scatter,~8,
        Cells.State{Lanes.Binary{Buffer.pack(~F32,left),Buffer.pack(~F32,right)},Tree.pack(Tile.Tile8,scratch),position,slot,Unit{}})),
    CellLaws.model_stored(~(Buffer.Content<F32> & Buffer.Content<F32>),~Tile.Tile8,~Unit,~pair_scatter_model,~8,position,slot,Unit{},scratch,
      pair_read_contents(Lanes.pair_gather(~operation,Lanes.Binary{Buffer.pack(~F32,left),Buffer.pack(~F32,right)},position,Unit{}))),
    CellLaws.model_stored(~(Buffer.Content<F32> & Buffer.Content<F32>),~Tile.Tile8,~Unit,~pair_scatter_model,~8,position,slot,Unit{},scratch,
      ((Buffer.content(~F32,left),Buffer.content(~F32,right)),pair_cell(~operation,Buffer.tree(~F32,left),Buffer.tree(~F32,right),position))),
    CellLaws.stored_observed(~Lanes.Binary,~(Buffer.Content<F32> & Buffer.Content<F32>),~Tile.Tile8,~Unit,~pair_contents,
      ~Lanes.pair_scatter,~pair_scatter_model,~8,~pair_scatter_native,position,slot,Unit{},scratch,
      Lanes.pair_gather(~operation,Lanes.Binary{Buffer.pack(~F32,left),Buffer.pack(~F32,right)},position,Unit{})),
    Equal.cong((Buffer.Content<F32> & Buffer.Content<F32>) & Tile.Tile8,CellLaws.Observed<(Buffer.Content<F32> & Buffer.Content<F32>),Tile.Tile8,Unit>,
      result => CellLaws.model_stored(~(Buffer.Content<F32> & Buffer.Content<F32>),~Tile.Tile8,~Unit,~pair_scatter_model,~8,position,slot,Unit{},scratch,result),
      pair_read_contents(Lanes.pair_gather(~operation,Lanes.Binary{Buffer.pack(~F32,left),Buffer.pack(~F32,right)},position,Unit{})),
      ((Buffer.content(~F32,left),Buffer.content(~F32,right)),pair_cell(~operation,Buffer.tree(~F32,left),Buffer.tree(~F32,right),position)),pair_gather(~operation,left,right,position)))

def packed_state(left: Buffer.Model<F32>,right: Buffer.Model<F32>,scratch: Tree.Tree<Tile.Tile8>,position: U32,slot: U32) -> Cells.State<Lanes.Binary,Tile.Tile8,Unit>:
  Cells.State{Lanes.Binary{Buffer.pack(~F32,left),Buffer.pack(~F32,right)},Tree.pack(Tile.Tile8,scratch),position,slot,Unit{}}

def state_contents(state: Cells.State<Lanes.Binary,Tile.Tile8,Unit>) -> CellLaws.Observed<(Buffer.Content<F32> & Buffer.Content<F32>),Tile.Tile8,Unit>:
  CellLaws.observe(~Lanes.Binary,~(Buffer.Content<F32> & Buffer.Content<F32>),~Tile.Tile8,~Unit,~pair_contents,state)

def packed_state_contents(+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,+scratch: Tree.Tree<Tile.Tile8>,+position: U32,+slot: U32) ->
  {state_contents(packed_state(left,right,scratch,position,slot)) ==
    CellLaws.Observed{(Buffer.content(~F32,left),Buffer.content(~F32,right)),scratch,position,slot,Unit{}} : CellLaws.Observed<(Buffer.Content<F32> & Buffer.Content<F32>),Tile.Tile8,Unit>}:
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,scratch)),scratch,StorageLaws.reflect_packed(Tile.Tile8,scratch)) :
    {CellLaws.Observed{pair_contents(Lanes.Binary{Buffer.pack(~F32,left),Buffer.pack(~F32,right)}),_,position,slot,Unit{}} ==
      CellLaws.Observed{(Buffer.content(~F32,left),Buffer.content(~F32,right)),scratch,position,slot,Unit{}} : CellLaws.Observed<(Buffer.Content<F32> & Buffer.Content<F32>),Tile.Tile8,Unit>}
  Equal.cong(Buffer.Content<F32> & Buffer.Content<F32>,CellLaws.Observed<(Buffer.Content<F32> & Buffer.Content<F32>),Tile.Tile8,Unit>,
    owners => CellLaws.Observed{owners,scratch,position,slot,Unit{}},pair_contents(Lanes.Binary{Buffer.pack(~F32,left),Buffer.pack(~F32,right)}),
    (Buffer.content(~F32,left),Buffer.content(~F32,right)),pair_packed_contents(left,right))

type StateObservation<-state: Cells.State<Lanes.Binary,Tile.Tile8,Unit>> is Data:
  StateObservation{left: Buffer.Model<F32>,right: Buffer.Model<F32>,scratch: Tree.Tree<Tile.Tile8>,position: U32,slot: U32,
    equation: {packed_state(left,right,scratch,position,slot) == state : Cells.State<Lanes.Binary,Tile.Tile8,Unit>}}

def state_seen(-left: Storage.Buffer<F32>,-right: Storage.Buffer<F32>,-scratch: Array<Tile.Tile8>,+position: U32,+slot: U32,
  left_seen: Buffer.Observation<F32,left>,right_seen: Buffer.Observation<F32,right>,scratch_seen: StorageLaws.Observation<Tile.Tile8,scratch>) ->
  StateObservation<Cells.State{Lanes.Binary{left,right},scratch,position,slot,Unit{}}>:
  Buffer.Observation{+a,left_equal} = left_seen
  Buffer.Observation{+b,right_equal} = right_seen
  StorageLaws.Observation{+tree,packed,reflected} = scratch_seen
  StateObservation{a,b,tree,position,slot,
    %left_equal : {packed_state(a,b,tree,position,slot) == Cells.State{Lanes.Binary{_,right},scratch,position,slot,Unit{}} : Cells.State<Lanes.Binary,Tile.Tile8,Unit>}
    %right_equal : {packed_state(a,b,tree,position,slot) == Cells.State{Lanes.Binary{Buffer.pack(~F32,a),_},scratch,position,slot,Unit{}} : Cells.State<Lanes.Binary,Tile.Tile8,Unit>}
    %packed : {packed_state(a,b,tree,position,slot) == Cells.State{Lanes.Binary{Buffer.pack(~F32,a),Buffer.pack(~F32,b)},_,position,slot,Unit{}} : Cells.State<Lanes.Binary,Tile.Tile8,Unit>}
    {==}}

def observe_state(state: Cells.State<Lanes.Binary,Tile.Tile8,Unit>) -> StateObservation<state>:
  Cells.State{Lanes.Binary{left,right},scratch,+position,+slot,Unit{}} = state
  state_seen(left,right,scratch,position,slot,Buffer.observe(~F32,left),Buffer.observe(~F32,right),StorageLaws.observe(Tile.Tile8,scratch))

def pair_rounds_property(~operation: F32 -> F32 -> F32,count: Nat) -> Type:
  @+left: Buffer.Model<F32> -> @+right: Buffer.Model<F32> -> @+scratch: Tree.Tree<Tile.Tile8> -> @+start: Nat -> @+slot: U32 ->
  @+depth: Nat -> @+capacity: U32 -> @+capacity_equal: {Tree.capacity(F32,Buffer.tree(~F32,left)) == capacity : U32} ->
  @+balanced: Shape.Balanced(F32,depth,Buffer.tree(~F32,left)) -> @+limit: {Nat.is_le(depth,31n) == True{} : Bool} ->
  @+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool} -> @+room: {Nat.is_le(Nat.add(start,count),U32.to_nat(capacity)) == True{} : Bool} ->
  {pair_contents(Unary.rounds_with(~Lanes.Binary,~Unit,~Lanes.pair_gather(~operation),~Lanes.pair_scatter,~Lanes.pair_finish(~operation),count,packed_state(left,right,scratch,U32.from_nat(start),slot))) ==
    (Buffer.Content{MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,U32.from_nat(start),Buffer.tree(~F32,right),Buffer.tree(~F32,left)),Buffer.length(~F32,left)},Buffer.content(~F32,right)) :
    Buffer.Content<F32> & Buffer.Content<F32>}

# Transport the recursive hypothesis through the actual owners' observation.
# Their live certificates need not be identified with an invented certificate.
def Continuation(~operation: F32 -> F32 -> F32,count: Nat,-state: Cells.State<Lanes.Binary,Tile.Tile8,Unit>,
  observed: CellLaws.Observed<(Buffer.Content<F32> & Buffer.Content<F32>),Tile.Tile8,Unit>) -> Type:
  CellLaws.Observed{(Buffer.Content{left,left_length},Buffer.Content{right,right_length}),scratch,position,slot,Unit{}} = observed
  @+start: Nat -> @position_equal: {position == U32.from_nat(start) : U32} -> @+depth: Nat -> @+capacity: U32 ->
  @+capacity_equal: {Tree.capacity(F32,left) == capacity : U32} -> @+balanced: Shape.Balanced(F32,depth,left) -> @+limit: {Nat.is_le(depth,31n) == True{} : Bool} ->
  @+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool} -> @+room: {Nat.is_le(Nat.add(start,count),U32.to_nat(capacity)) == True{} : Bool} ->
  {pair_contents(Unary.rounds_with(~Lanes.Binary,~Unit,~Lanes.pair_gather(~operation),~Lanes.pair_scatter,~Lanes.pair_finish(~operation),count,state)) ==
    (Buffer.Content{MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,position,right,left),left_length},Buffer.Content{right,right_length}) : Buffer.Content<F32> & Buffer.Content<F32>}

def continue_canonical(~operation: F32 -> F32 -> F32,+count: Nat,previous: pair_rounds_property(~operation,count),
  +left: Buffer.Model<F32>,+right: Buffer.Model<F32>,+scratch: Tree.Tree<Tile.Tile8>,+position: U32,+slot: U32) ->
  Continuation(~operation,count,packed_state(left,right,scratch,position,slot),
    CellLaws.Observed{(Buffer.content(~F32,left),Buffer.content(~F32,right)),scratch,position,slot,Unit{}}):
  Buffer.Model{+a,+al,+ac} = left
  Buffer.Model{+b,+bl,+bc} = right
  start => position_equal => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
    %Equal.sym(U32,position,U32.from_nat(start),position_equal) :
      {pair_contents(Unary.rounds_with(~Lanes.Binary,~Unit,~Lanes.pair_gather(~operation),~Lanes.pair_scatter,~Lanes.pair_finish(~operation),count,
        packed_state(Buffer.Model{a,al,ac},Buffer.Model{b,bl,bc},scratch,_,slot))) ==
        (Buffer.Content{MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,_,b,a),al},Buffer.Content{b,bl}) : Buffer.Content<F32> & Buffer.Content<F32>}
    previous(Buffer.Model{a,al,ac},Buffer.Model{b,bl,bc},scratch,start,slot,depth,capacity,capacity_equal,balanced,limit,capacity_ok,room)

def continue_seen(~operation: F32 -> F32 -> F32,+count: Nat,previous: pair_rounds_property(~operation,count),
  -state: Cells.State<Lanes.Binary,Tile.Tile8,Unit>,observation: StateObservation<state>) -> Continuation(~operation,count,state,state_contents(state)):
  StateObservation{+left,+right,+scratch,+position,+slot,equation} = observation
  %equation : Continuation(~operation,count,_,state_contents(_))
  %Equal.sym(CellLaws.Observed<(Buffer.Content<F32> & Buffer.Content<F32>),Tile.Tile8,Unit>,state_contents(packed_state(left,right,scratch,position,slot)),
    CellLaws.Observed{(Buffer.content(~F32,left),Buffer.content(~F32,right)),scratch,position,slot,Unit{}},packed_state_contents(left,right,scratch,position,slot)) :
    Continuation(~operation,count,packed_state(left,right,scratch,position,slot),_)
  continue_canonical(~operation,count,previous,left,right,scratch,position,slot)

def pair_round_mapped(~operation: F32 -> F32 -> F32,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,+scratch: Tree.Tree<Tile.Tile8>,+start: Nat,+slot: U32,
  +depth: Nat,+capacity: U32,+capacity_equal: {Tree.capacity(F32,Buffer.tree(~F32,left)) == capacity : U32},+balanced: Shape.Balanced(F32,depth,Buffer.tree(~F32,left)),
  +limit: {Nat.is_le(depth,31n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(start,8n),U32.to_nat(capacity)) == True{} : Bool}) ->
  {state_contents(Cells.step(~Lanes.Binary,~Tile.Tile8,~Unit,~Lanes.pair_gather(~operation),~Lanes.pair_scatter,~8,packed_state(left,right,scratch,U32.from_nat(start),slot))) ==
    CellLaws.Observed{(Buffer.Content{MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),8n,U32.from_nat(start),Buffer.tree(~F32,right),Buffer.tree(~F32,left)),Buffer.length(~F32,left)},Buffer.content(~F32,right)),
      Tree.write(Tile.Tile8,scratch,slot,pair_cell(~operation,Buffer.tree(~F32,left),Buffer.tree(~F32,right),U32.from_nat(start))),U32.from_nat(8n+start),U32.and((slot + 1 : U32),63),Unit{}} :
    CellLaws.Observed<(Buffer.Content<F32> & Buffer.Content<F32>),Tile.Tile8,Unit>}:
  Buffer.Model{+a,+al,+ac} = left
  Buffer.Model{+b,+bl,+bc} = right
  +position = U32.from_nat(start)
  +cell = pair_cell(~operation,a,b,position)
  +mapped = MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),8n,position,b,a)
  %UnaryLaws.after_round(start) :
    {state_contents(Cells.step(~Lanes.Binary,~Tile.Tile8,~Unit,~Lanes.pair_gather(~operation),~Lanes.pair_scatter,~8,
      packed_state(Buffer.Model{a,al,ac},Buffer.Model{b,bl,bc},scratch,position,slot))) ==
      CellLaws.Observed{(Buffer.Content{mapped,al},Buffer.Content{b,bl}),Tree.write(Tile.Tile8,scratch,slot,cell),_,U32.and((slot + 1 : U32),63),Unit{}} :
      CellLaws.Observed<(Buffer.Content<F32> & Buffer.Content<F32>),Tile.Tile8,Unit>}
  %pair_batch_mapped(~operation,a,b,start,depth,capacity,capacity_equal,balanced,limit,capacity_ok,room) :
    {state_contents(Cells.step(~Lanes.Binary,~Tile.Tile8,~Unit,~Lanes.pair_gather(~operation),~Lanes.pair_scatter,~8,
      packed_state(Buffer.Model{a,al,ac},Buffer.Model{b,bl,bc},scratch,position,slot))) ==
      CellLaws.Observed{(Buffer.Content{_,al},Buffer.Content{b,bl}),Tree.write(Tile.Tile8,scratch,slot,cell),(position + 8 : U32),U32.and((slot + 1 : U32),63),Unit{}} :
      CellLaws.Observed<(Buffer.Content<F32> & Buffer.Content<F32>),Tile.Tile8,Unit>}
  pair_round(~operation,Buffer.Model{a,al,ac},Buffer.Model{b,bl,bc},scratch,position,slot)


def pair_few_rounds(~operation: F32 -> F32 -> F32,+count: Nat,small: {Nat.is_lt(count,8n) == True{} : Bool}) -> pair_rounds_property(~operation,count):
  +small = small
  left => right => scratch => start => slot => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
    %Equal.sym(Lanes.Binary,
      Unary.rounds_with(~Lanes.Binary,~Unit,~Lanes.pair_gather(~operation),~Lanes.pair_scatter,~Lanes.pair_finish(~operation),count,packed_state(left,right,scratch,U32.from_nat(start),slot)),
      Lanes.pair_finish(~operation,count,packed_state(left,right,scratch,U32.from_nat(start),slot)),
      UnaryLaws.finish_small(~Lanes.Binary,~Unit,~Lanes.pair_gather(~operation),~Lanes.pair_scatter,~Lanes.pair_finish(~operation),count,packed_state(left,right,scratch,U32.from_nat(start),slot),small)) :
      {pair_contents(_) == (Buffer.Content{MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,U32.from_nat(start),Buffer.tree(~F32,right),Buffer.tree(~F32,left)),Buffer.length(~F32,left)},Buffer.content(~F32,right)) :
        Buffer.Content<F32> & Buffer.Content<F32>}
    pair_tail(~operation,count,start,slot,left,right,scratch,small)


def pair_more_canonical(~operation: F32 -> F32 -> F32,+rest: Nat,previous: pair_rounds_property(~operation,rest),
  +left: Buffer.Model<F32>,+right: Buffer.Model<F32>,+scratch: Tree.Tree<Tile.Tile8>,+start: Nat,+slot: U32,
  +depth: Nat,+capacity: U32,+capacity_equal: {Tree.capacity(F32,Buffer.tree(~F32,left)) == capacity : U32},
  +balanced: Shape.Balanced(F32,depth,Buffer.tree(~F32,left)),+limit: {Nat.is_le(depth,31n) == True{} : Bool},
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.add(start,8n+rest),U32.to_nat(capacity)) == True{} : Bool}) ->
  {pair_contents(Unary.rounds_with(~Lanes.Binary,~Unit,~Lanes.pair_gather(~operation),~Lanes.pair_scatter,~Lanes.pair_finish(~operation),8n+rest,packed_state(left,right,scratch,U32.from_nat(start),slot))) ==
    (Buffer.Content{MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),8n+rest,U32.from_nat(start),Buffer.tree(~F32,right),Buffer.tree(~F32,left)),Buffer.length(~F32,left)},Buffer.content(~F32,right)) :
    Buffer.Content<F32> & Buffer.Content<F32>}:
  Buffer.Model{+at,+al,+ac} = left
  Buffer.Model{+bt,+bl,+bc} = right
  +a = Buffer.tree(~F32,left)
  +b = Buffer.tree(~F32,right)
  +position = U32.from_nat(start)
  +mapped = MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),8n,position,b,a)
  +next_scratch = Tree.write(Tile.Tile8,scratch,slot,pair_cell(~operation,a,b,position))
  +turn = U32.and((slot + 1 : U32),63)
  +capacity_next = Equal.trans(U32,Tree.capacity(F32,mapped),Tree.capacity(F32,a),capacity,
    MapReadback.capacity_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),8n,position,b,a),capacity_equal)
  +balanced_next = MapReadback.balanced_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),8n,position,b,a,depth,balanced)
  +remaining_room = UnaryLaws.rest_room(start,rest,U32.to_nat(capacity),room)
  continued = {
    %pair_round_mapped(~operation,left,right,scratch,start,slot,depth,capacity,capacity_equal,balanced,limit,capacity_ok,UnaryLaws.round_room(start,rest,U32.to_nat(capacity),room)) :
      Continuation(~operation,rest,Cells.step(~Lanes.Binary,~Tile.Tile8,~Unit,~Lanes.pair_gather(~operation),~Lanes.pair_scatter,~8,packed_state(left,right,scratch,position,slot)),_)
    continue_seen(~operation,rest,previous,
      Cells.step(~Lanes.Binary,~Tile.Tile8,~Unit,~Lanes.pair_gather(~operation),~Lanes.pair_scatter,~8,packed_state(left,right,scratch,position,slot)),
      observe_state(Cells.step(~Lanes.Binary,~Tile.Tile8,~Unit,~Lanes.pair_gather(~operation),~Lanes.pair_scatter,~8,packed_state(left,right,scratch,position,slot)))) :
    Continuation(~operation,rest,Cells.step(~Lanes.Binary,~Tile.Tile8,~Unit,~Lanes.pair_gather(~operation),~Lanes.pair_scatter,~8,packed_state(left,right,scratch,position,slot)),
      CellLaws.Observed{(Buffer.Content{mapped,Buffer.length(~F32,left)},Buffer.content(~F32,right)),next_scratch,U32.from_nat(8n+start),turn,Unit{}})}
  %Equal.sym(U32,UnaryLaws.stepped(8n,start),U32.from_nat(8n+start),UnaryLaws.advanced(8n,start)) :
    {pair_contents(Unary.rounds_with(~Lanes.Binary,~Unit,~Lanes.pair_gather(~operation),~Lanes.pair_scatter,~Lanes.pair_finish(~operation),8n+rest,packed_state(left,right,scratch,position,slot))) ==
      (Buffer.Content{MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),rest,_,b,mapped),Buffer.length(~F32,left)},Buffer.content(~F32,right)) : Buffer.Content<F32> & Buffer.Content<F32>}
  continued(8n+start,{==},depth,capacity,capacity_next,balanced_next,limit,capacity_ok,remaining_room)

def pair_more_rounds(~operation: F32 -> F32 -> F32,+rest: Nat,previous: pair_rounds_property(~operation,rest)) -> pair_rounds_property(~operation,8n+rest):
  left => right => scratch => start => slot => depth => capacity => capacity_equal => balanced => limit => capacity_ok => room =>
  pair_more_canonical(~operation,rest,previous,left,right,scratch,start,slot,depth,capacity,capacity_equal,balanced,limit,capacity_ok,room)

def pair_rounds(~operation: F32 -> F32 -> F32,count: Nat) -> pair_rounds_property(~operation,count):
  UnaryLaws.by_rounds(~pair_rounds_property(~operation),~pair_few_rounds(~operation),~pair_more_rounds(~operation),count)

def pair_map_bounded(~operation: F32 -> F32 -> F32,+count: Nat,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,+depth: Nat,+capacity: U32,
  +capacity_equal: {Tree.capacity(F32,Buffer.tree(~F32,left)) == capacity : U32},+balanced: Shape.Balanced(F32,depth,Buffer.tree(~F32,left)),
  +limit: {Nat.is_le(depth,31n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(count,U32.to_nat(capacity)) == True{} : Bool}) ->
  {pair_contents(Lanes.pair_map(~operation,count,0,Buffer.pack(~F32,left),Buffer.pack(~F32,right))) ==
    (Buffer.Content{MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,0,Buffer.tree(~F32,right),Buffer.tree(~F32,left)),Buffer.length(~F32,left)},Buffer.content(~F32,right)) : Buffer.Content<F32> & Buffer.Content<F32>}:
  %Equal.sym(Array<Tile.Tile8>,Unary.scratch(),Tree.pack(Tile.Tile8,Tree.fresh(Tile.Tile8,6n,Tile.filled(0.0))),StorageLaws.fresh(Tile.Tile8,6n,Tile.filled(0.0))) :
    {pair_contents(Unary.rounds_with(~Lanes.Binary,~Unit,~Lanes.pair_gather(~operation),~Lanes.pair_scatter,~Lanes.pair_finish(~operation),count,
      Cells.State{Lanes.Binary{Buffer.pack(~F32,left),Buffer.pack(~F32,right)},_,0,0,Unit{}})) ==
      (Buffer.Content{MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,0,Buffer.tree(~F32,right),Buffer.tree(~F32,left)),Buffer.length(~F32,left)},Buffer.content(~F32,right)) : Buffer.Content<F32> & Buffer.Content<F32>}
  pair_rounds(~operation,count)(left,right,Tree.fresh(Tile.Tile8,6n,Tile.filled(0.0)),0n,0,depth,capacity,capacity_equal,balanced,limit,capacity_ok,room)

def pair_map_zero(~operation: F32 -> F32 -> F32,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>) ->
  {pair_contents(Lanes.pair_map(~operation,0n,0,Buffer.pack(~F32,left),Buffer.pack(~F32,right))) ==
    (Buffer.Content{Buffer.tree(~F32,left),Buffer.length(~F32,left)},Buffer.content(~F32,right)) : Buffer.Content<F32> & Buffer.Content<F32>}:
  %Equal.sym(Array<Tile.Tile8>,Unary.scratch(),Tree.pack(Tile.Tile8,Tree.fresh(Tile.Tile8,6n,Tile.filled(0.0))),StorageLaws.fresh(Tile.Tile8,6n,Tile.filled(0.0))) :
    {pair_contents(Lanes.pair_finish(~operation,0n,Cells.State{Lanes.Binary{Buffer.pack(~F32,left),Buffer.pack(~F32,right)},_,0,0,Unit{}})) ==
      (Buffer.Content{Buffer.tree(~F32,left),Buffer.length(~F32,left)},Buffer.content(~F32,right)) : Buffer.Content<F32> & Buffer.Content<F32>}
  pair_tail(~operation,0n,0n,0,left,right,Tree.fresh(Tile.Tile8,6n,Tile.filled(0.0)),{==})

def pair_map(~operation: F32 -> F32 -> F32,+count: Nat,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,
  +room: {Nat.is_le(count,U32.to_nat(U32.shln(1,Buffer.depth(~F32,left)))) == True{} : Bool}) ->
  {pair_contents(Lanes.pair_map(~operation,count,0,Buffer.pack(~F32,left),Buffer.pack(~F32,right))) ==
    (Buffer.Content{MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,0,Buffer.tree(~F32,right),Buffer.tree(~F32,left)),Buffer.length(~F32,left)},Buffer.content(~F32,right)) : Buffer.Content<F32> & Buffer.Content<F32>}:
  match count:
    case 0n: pair_map_zero(~operation,left,right)
    case 1n+ +rest:
      +depth = Buffer.depth(~F32,left)
      +capacity = U32.shln(1,depth)
      +positive = Order.room_index(0n,rest,U32.to_nat(capacity),room)
      +limit = Order.strict_follow(depth,32n,Power.native_nonempty_depth(depth,positive))
      pair_map_bounded(~operation,1n+rest,left,right,depth,capacity,Shape.capacity(F32,depth,Buffer.tree(~F32,left),Buffer.balanced(~F32,left)),
        Buffer.balanced(~F32,left),limit,Power.native_power_bound(depth,limit),room)

def checked_room(+left: Buffer.Model<F32>,+left_view: Views.View,+right: Buffer.Model<F32>,+right_view: Views.View,+count: U32,
  counted: {Views.numel(left_view) == Some{count} : Maybe<&2,U32>},
  +accepted: {OperationBounds.combine_guard(~F32,Buffer.pack(~F32,left),left_view,Buffer.pack(~F32,right),right_view) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(count),U32.to_nat(U32.shln(1,Buffer.depth(~F32,left)))) == True{} : Bool}:
  Buffer.Model{+a,+al,Certificate.Certificate{+ad,+ae}} = left
  Buffer.Model{+b,+bl,Certificate.Certificate{+bd,+be}} = right
  +capacity = U32.shln(1,ad)
  +left_valid = Operations.valid_buffer(al,capacity,left_view)
  +right_valid = Operations.valid_buffer(bl,U32.shln(1,bd),right_view)
  +conditions = {[right_valid,left_valid,Views.is_contiguous(left_view),Shapes.same(Views.shape(left_view),Views.shape(right_view))] : List<&2,Bool>}
  +valid = Boolean.every_condition_holds(conditions,1n,accepted)
  +length_room = Boolean.left(Nat.is_le(al,U32.to_nat(capacity)),Views.fits(left_view,U32.from_nat(al)),valid)
  +interval = ContiguousBounds.checked_interval(left_view,count,U32.from_nat(al),Boolean.every_condition_holds(conditions,2n,accepted),counted,
    Boolean.right(Nat.is_le(al,U32.to_nat(capacity)),Views.fits(left_view,U32.from_nat(al)),valid))
  +offset = U32.to_nat(ContiguousBounds.offset(left_view))
  Order.transitive(U32.to_nat(count),al,U32.to_nat(capacity),
    %FullRange.roundtrip(al,capacity,length_room) : {Nat.is_le(U32.to_nat(count),_) == True{} : Bool}
    Order.transitive(U32.to_nat(count),Nat.add(offset,U32.to_nat(count)),U32.to_nat(U32.from_nat(al)),
      %Order.commutative(U32.to_nat(count),offset) : {Nat.is_le(U32.to_nat(count),_) == True{} : Bool}
      Order.le_add(U32.to_nat(count),offset),interval),length_room)

def scalar_fast_contents(~operation: F32 -> F32 -> F32,+count: Nat,+source: Views.View,+position: U32,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,
  fast: {Access.zero_contiguous(source) && U32.is_zero(position) == True{} : Bool}) ->
  {CloneLaws.contents(~F32,Operations.combine_kernel(~F32,~operation,count,source,position,Buffer.pack(~F32,left),Buffer.pack(~F32,right))) ==
    (Buffer.Content{MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,0,Buffer.tree(~F32,right),Buffer.tree(~F32,left)),Buffer.length(~F32,left)},Buffer.content(~F32,right)) : Buffer.Content<F32> & Buffer.Content<F32>}:
  Buffer.Model{+a,+al,+ac} = left
  Buffer.Model{+b,+bl,+bc} = right
  +model = AccessLaws.combine_model(~F32,~operation,True{},count,source,position,b,a)
  Equal.trans(Buffer.Content<F32> & Buffer.Content<F32>,
    CloneLaws.contents(~F32,Operations.combine_kernel(~F32,~operation,count,source,position,Buffer.pack(~F32,Buffer.Model{a,al,ac}),Buffer.pack(~F32,Buffer.Model{b,bl,bc}))),
    pair_model_contents(al,bl,model),(Buffer.Content{MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,0,b,a),al},Buffer.Content{b,bl}),
    %fast : {CloneLaws.contents(~F32,Operations.combine_kernel(~F32,~operation,count,source,position,Buffer.pack(~F32,Buffer.Model{a,al,ac}),Buffer.pack(~F32,Buffer.Model{b,bl,bc}))) ==
      pair_model_contents(al,bl,AccessLaws.combine_model(~F32,~operation,_,count,source,position,b,a)) : Buffer.Content<F32> & Buffer.Content<F32>}
    scalar_core(~operation,count,source,position,Buffer.Model{a,al,ac},Buffer.Model{b,bl,bc}),
    Equal.cong(AccessLaws.ModelResult<Tree.Tree<F32>,F32>,Buffer.Content<F32> & Buffer.Content<F32>,result => pair_model_contents(al,bl,result),
      model,AccessLaws.ModelResult{b,MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,0,b,a)},scalar_fast(~operation,count,source,position,b,a)))

def pair_buffers_contents(owners: Lanes.Binary) ->
  {CloneLaws.contents(~F32,Lanes.pair_buffers(owners)) == pair_contents(owners) : Buffer.Content<F32> & Buffer.Content<F32>}:
  Lanes.Binary{left,right} = owners
  {==}

def combine_when(~operation: F32 -> F32 -> F32,+fast: Bool,+count: Nat,+source: Views.View,+position: U32,+left: Buffer.Model<F32>,+right: Buffer.Model<F32>,
  equation: {Access.zero_contiguous(source) && U32.is_zero(position) == fast : Bool},
  +room: {Nat.is_le(count,U32.to_nat(U32.shln(1,Buffer.depth(~F32,left)))) == True{} : Bool}) ->
  {CloneLaws.contents(~F32,Lanes.pair_when(~operation,fast,count,source,position,Buffer.pack(~F32,left),Buffer.pack(~F32,right))) ==
    CloneLaws.contents(~F32,Operations.combine_kernel(~F32,~operation,count,source,position,Buffer.pack(~F32,left),Buffer.pack(~F32,right))) : Buffer.Content<F32> & Buffer.Content<F32>}:
  match fast:
    case False{}: {==}
    case True{}:
      %Equal.sym(Buffer.Content<F32> & Buffer.Content<F32>,
        CloneLaws.contents(~F32,Operations.combine_kernel(~F32,~operation,count,source,position,Buffer.pack(~F32,left),Buffer.pack(~F32,right))),
        (Buffer.Content{MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,0,Buffer.tree(~F32,right),Buffer.tree(~F32,left)),Buffer.length(~F32,left)},Buffer.content(~F32,right)),
        scalar_fast_contents(~operation,count,source,position,left,right,equation)) :
        {CloneLaws.contents(~F32,Lanes.pair_buffers(Lanes.pair_map(~operation,count,0,Buffer.pack(~F32,left),Buffer.pack(~F32,right)))) == _ : Buffer.Content<F32> & Buffer.Content<F32>}
      Equal.trans(Buffer.Content<F32> & Buffer.Content<F32>,
        CloneLaws.contents(~F32,Lanes.pair_buffers(Lanes.pair_map(~operation,count,0,Buffer.pack(~F32,left),Buffer.pack(~F32,right)))),
        pair_contents(Lanes.pair_map(~operation,count,0,Buffer.pack(~F32,left),Buffer.pack(~F32,right))),
        (Buffer.Content{MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~pair_value(~operation),count,0,Buffer.tree(~F32,right),Buffer.tree(~F32,left)),Buffer.length(~F32,left)},Buffer.content(~F32,right)),
        pair_buffers_contents(Lanes.pair_map(~operation,count,0,Buffer.pack(~F32,left),Buffer.pack(~F32,right))),pair_map(~operation,count,left,right,room))

# Exactly the checked entry's guard and count, with no extra owner premise.
def combine_packed(~operation: F32 -> F32 -> F32,+left: Buffer.Model<F32>,+left_view: Views.View,+right: Buffer.Model<F32>,+right_view: Views.View,+count: U32,
  counted: {Views.numel(left_view) == Some{count} : Maybe<&2,U32>},
  accepted: {OperationBounds.combine_guard(~F32,Buffer.pack(~F32,left),left_view,Buffer.pack(~F32,right),right_view) == True{} : Bool}) ->
  {CloneLaws.contents(~F32,Lanes.combine(~operation,U32.to_nat(count),right_view,ContiguousBounds.offset(left_view),Buffer.pack(~F32,left),Buffer.pack(~F32,right))) ==
    CloneLaws.contents(~F32,Operations.combine_kernel(~F32,~operation,U32.to_nat(count),right_view,ContiguousBounds.offset(left_view),Buffer.pack(~F32,left),Buffer.pack(~F32,right))) : Buffer.Content<F32> & Buffer.Content<F32>}:
  combine_when(~operation,Access.zero_contiguous(right_view) && U32.is_zero(ContiguousBounds.offset(left_view)),U32.to_nat(count),right_view,ContiguousBounds.offset(left_view),left,right,{==},
    checked_room(left,left_view,right,right_view,count,counted,accepted))

def combine_seen(~operation: F32 -> F32 -> F32,-left: Storage.Buffer<F32>,+left_view: Views.View,-right: Storage.Buffer<F32>,+right_view: Views.View,+count: U32,
  counted: {Views.numel(left_view) == Some{count} : Maybe<&2,U32>},left_seen: Buffer.Observation<F32,left>,right_seen: Buffer.Observation<F32,right>) ->
  @accepted: {OperationBounds.combine_guard(~F32,left,left_view,right,right_view) == True{} : Bool} ->
  {CloneLaws.contents(~F32,Lanes.combine(~operation,U32.to_nat(count),right_view,ContiguousBounds.offset(left_view),left,right)) ==
    CloneLaws.contents(~F32,Operations.combine_kernel(~F32,~operation,U32.to_nat(count),right_view,ContiguousBounds.offset(left_view),left,right)) : Buffer.Content<F32> & Buffer.Content<F32>}:
  Buffer.Observation{+a,left_equal} = left_seen
  Buffer.Observation{+b,right_equal} = right_seen
  %left_equal : @accepted: {OperationBounds.combine_guard(~F32,_,left_view,right,right_view) == True{} : Bool} ->
    {CloneLaws.contents(~F32,Lanes.combine(~operation,U32.to_nat(count),right_view,ContiguousBounds.offset(left_view),_,right)) ==
      CloneLaws.contents(~F32,Operations.combine_kernel(~F32,~operation,U32.to_nat(count),right_view,ContiguousBounds.offset(left_view),_,right)) : Buffer.Content<F32> & Buffer.Content<F32>}
  %right_equal : @accepted: {OperationBounds.combine_guard(~F32,Buffer.pack(~F32,a),left_view,_,right_view) == True{} : Bool} ->
    {CloneLaws.contents(~F32,Lanes.combine(~operation,U32.to_nat(count),right_view,ContiguousBounds.offset(left_view),Buffer.pack(~F32,a),_)) ==
      CloneLaws.contents(~F32,Operations.combine_kernel(~F32,~operation,U32.to_nat(count),right_view,ContiguousBounds.offset(left_view),Buffer.pack(~F32,a),_)) : Buffer.Content<F32> & Buffer.Content<F32>}
  accepted => combine_packed(~operation,a,left_view,b,right_view,count,counted,accepted)

def combine(~operation: F32 -> F32 -> F32,left: Storage.Buffer<F32>,+left_view: Views.View,right: Storage.Buffer<F32>,+right_view: Views.View,+count: U32,
  counted: {Views.numel(left_view) == Some{count} : Maybe<&2,U32>},
  accepted: {OperationBounds.combine_guard(~F32,left,left_view,right,right_view) == True{} : Bool}) ->
  {CloneLaws.contents(~F32,Lanes.combine(~operation,U32.to_nat(count),right_view,ContiguousBounds.offset(left_view),left,right)) ==
    CloneLaws.contents(~F32,Operations.combine_kernel(~F32,~operation,U32.to_nat(count),right_view,ContiguousBounds.offset(left_view),left,right)) : Buffer.Content<F32> & Buffer.Content<F32>}:
  combine_seen(~operation,left,left_view,right,right_view,count,counted,Buffer.observe(~F32,left),Buffer.observe(~F32,right))(accepted)
