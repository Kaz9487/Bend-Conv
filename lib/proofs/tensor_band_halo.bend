# Halo preparation and extraction: preserved core contents and the values of
# every allocated halo range.
import Base
import ./storage_buffer_model.bend as Buffer
import ./tensor_band_model.bend as Model
import ./tensor_band_range.bend as BandRange
import ../tensor_band.bend as Bands
import ../tensor_band_window.bend as BandWindow
import ./tensor_band_repartition.bend as BandRepartition
import ../machine_limits.bend as Machine
import ./tensor_band_dense.bend as BandDense
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ../storage_buffer.bend as Storage
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_shape.bend as Shape
import ./storage_tree_model.bend as StorageTree
import ../traversal_partition.bend as Partition

# The actual halo extractor allocates a destination and invokes the shared
# range reader. Its source remains intact and its result uses the range model.
def extract_refinement_result(~Element: Data,geometry: Bands.Geometry,fill: Element,rows: BandWindow.Rows,source: Model.Content<Element>) -> Buffer.Content<Element>:
  BandWindow.Rows{first,count} = rows
  BandRepartition.leaf_buffer(~Element,geometry,fill,first,count,source)

def packed(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+rows: BandWindow.Rows,+source: Model.Model<Element>) ->
  {BandRange.contents(~Element,BandWindow.extract(~Element,geometry,fill,rows,Model.pack(~Element,source))) ==
    BandRange.Content{Model.content(~Element,source),extract_refinement_result(~Element,geometry,fill,rows,Model.content(~Element,source))} : BandRange.Content<Element>}:
  BandWindow.Rows{+first,+count} = rows
  +request = {BandRange.Request{geometry,first,count,0} : BandRange.Request}
  +length = Bands.local_count(geometry,count)
  Equal.trans(BandRange.Content<Element>,BandRange.contents(~Element,BandWindow.extract(~Element,geometry,fill,rows,Model.pack(~Element,source))),
    BandRange.written(~Element,request,Model.content(~Element,source),Buffer.contents(~Element,Bands.allocate(~Element,fill,length))),
    BandRange.Content{Model.content(~Element,source),extract_refinement_result(~Element,geometry,fill,rows,Model.content(~Element,source))},
    BandRange.native_range(~Element,request,source,Bands.allocate(~Element,fill,length)),
    Equal.cong(Buffer.Content<Element>,BandRange.Content<Element>,output => BandRange.written(~Element,request,Model.content(~Element,source),output),
      Buffer.contents(~Element,Bands.allocate(~Element,fill,length)),Buffer.Content{StorageTree.fresh(Element,Storage.depth_for(length),fill),U32.to_nat(length)},
      AllocationLaws.contents(~Element,Storage.depth_for(length),fill,U32.to_nat(length))))

type ExtractRefinementContext<-Element: Data> is Data:
  ExtractRefinementContext{geometry: Bands.Geometry,fill: Element,rows: BandWindow.Rows}

def extract_refinement_proposition(~Element: Data,context: ExtractRefinementContext<Element>,-source: Bands.Bands(~Element)) -> Type:
  ExtractRefinementContext{+geometry,+fill,+rows} = context
  {BandRange.contents(~Element,BandWindow.extract(~Element,geometry,fill,rows,source)) ==
    BandRange.Content{Model.contents(~Element,source),extract_refinement_result(~Element,geometry,fill,rows,Model.contents(~Element,source))} : BandRange.Content<Element>}

def extract(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+rows: BandWindow.Rows,source: Bands.Bands(~Element)) ->
  {BandRange.contents(~Element,BandWindow.extract(~Element,geometry,fill,rows,source)) ==
    BandRange.Content{Model.contents(~Element,source),extract_refinement_result(~Element,geometry,fill,rows,Model.contents(~Element,source))} : BandRange.Content<Element>}:
  Model.lift(~Element,~ExtractRefinementContext<Element>,~extract_refinement_proposition(~Element),ExtractRefinementContext{geometry,fill,rows},source,Model.observe(~Element,source),model =>
    +model = model
    %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,model)),Model.content(~Element,model),Model.packed_contents(~Element,model)) :
      {BandRange.contents(~Element,BandWindow.extract(~Element,geometry,fill,rows,Model.pack(~Element,model))) == BandRange.Content{_,extract_refinement_result(~Element,geometry,fill,rows,_)} : BandRange.Content<Element>}
    packed(~Element,geometry,fill,rows,model))

# The real preparation pass preserves core contents and constructs exactly
# the requested upper/lower buffers. No kernel or reduction is modeled here.
type Tree<-Element: Data> is Data:
  Leaf{above: Buffer.Content<Element>,below: Buffer.Content<Element>,window: BandWindow.RowWindow,first: U32,rows: U32}
  Fork{left: Tree<Element>,right: Tree<Element>}

# Canonical halo owners retain the actual Buffer certificates, just as the
# core's Model does. Only refinement proofs use this representation.
type Owners<-Element: Data> is Data:
  OwnedLeaf{above: Buffer.Model<Element>,below: Buffer.Model<Element>,window: BandWindow.RowWindow,first: U32,rows: U32}
  OwnedFork{left: Owners<Element>,right: Owners<Element>}

def pack_owners(~Element: Data,owners: Owners<Element>) -> BandWindow.Halos(~Element):
  match owners:
    case OwnedLeaf{above,below,window,first,rows}:
      Partition.Leaf{BandWindow.Halo{Buffer.pack(~Element,above),Buffer.pack(~Element,below),window},first,rows,Unit{}}
    case OwnedFork{left,right}: Partition.Fork{pack_owners(~Element,left),pack_owners(~Element,right)}

def owner_contents(~Element: Data,owners: Owners<Element>) -> Tree<Element>:
  match owners:
    case OwnedLeaf{above,below,window,first,rows}: Leaf{Buffer.content(~Element,above),Buffer.content(~Element,below),window,first,rows}
    case OwnedFork{left,right}: Fork{owner_contents(~Element,left),owner_contents(~Element,right)}

type OwnerObservation<-Element: Data,-halos: BandWindow.Halos(~Element)> is Data:
  OwnerObservation{owners: Owners<Element>,equation: {pack_owners(~Element,owners) == halos : BandWindow.Halos(~Element)}}

def observe_leaf(~Element: Data,+window: BandWindow.RowWindow,+first: U32,+rows: U32,-above: Storage.Buffer<Element>,-below: Storage.Buffer<Element>,
  above_observation: Buffer.Observation<Element,above>,below_observation: Buffer.Observation<Element,below>) ->
  OwnerObservation<Element,Partition.Leaf{BandWindow.Halo{above,below,window},first,rows,Unit{}}>:
  Buffer.Observation{+above_model,above_equal} = above_observation
  Buffer.Observation{+below_model,below_equal} = below_observation
  OwnerObservation{OwnedLeaf{above_model,below_model,window,first,rows},
    %above_equal : {Partition.Leaf{BandWindow.Halo{Buffer.pack(~Element,above_model),Buffer.pack(~Element,below_model),window},first,rows,Unit{}} ==
      Partition.Leaf{BandWindow.Halo{_,below,window},first,rows,Unit{}} : BandWindow.Halos(~Element)}
    %below_equal : {Partition.Leaf{BandWindow.Halo{Buffer.pack(~Element,above_model),Buffer.pack(~Element,below_model),window},first,rows,Unit{}} ==
      Partition.Leaf{BandWindow.Halo{Buffer.pack(~Element,above_model),_,window},first,rows,Unit{}} : BandWindow.Halos(~Element)}
    {==}}

def observe_fork(~Element: Data,-left: BandWindow.Halos(~Element),-right: BandWindow.Halos(~Element),
  left_observation: OwnerObservation<Element,left>,right_observation: OwnerObservation<Element,right>) -> OwnerObservation<Element,Partition.Fork{left,right}>:
  OwnerObservation{+left_owners,left_equal} = left_observation
  OwnerObservation{+right_owners,right_equal} = right_observation
  OwnerObservation{OwnedFork{left_owners,right_owners},
    %left_equal : {Partition.Fork{pack_owners(~Element,left_owners),pack_owners(~Element,right_owners)} == Partition.Fork{_,right} : BandWindow.Halos(~Element)}
    %right_equal : {Partition.Fork{pack_owners(~Element,left_owners),pack_owners(~Element,right_owners)} == Partition.Fork{pack_owners(~Element,left_owners),_} : BandWindow.Halos(~Element)}
    {==}}

def observe_owners(~Element: Data,halos: BandWindow.Halos(~Element)) -> OwnerObservation<Element,halos>:
  match halos:
    case Partition.Leaf{BandWindow.Halo{above,below,+window},+first,+rows,Unit{}}:
      observe_leaf(~Element,window,first,rows,above,below,Buffer.observe(~Element,above),Buffer.observe(~Element,below))
    case Partition.Fork{left,right}: observe_fork(~Element,left,right,observe_owners(~Element,left),observe_owners(~Element,right))

type Content<-Element: Data> is Data:
  Content{source: Model.Content<Element>,halos: Tree<Element>}

def reflect(~Element: Data,halos: BandWindow.Halos(~Element)) -> Tree<Element>:
  match halos:
    case Partition.Leaf{BandWindow.Halo{above,below,window},first,rows,details}:
      Leaf{Buffer.contents(~Element,above),Buffer.contents(~Element,below),window,first,rows}
    case Partition.Fork{left,right}: Fork{reflect(~Element,left),reflect(~Element,right)}

def packed_owners(~Element: Data,+owners: Owners<Element>) ->
  {reflect(~Element,pack_owners(~Element,owners)) == owner_contents(~Element,owners) : Tree<Element>}:
  match owners:
    case OwnedLeaf{+above,+below,+window,+first,+rows}:
      %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,above)),Buffer.content(~Element,above),Buffer.packed_contents(~Element,above)) :
        {Leaf{_,Buffer.contents(~Element,Buffer.pack(~Element,below)),window,first,rows} == Leaf{Buffer.content(~Element,above),Buffer.content(~Element,below),window,first,rows} : Tree<Element>}
      %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,below)),Buffer.content(~Element,below),Buffer.packed_contents(~Element,below)) :
        {Leaf{Buffer.content(~Element,above),_,window,first,rows} == Leaf{Buffer.content(~Element,above),Buffer.content(~Element,below),window,first,rows} : Tree<Element>}
      {==}
    case OwnedFork{+left,+right}:
      %Equal.sym(Tree<Element>,reflect(~Element,pack_owners(~Element,left)),owner_contents(~Element,left),packed_owners(~Element,left)) :
        {Fork{_,reflect(~Element,pack_owners(~Element,right))} == Fork{owner_contents(~Element,left),owner_contents(~Element,right)} : Tree<Element>}
      %Equal.sym(Tree<Element>,reflect(~Element,pack_owners(~Element,right)),owner_contents(~Element,right),packed_owners(~Element,right)) :
        {Fork{owner_contents(~Element,left),_} == Fork{owner_contents(~Element,left),owner_contents(~Element,right)} : Tree<Element>}
      {==}

def refinement_contents(~Element: Data,owners: Bands.Bands(~Element) & BandWindow.Halos(~Element)) -> Content<Element>:
  (source,halos) = owners
  Content{Model.contents(~Element,source),reflect(~Element,halos)}

def leaf_model(~Element: Data,+geometry: Bands.Geometry,+fill: Element,first: U32,rows: U32,+window: BandWindow.RowWindow,+source: Model.Content<Element>) -> Tree<Element>:
  BandWindow.RowWindow{output,+above,+below,read} = window
  Leaf{extract_refinement_result(~Element,geometry,fill,above,source),extract_refinement_result(~Element,geometry,fill,below,source),window,first,rows}

def completed(~Element: Data,first: U32,rows: U32,window: BandWindow.RowWindow,above: Buffer.Content<Element>,lower: BandRange.Content<Element>) -> Content<Element>:
  BandRange.Content{source,below} = lower
  Content{source,Leaf{above,below,window,first,rows}}

def finish(~Element: Data,+first: U32,+rows: U32,+window: BandWindow.RowWindow,above: Storage.Buffer<Element>,lower: Bands.Bands(~Element) & Storage.Buffer<Element>) -> Bands.Bands(~Element) & BandWindow.Halos(~Element):
  Storage.bind_pair(~Bands.Bands(~Element),~Storage.Buffer<Element>,~(Bands.Bands(~Element) & BandWindow.Halos(~Element)),lower,
    source => below => (source,Partition.Leaf{BandWindow.Halo{above,below,window},first,rows,Unit{}}))

def finished(~Element: Data,+first: U32,+rows: U32,+window: BandWindow.RowWindow,-above: Storage.Buffer<Element>,lower: Bands.Bands(~Element) & Storage.Buffer<Element>) ->
  {refinement_contents(~Element,finish(~Element,first,rows,window,above,lower)) == completed(~Element,first,rows,window,Buffer.contents(~Element,above),BandRange.contents(~Element,lower)) : Content<Element>}:
  (source,below) = lower
  {==}

def after_upper(~Element: Data,geometry: Bands.Geometry,fill: Element,first: U32,rows: U32,+window: BandWindow.RowWindow,upper: BandRange.Content<Element>) -> Content<Element>:
  BandWindow.RowWindow{output,above,+below,read} = window
  BandRange.Content{+source,upper} = upper
  Content{source,Leaf{upper,extract_refinement_result(~Element,geometry,fill,below,source),window,first,rows}}

def below(window: BandWindow.RowWindow) -> BandWindow.Rows:
  BandWindow.RowWindow{output,above,below,read} = window
  below

def lower_packed(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+window: BandWindow.RowWindow,+source_model: Model.Model<Element>,-above: Storage.Buffer<Element>) ->
  {refinement_contents(~Element,finish(~Element,first,rows,window,above,BandWindow.extract(~Element,geometry,fill,below(window),Model.pack(~Element,source_model)))) ==
    after_upper(~Element,geometry,fill,first,rows,window,BandRange.Content{Model.contents(~Element,Model.pack(~Element,source_model)),Buffer.contents(~Element,above)}) : Content<Element>}:
  BandWindow.RowWindow{output,upper_rows,below_rows,read} = window
  -source = Model.pack(~Element,source_model)
  Equal.trans(Content<Element>,refinement_contents(~Element,finish(~Element,first,rows,window,above,BandWindow.extract(~Element,geometry,fill,below_rows,source))),
    completed(~Element,first,rows,window,Buffer.contents(~Element,above),BandRange.contents(~Element,BandWindow.extract(~Element,geometry,fill,below_rows,source))),
    after_upper(~Element,geometry,fill,first,rows,window,BandRange.Content{Model.contents(~Element,source),Buffer.contents(~Element,above)}),
    finished(~Element,first,rows,window,above,BandWindow.extract(~Element,geometry,fill,below_rows,Model.pack(~Element,source_model))),
    Equal.cong(BandRange.Content<Element>,Content<Element>,result => completed(~Element,first,rows,window,Buffer.contents(~Element,above),result),
      BandRange.contents(~Element,BandWindow.extract(~Element,geometry,fill,below_rows,source)),
      BandRange.Content{Model.contents(~Element,source),extract_refinement_result(~Element,geometry,fill,below_rows,Model.contents(~Element,source))},extract(~Element,geometry,fill,below_rows,Model.pack(~Element,source_model))))

def lower_observed(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+window: BandWindow.RowWindow,-source: Bands.Bands(~Element),-above: Storage.Buffer<Element>,observation: Model.Observation<Element,source>) ->
  {refinement_contents(~Element,finish(~Element,first,rows,window,above,BandWindow.extract(~Element,geometry,fill,below(window),source))) ==
    after_upper(~Element,geometry,fill,first,rows,window,BandRange.Content{Model.contents(~Element,source),Buffer.contents(~Element,above)}) : Content<Element>}:
  Model.Observation{source_model,equation} = observation
  %equation : {refinement_contents(~Element,finish(~Element,first,rows,window,above,BandWindow.extract(~Element,geometry,fill,below(window),_))) ==
    after_upper(~Element,geometry,fill,first,rows,window,BandRange.Content{Model.contents(~Element,_),Buffer.contents(~Element,above)}) : Content<Element>}
  lower_packed(~Element,geometry,fill,first,rows,window,source_model,above)

def continued(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+window: BandWindow.RowWindow,upper: Bands.Bands(~Element) & Storage.Buffer<Element>) -> Bands.Bands(~Element) & BandWindow.Halos(~Element):
  Storage.bind_pair(~Bands.Bands(~Element),~Storage.Buffer<Element>,~(Bands.Bands(~Element) & BandWindow.Halos(~Element)),upper,
    source => above => finish(~Element,first,rows,window,above,BandWindow.extract(~Element,geometry,fill,below(window),source)))

def continuation(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+window: BandWindow.RowWindow,upper: Bands.Bands(~Element) & Storage.Buffer<Element>) ->
  {refinement_contents(~Element,continued(~Element,geometry,fill,first,rows,window,upper)) == after_upper(~Element,geometry,fill,first,rows,window,BandRange.contents(~Element,upper)) : Content<Element>}:
  (source,above) = upper
  lower_observed(~Element,geometry,fill,first,rows,window,source,above,Model.observe(~Element,source))

def halo_packed(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+window: BandWindow.RowWindow,+source_model: Model.Model<Element>) ->
  {refinement_contents(~Element,BandWindow.halo(~Element,geometry,fill,first,rows,window,Model.pack(~Element,source_model))) ==
    Content{Model.contents(~Element,Model.pack(~Element,source_model)),leaf_model(~Element,geometry,fill,first,rows,window,Model.contents(~Element,Model.pack(~Element,source_model)))} : Content<Element>}:
  BandWindow.RowWindow{output,above,below,read} = window
  -source = Model.pack(~Element,source_model)
  Equal.trans(Content<Element>,refinement_contents(~Element,BandWindow.halo(~Element,geometry,fill,first,rows,window,source)),
    after_upper(~Element,geometry,fill,first,rows,window,BandRange.contents(~Element,BandWindow.extract(~Element,geometry,fill,above,source))),
    Content{Model.contents(~Element,source),leaf_model(~Element,geometry,fill,first,rows,window,Model.contents(~Element,source))},
    continuation(~Element,geometry,fill,first,rows,window,BandWindow.extract(~Element,geometry,fill,above,Model.pack(~Element,source_model))),
    Equal.cong(BandRange.Content<Element>,Content<Element>,result => after_upper(~Element,geometry,fill,first,rows,window,result),
      BandRange.contents(~Element,BandWindow.extract(~Element,geometry,fill,above,source)),
      BandRange.Content{Model.contents(~Element,source),extract_refinement_result(~Element,geometry,fill,above,Model.contents(~Element,source))},extract(~Element,geometry,fill,above,Model.pack(~Element,source_model))))

type LeafContext<-Element: Data> is Data:
  LeafContext{geometry: Bands.Geometry,fill: Element,first: U32,rows: U32,window: BandWindow.RowWindow}

def leaf_proposition(~Element: Data,context: LeafContext<Element>,-source: Bands.Bands(~Element)) -> Type:
  LeafContext{+geometry,+fill,+first,+rows,+window} = context
  {refinement_contents(~Element,BandWindow.halo(~Element,geometry,fill,first,rows,window,source)) ==
    Content{Model.contents(~Element,source),leaf_model(~Element,geometry,fill,first,rows,window,Model.contents(~Element,source))} : Content<Element>}

def halo(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+window: BandWindow.RowWindow,source: Bands.Bands(~Element)) ->
  {refinement_contents(~Element,BandWindow.halo(~Element,geometry,fill,first,rows,window,source)) ==
    Content{Model.contents(~Element,source),leaf_model(~Element,geometry,fill,first,rows,window,Model.contents(~Element,source))} : Content<Element>}:
  Model.lift(~Element,~LeafContext<Element>,~leaf_proposition(~Element),LeafContext{geometry,fill,first,rows,window},source,Model.observe(~Element,source),model => halo_packed(~Element,geometry,fill,first,rows,window,model))

def model(~Element: Data,+geometry: BandWindow.Geometry,+fill: Element,layout: Bands.Plan,+source: Model.Content<Element>) -> Tree<Element>:
  match layout:
    case Bands.Rows{+first,+rows}: leaf_model(~Element,BandWindow.input_geometry(geometry),fill,first,rows,BandWindow.window(geometry,first,rows),source)
    case Bands.Split{left,right}: Fork{model(~Element,geometry,fill,left,source),model(~Element,geometry,fill,right,source)}

def joined(~Element: Data,left: Tree<Element>,right: Content<Element>) -> Content<Element>:
  Content{source,right} = right
  Content{source,Fork{left,right}}

def join_native(~Element: Data,left: BandWindow.Halos(~Element),right: Bands.Bands(~Element) & BandWindow.Halos(~Element)) -> Bands.Bands(~Element) & BandWindow.Halos(~Element):
  Storage.bind_pair(~Bands.Bands(~Element),~BandWindow.Halos(~Element),~(Bands.Bands(~Element) & BandWindow.Halos(~Element)),right,
    source => right => (source,Partition.Fork{left,right}))

def join_contents(~Element: Data,-left: BandWindow.Halos(~Element),right: Bands.Bands(~Element) & BandWindow.Halos(~Element)) ->
  {refinement_contents(~Element,join_native(~Element,left,right)) == joined(~Element,reflect(~Element,left),refinement_contents(~Element,right)) : Content<Element>}:
  (source,right) = right
  {==}

def after_left(~Element: Data,geometry: BandWindow.Geometry,fill: Element,layout: Bands.Plan,left: Content<Element>) -> Content<Element>:
  Content{+source,left} = left
  Content{source,Fork{left,model(~Element,geometry,fill,layout,source)}}

def continue_tree(~Element: Data,+geometry: BandWindow.Geometry,+fill: Element,+layout: Bands.Plan,left: Bands.Bands(~Element) & BandWindow.Halos(~Element)) -> Bands.Bands(~Element) & BandWindow.Halos(~Element):
  Storage.bind_pair(~Bands.Bands(~Element),~BandWindow.Halos(~Element),~(Bands.Bands(~Element) & BandWindow.Halos(~Element)),left,
    source => left => join_native(~Element,left,BandWindow.prepare(~Element,geometry,fill,layout,source)))

def continue_packed(~Element: Data,+geometry: BandWindow.Geometry,+fill: Element,+layout: Bands.Plan,+source_model: Model.Model<Element>,-left: BandWindow.Halos(~Element),
  rule: @source: Bands.Bands(~Element) -> {refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,layout,source)) == Content{Model.contents(~Element,source),model(~Element,geometry,fill,layout,Model.contents(~Element,source))} : Content<Element>}) ->
  {refinement_contents(~Element,continue_tree(~Element,geometry,fill,layout,(Model.pack(~Element,source_model),left))) == after_left(~Element,geometry,fill,layout,refinement_contents(~Element,(Model.pack(~Element,source_model),left))) : Content<Element>}:
  -source = Model.pack(~Element,source_model)
  Equal.trans(Content<Element>,refinement_contents(~Element,join_native(~Element,left,BandWindow.prepare(~Element,geometry,fill,layout,source))),
    joined(~Element,reflect(~Element,left),refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,layout,source))),
    after_left(~Element,geometry,fill,layout,refinement_contents(~Element,(source,left))),
    join_contents(~Element,left,BandWindow.prepare(~Element,geometry,fill,layout,Model.pack(~Element,source_model))),
    Equal.cong(Content<Element>,Content<Element>,result => joined(~Element,reflect(~Element,left),result),
      refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,layout,source)),Content{Model.contents(~Element,source),model(~Element,geometry,fill,layout,Model.contents(~Element,source))},rule(Model.pack(~Element,source_model))))

def continue_observed(~Element: Data,+geometry: BandWindow.Geometry,+fill: Element,+layout: Bands.Plan,-source: Bands.Bands(~Element),-left: BandWindow.Halos(~Element),observation: Model.Observation<Element,source>,
  rule: @source: Bands.Bands(~Element) -> {refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,layout,source)) == Content{Model.contents(~Element,source),model(~Element,geometry,fill,layout,Model.contents(~Element,source))} : Content<Element>}) ->
  {refinement_contents(~Element,continue_tree(~Element,geometry,fill,layout,(source,left))) == after_left(~Element,geometry,fill,layout,refinement_contents(~Element,(source,left))) : Content<Element>}:
  Model.Observation{source_model,equation} = observation
  %equation : {refinement_contents(~Element,continue_tree(~Element,geometry,fill,layout,(_,left))) == after_left(~Element,geometry,fill,layout,refinement_contents(~Element,(_,left))) : Content<Element>}
  continue_packed(~Element,geometry,fill,layout,source_model,left,rule)

def continue_contents(~Element: Data,+geometry: BandWindow.Geometry,+fill: Element,+layout: Bands.Plan,left: Bands.Bands(~Element) & BandWindow.Halos(~Element),
  rule: @source: Bands.Bands(~Element) -> {refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,layout,source)) == Content{Model.contents(~Element,source),model(~Element,geometry,fill,layout,Model.contents(~Element,source))} : Content<Element>}) ->
  {refinement_contents(~Element,continue_tree(~Element,geometry,fill,layout,left)) == after_left(~Element,geometry,fill,layout,refinement_contents(~Element,left)) : Content<Element>}:
  (source,left) = left
  continue_observed(~Element,geometry,fill,layout,source,left,Model.observe(~Element,source),rule)

type RefinementContext<-Element: Data> is Data:
  RefinementContext{geometry: BandWindow.Geometry,fill: Element,layout: Bands.Plan}

def refinement_proposition(~Element: Data,context: RefinementContext<Element>,-source: Bands.Bands(~Element)) -> Type:
  RefinementContext{+geometry,+fill,+layout} = context
  {refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,layout,source)) == Content{Model.contents(~Element,source),model(~Element,geometry,fill,layout,Model.contents(~Element,source))} : Content<Element>}

def prepare(~Element: Data,+geometry: BandWindow.Geometry,+fill: Element,layout: Bands.Plan,source: Bands.Bands(~Element)) ->
  {refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,layout,source)) == Content{Model.contents(~Element,source),model(~Element,geometry,fill,layout,Model.contents(~Element,source))} : Content<Element>}:
  match layout:
    case Bands.Rows{+first,+rows}: halo(~Element,BandWindow.input_geometry(geometry),fill,first,rows,BandWindow.window(geometry,first,rows),source)
    case Bands.Split{+left,+right}:
      Model.lift(~Element,~RefinementContext<Element>,~refinement_proposition(~Element),RefinementContext{geometry,fill,layout},source,Model.observe(~Element,source),source_model =>
        +source_model = source_model
        -source = Model.pack(~Element,source_model)
        Equal.trans(Content<Element>,refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,layout,source)),
          after_left(~Element,geometry,fill,right,refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,left,source))),
          Content{Model.contents(~Element,source),model(~Element,geometry,fill,layout,Model.contents(~Element,source))},
          continue_contents(~Element,geometry,fill,right,BandWindow.prepare(~Element,geometry,fill,left,Model.pack(~Element,source_model)),input => prepare(~Element,geometry,fill,right,input)),
          Equal.cong(Content<Element>,Content<Element>,result => after_left(~Element,geometry,fill,right,result),
            refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,left,source)),Content{Model.contents(~Element,source),model(~Element,geometry,fill,left,Model.contents(~Element,source))},prepare(~Element,geometry,fill,left,Model.pack(~Element,source_model)))))

# Semantics of the real halo preparation result. Window span obligations are
# explicit; the proof constructs all buffer readback facts from source Values.
def RangeFacts(geometry: Bands.Geometry,rows: BandWindow.Rows,capacity: U32,division_capacity: U32) -> Data:
  BandWindow.Rows{first,count} = rows
  BandDense.LeafFacts<geometry,first,count,capacity,division_capacity>

def LeafFacts(+geometry: Bands.Geometry,window: BandWindow.RowWindow,+capacity: U32,+division_capacity: U32) -> Data:
  BandWindow.RowWindow{output,above,below,read} = window
  Shape.Both<RangeFacts(geometry,above,capacity,division_capacity),RangeFacts(geometry,below,capacity,division_capacity)>

def Fits(+geometry: BandWindow.Geometry,layout: Bands.Plan,+capacity: U32,+division_capacity: U32) -> Data:
  match layout:
    case Bands.Rows{first,rows}: LeafFacts(BandWindow.input_geometry(geometry),BandWindow.window(geometry,first,rows),capacity,division_capacity)
    case Bands.Split{left,right}: Shape.Both<Fits(geometry,left,capacity,division_capacity),Fits(geometry,right,capacity,division_capacity)>

def RangeValues(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,rows: BandWindow.Rows,+buffer: Buffer.Content<Element>) -> Data:
  BandWindow.Rows{+first,+count} = rows
  Buffer.Content{tree,length} = buffer
  Shape.Both<Buffer.ContentReady<Element,buffer>,Shape.Both<BandShape.Leaf(geometry,first,count,length),
    BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.RowBand{buffer,first,count})>>

def TreeValues(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,tree: Tree<Element>) -> Data:
  match tree:
    case Leaf{above,below,BandWindow.RowWindow{output,above_rows,below_rows,read},first,rows}:
      Shape.Both<RangeValues(~Element,~Logical,~original,logical,geometry,above_rows,above),RangeValues(~Element,~Logical,~original,logical,geometry,below_rows,below)>
    case Fork{left,right}:
      Shape.Both<TreeValues(~Element,~Logical,~original,logical,geometry,left),TreeValues(~Element,~Logical,~original,logical,geometry,right)>

def range(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+fill: Element,+rows: BandWindow.Rows,+source: Model.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,
  shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source)),
  +target: RangeFacts(geometry,rows,capacity,Machine.half_word(limits))) ->
  RangeValues(~Element,~Logical,~original,logical,geometry,rows,extract_refinement_result(~Element,geometry,fill,rows,Model.content(~Element,source))):
  BandWindow.Rows{+first,+count} = rows
  BandDense.LeafFacts{allocation,contents,metadata} = target
  Shape.Both{BandRepartition.leaf_ready(~Element,geometry,fill,first,count,Model.content(~Element,source),capacity,Machine.half_word(limits),target),
    Shape.Both{metadata,BandRepartition.modeled_values(~Element,~Logical,~original,logical,geometry,fill,Bands.Rows{first,count},source,limits,capacity,domain,shape,values,target)}}

def leaf(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+window: BandWindow.RowWindow,+source: Model.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,
  +shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,+values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source)),
  target: LeafFacts(geometry,window,capacity,Machine.half_word(limits))) ->
  TreeValues(~Element,~Logical,~original,logical,geometry,leaf_model(~Element,geometry,fill,first,rows,window,Model.content(~Element,source))):
  BandWindow.RowWindow{output,+above,+below,read} = window
  Shape.Both{above_facts,below_facts} = target
  Shape.Both{range(~Element,~Logical,~original,logical,geometry,fill,above,source,limits,capacity,domain,shape,values,above_facts),
    range(~Element,~Logical,~original,logical,geometry,fill,below,source,limits,capacity,domain,shape,values,below_facts)}

def modeled(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: BandWindow.Geometry,+fill: Element,layout: Bands.Plan,+source: Model.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<BandWindow.input_geometry(geometry),limits,capacity>,
  +shape: BandShape.WellFormed<Element,BandWindow.input_geometry(geometry),Model.content(~Element,source)>,+values: BandValues.Values(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),Model.content(~Element,source)),
  target: Fits(geometry,layout,capacity,Machine.half_word(limits))) ->
  TreeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),model(~Element,geometry,fill,layout,Model.content(~Element,source))):
  match layout:
    case Bands.Rows{+first,+rows}: leaf(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),fill,first,rows,BandWindow.window(geometry,first,rows),source,limits,capacity,domain,shape,values,target)
    case Bands.Split{+left,+right}:
      Shape.Both{left_facts,right_facts} = target
      Shape.Both{modeled(~Element,~Logical,~original,logical,geometry,fill,left,source,limits,capacity,domain,shape,values,left_facts),
        modeled(~Element,~Logical,~original,logical,geometry,fill,right,source,limits,capacity,domain,shape,values,right_facts)}

def Prepared(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,content: Content<Element>) -> Data:
  Content{+source,halos} = content
  Shape.Both<BandShape.WellFormed<Element,geometry,source>,Shape.Both<BandValues.Values(~Element,~Logical,~original,logical,geometry,source),TreeValues(~Element,~Logical,~original,logical,geometry,halos)>>

def prepared(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: BandWindow.Geometry,+fill: Element,+layout: Bands.Plan,+source: Model.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<BandWindow.input_geometry(geometry),limits,capacity>,
  +shape: BandShape.WellFormed<Element,BandWindow.input_geometry(geometry),Model.content(~Element,source)>,+values: BandValues.Values(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),Model.content(~Element,source)),
  target: Fits(geometry,layout,capacity,Machine.half_word(limits))) ->
  Prepared(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,layout,Model.pack(~Element,source)))):
  %Equal.sym(Content<Element>,refinement_contents(~Element,BandWindow.prepare(~Element,geometry,fill,layout,Model.pack(~Element,source))),
    Content{Model.contents(~Element,Model.pack(~Element,source)),model(~Element,geometry,fill,layout,Model.contents(~Element,Model.pack(~Element,source)))},prepare(~Element,geometry,fill,layout,Model.pack(~Element,source))) :
    Prepared(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),_)
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,source)),Model.content(~Element,source),Model.packed_contents(~Element,source)) :
    Prepared(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),Content{_,model(~Element,geometry,fill,layout,_)})
  Shape.Both{shape,Shape.Both{values,modeled(~Element,~Logical,~original,logical,geometry,fill,layout,source,limits,capacity,domain,shape,values,target)}}

# Readiness of an actual allocated halo range. Window geometry supplies LeafFacts;
# producing those facts for above and below remains the window caller's obligation.
def ready(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+source: Model.Model<Element>,+capacity: U32,+division_capacity: U32,
  target: BandDense.LeafFacts<geometry,first,rows,capacity,division_capacity>) ->
  Buffer.ContentReady<Element,BandRange.result(~Element,BandRange.contents(~Element,BandWindow.extract(~Element,geometry,fill,BandWindow.Rows{first,rows},Model.pack(~Element,source))))>:
  %Equal.sym(BandRange.Content<Element>,BandRange.contents(~Element,BandWindow.extract(~Element,geometry,fill,BandWindow.Rows{first,rows},Model.pack(~Element,source))),
    BandRange.Content{Model.content(~Element,source),extract_refinement_result(~Element,geometry,fill,BandWindow.Rows{first,rows},Model.content(~Element,source))},packed(~Element,geometry,fill,BandWindow.Rows{first,rows},source)) : Buffer.ContentReady<Element,BandRange.result(~Element,_)>
  BandRepartition.leaf_ready(~Element,geometry,fill,first,rows,Model.content(~Element,source),capacity,division_capacity,target)
