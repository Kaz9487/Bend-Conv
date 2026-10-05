# The shared band recursion, by structure. A parallel split and a leaf task
# compute what the sequential recursion computes; every band operation uses
# these equations instead of unfolding the recursion itself.
import Base
import ../tensor_band.bend as Bands
import ../traversal_partition.bend as Partition
import ../storage_buffer.bend as Storage
import ./tensor_band_model.bend as Model
import ./storage_buffer_model.bend as Buffer

def Tree(~Block: Type) -> Type:
  Partition.Tree<&1,Block,Unit>

# Both halves of a split, each with its own resource.
def forked(~Block: Type,~Output: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Block -> Resource -> Tree(~Output),
  ~keep: Tree(~Block) -> Tree(~Output),
  +left_plan: Bands.Plan,+right_plan: Bands.Plan,+enabled: Bool,+geometry: Bands.Geometry,+context: Context,
  left: Tree(~Block),right: Tree(~Block),resources: Resource & Resource) -> Tree(~Output):
  Storage.bind_pair(~Resource,~Resource,~Tree(~Output),resources,left_resource => right_resource =>
    Partition.Fork{Bands.each(~Block,~Output,~Resource,~Context,~share,~leaf,~keep,left_plan,enabled && Bands.same_image(geometry,left_plan),enabled,geometry,context,left,left_resource),
      Bands.each(~Block,~Output,~Resource,~Context,~share,~leaf,~keep,right_plan,enabled && Bands.same_image(geometry,right_plan),enabled,geometry,context,right,right_resource)})

# A split plan on a fork: both halves, each on its own plan. The parallel
# split and its leaf tasks equal the sequential recursion.
def fork(~Block: Type,~Output: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Block -> Resource -> Tree(~Output),
  ~keep: Tree(~Block) -> Tree(~Output),
  +left_plan: Bands.Plan,+right_plan: Bands.Plan,+mode: Bool,+enabled: Bool,+geometry: Bands.Geometry,+context: Context,
  -left: Tree(~Block),-right: Tree(~Block),-resource: Resource) ->
  {Bands.each(~Block,~Output,~Resource,~Context,~share,~leaf,~keep,Bands.Split{left_plan,right_plan},mode,enabled,geometry,context,Partition.Fork{left,right},resource) ==
    forked(~Block,~Output,~Resource,~Context,~share,~leaf,~keep,left_plan,right_plan,enabled,geometry,context,left,right,share(resource)) : Tree(~Output)}:
  match left_plan right_plan mode:
    case Bands.Rows{left_first,left_rows} Bands.Rows{right_first,right_rows} False{}: {==}
    case Bands.Rows{left_first,left_rows} Bands.Rows{right_first,right_rows} True{}: {==}
    case Bands.Rows{left_first,left_rows} Bands.Split{right_left,right_right} False{}: {==}
    case Bands.Rows{left_first,left_rows} Bands.Split{right_left,right_right} True{}: {==}
    case Bands.Split{left_left,left_right} Bands.Rows{right_first,right_rows} False{}: {==}
    case Bands.Split{left_left,left_right} Bands.Rows{right_first,right_rows} True{}: {==}
    case Bands.Split{left_left,left_right} Bands.Split{right_left,right_right} False{}: {==}
    case Bands.Split{left_left,left_right} Bands.Split{right_left,right_right} True{}: {==}

# A split plan on one leaf keeps the leaf.
def split_leaf(~Block: Type,~Output: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Block -> Resource -> Tree(~Output),
  ~keep: Tree(~Block) -> Tree(~Output),
  +left_plan: Bands.Plan,+right_plan: Bands.Plan,+mode: Bool,+enabled: Bool,+geometry: Bands.Geometry,+context: Context,
  -block: Block,-first: U32,-rows: U32,-details: Unit,-resource: Resource) ->
  {Bands.each(~Block,~Output,~Resource,~Context,~share,~leaf,~keep,Bands.Split{left_plan,right_plan},mode,enabled,geometry,context,Partition.Leaf{block,first,rows,details},resource) ==
    keep(Partition.Leaf{block,first,rows,details}) : Tree(~Output)}:
  match left_plan right_plan mode:
    case Bands.Rows{left_first,left_rows} Bands.Rows{right_first,right_rows} False{}: {==}
    case Bands.Rows{left_first,left_rows} Bands.Rows{right_first,right_rows} True{}: {==}
    case Bands.Rows{left_first,left_rows} Bands.Split{right_left,right_right} False{}: {==}
    case Bands.Rows{left_first,left_rows} Bands.Split{right_left,right_right} True{}: {==}
    case Bands.Split{left_left,left_right} Bands.Rows{right_first,right_rows} False{}: {==}
    case Bands.Split{left_left,left_right} Bands.Rows{right_first,right_rows} True{}: {==}
    case Bands.Split{left_left,left_right} Bands.Split{right_left,right_right} False{}: {==}
    case Bands.Split{left_left,left_right} Bands.Split{right_left,right_right} True{}: {==}

# An unpaired tree comes back unchanged.
def only_left(~Left: Type,~Right: Type,tree: Tree(~Left)) ->
  {Bands.left_of(~Left,~Right,Bands.only(~Left,~Right,tree)) == tree : Tree(~Left)}:
  match tree:
    case Partition.Leaf{block,first,rows,Unit{}}: {==}
    case Partition.Fork{left,right}:
      %only_left(~Left,~Right,left) :
        {Partition.Fork{Bands.left_of(~Left,~Right,Bands.only(~Left,~Right,left)),Bands.left_of(~Left,~Right,Bands.only(~Left,~Right,right))} == Partition.Fork{_,right} : Tree(~Left)}
      %only_left(~Left,~Right,right) :
        {Partition.Fork{Bands.left_of(~Left,~Right,Bands.only(~Left,~Right,left)),Bands.left_of(~Left,~Right,Bands.only(~Left,~Right,right))} ==
          Partition.Fork{Bands.left_of(~Left,~Right,Bands.only(~Left,~Right,left)),_} : Tree(~Left)}
      {==}

# The left tree of a pairing comes back unchanged, whatever the right tree.
def zip_left(~Left: Type,~Right: Type,left: Tree(~Left),right: Tree(~Right)) ->
  {Bands.left_of(~Left,~Right,Bands.zip(~Left,~Right,left,right)) == left : Tree(~Left)}:
  match left right:
    case Partition.Leaf{block,first,rows,Unit{}} Partition.Leaf{other,other_first,other_rows,other_details}: {==}
    case Partition.Leaf{block,first,rows,Unit{}} Partition.Fork{other_left,other_right}: {==}
    case Partition.Fork{left,right} Partition.Leaf{other,other_first,other_rows,other_details}:
      only_left(~Left,~Right,Partition.Fork{left,right})
    case Partition.Fork{left,right} Partition.Fork{other_left,other_right}:
      %zip_left(~Left,~Right,left,other_left) :
        {Partition.Fork{Bands.left_of(~Left,~Right,Bands.zip(~Left,~Right,left,other_left)),Bands.left_of(~Left,~Right,Bands.zip(~Left,~Right,right,other_right))} == Partition.Fork{_,right} : Tree(~Left)}
      %zip_left(~Left,~Right,right,other_right) :
        {Partition.Fork{Bands.left_of(~Left,~Right,Bands.zip(~Left,~Right,left,other_left)),Bands.left_of(~Left,~Right,Bands.zip(~Left,~Right,right,other_right))} ==
          Partition.Fork{Bands.left_of(~Left,~Right,Bands.zip(~Left,~Right,left,other_left)),_} : Tree(~Left)}
      {==}

def KeptRule(~Left: Type,~Right: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Left -> Right -> Resource -> Tree(~Left),
  +layout: Bands.Plan,+enabled: Bool,+geometry: Bands.Geometry,+context: Context,-tree: Tree(~Left)) -> Type:
  @resource: Resource ->
  {Bands.each(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
      layout,enabled && Bands.same_image(geometry,layout),enabled,geometry,context,Bands.only(~Left,~Right,tree),resource) == tree : Tree(~Left)}

def kept_fork(~Left: Type,~Right: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Left -> Right -> Resource -> Tree(~Left),
  +left_plan: Bands.Plan,+right_plan: Bands.Plan,+enabled: Bool,+geometry: Bands.Geometry,+context: Context,
  -left: Tree(~Left),-right: Tree(~Left),resources: Resource & Resource,
  left_rule: KeptRule(~Left,~Right,~Resource,~Context,~share,~leaf,left_plan,enabled,geometry,context,left),
  right_rule: KeptRule(~Left,~Right,~Resource,~Context,~share,~leaf,right_plan,enabled,geometry,context,right)) ->
  {forked(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
      left_plan,right_plan,enabled,geometry,context,Bands.only(~Left,~Right,left),Bands.only(~Left,~Right,right),resources) == Partition.Fork{left,right} : Tree(~Left)}:
  (left_resource,right_resource) = resources
  %left_rule(left_resource) :
    {Partition.Fork{Bands.each(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
          left_plan,enabled && Bands.same_image(geometry,left_plan),enabled,geometry,context,Bands.only(~Left,~Right,left),left_resource),
        Bands.each(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
          right_plan,enabled && Bands.same_image(geometry,right_plan),enabled,geometry,context,Bands.only(~Left,~Right,right),right_resource)} ==
      Partition.Fork{_,right} : Tree(~Left)}
  %right_rule(right_resource) :
    {Partition.Fork{Bands.each(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
          left_plan,enabled && Bands.same_image(geometry,left_plan),enabled,geometry,context,Bands.only(~Left,~Right,left),left_resource),
        Bands.each(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
          right_plan,enabled && Bands.same_image(geometry,right_plan),enabled,geometry,context,Bands.only(~Left,~Right,right),right_resource)} ==
      Partition.Fork{Bands.each(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
          left_plan,enabled && Bands.same_image(geometry,left_plan),enabled,geometry,context,Bands.only(~Left,~Right,left),left_resource),_} : Tree(~Left)}
  {==}

# A paired operation on an unpaired tree returns the tree, on every plan.
def only_kept(~Left: Type,~Right: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Left -> Right -> Resource -> Tree(~Left),
  +layout: Bands.Plan,+mode: Bool,+enabled: Bool,+geometry: Bands.Geometry,+context: Context,tree: Tree(~Left),resource: Resource) ->
  {Bands.each(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
      layout,mode,enabled,geometry,context,Bands.only(~Left,~Right,tree),resource) == tree : Tree(~Left)}:
  match layout tree:
    case Bands.Rows{first,rows} Partition.Leaf{block,own_first,own_rows,Unit{}}: {==}
    case Bands.Rows{first,rows} Partition.Fork{left,right}: only_left(~Left,~Right,Partition.Fork{left,right})
    case Bands.Split{+left_plan,+right_plan} Partition.Leaf{block,own_first,own_rows,Unit{}}:
      split_leaf(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
        left_plan,right_plan,mode,enabled,geometry,context,Bands.Only{block,own_first,own_rows},own_first,own_rows,Unit{},resource)
    case Bands.Split{+left_plan,+right_plan} Partition.Fork{left,right}:
      %Equal.sym(Tree(~Left),
        Bands.each(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
          Bands.Split{left_plan,right_plan},mode,enabled,geometry,context,Partition.Fork{Bands.only(~Left,~Right,left),Bands.only(~Left,~Right,right)},resource),
        forked(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
          left_plan,right_plan,enabled,geometry,context,Bands.only(~Left,~Right,left),Bands.only(~Left,~Right,right),share(resource)),
        fork(~Bands.Paired<Left,Right>,~Left,~Resource,~Context,~share,~Bands.paired(~Left,~Right,~Resource,~Context,~leaf),~Bands.left_of(~Left,~Right),
          left_plan,right_plan,mode,enabled,geometry,context,Bands.only(~Left,~Right,left),Bands.only(~Left,~Right,right),resource)) :
        {_ == Partition.Fork{left,right} : Tree(~Left)}
      kept_fork(~Left,~Right,~Resource,~Context,~share,~leaf,left_plan,right_plan,enabled,geometry,context,left,right,share(resource),
        left_resource => only_kept(~Left,~Right,~Resource,~Context,~share,~leaf,left_plan,enabled && Bands.same_image(geometry,left_plan),enabled,geometry,context,left,left_resource),
        right_resource => only_kept(~Left,~Right,~Resource,~Context,~share,~leaf,right_plan,enabled && Bands.same_image(geometry,right_plan),enabled,geometry,context,right,right_resource))

# Bands equal to a packed model have the model's contents.
def kept(~Element: Data,+model: Model.Model<Element>,-bands: Bands.Bands(~Element),equation: {bands == Model.pack(~Element,model) : Bands.Bands(~Element)}) ->
  {Model.contents(~Element,bands) == Model.content(~Element,model) : Model.Content<Element>}:
  %Equal.sym(Bands.Bands(~Element),bands,Model.pack(~Element,model),equation) : {Model.contents(~Element,_) == Model.content(~Element,model) : Model.Content<Element>}
  Model.packed_contents(~Element,model)

# The contents of an operation on every leaf of one tree.
def content_map(~Element: Data,~Context: Data,~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Element> -> Model.Content<Element>,
  +context: Context,bands: Model.Model<Element>) -> Model.Content<Element>:
  match bands:
    case Model.Leaf{input,first,rows}: leaf_model(context,first,rows,input)
    case Model.Fork{left,right}: Model.Joined{content_map(~Element,~Context,~leaf_model,context,left),content_map(~Element,~Context,~leaf_model,context,right)}

def MapRule(~Element: Data,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Unit -> Bands.Bands(~Element),
  ~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Element> -> Model.Content<Element>) -> Type:
  @+context: Context -> @+first: U32 -> @+rows: U32 -> @+input: Buffer.Model<Element> ->
  {Model.contents(~Element,leaf(context,first,rows,Buffer.pack(~Element,input),Unit{})) == leaf_model(context,first,rows,input) : Model.Content<Element>}

# A one-tree operation on its own plan: the tree induction, once. An
# operation supplies only the equation of its leaf.
def mapped(~Element: Data,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Unit -> Bands.Bands(~Element),
  ~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Element> -> Model.Content<Element>,
  ~leaf_rule: MapRule(~Element,~Context,~leaf,~leaf_model),
  bands: Model.Model<Element>,+mode: Bool,+enabled: Bool,+geometry: Bands.Geometry,+context: Context) ->
  {Model.contents(~Element,Bands.each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,~leaf,~Bands.same(~Storage.Buffer<Element>),
      Model.layout(~Element,bands),mode,enabled,geometry,context,Model.pack(~Element,bands),Unit{})) ==
    content_map(~Element,~Context,~leaf_model,context,bands) : Model.Content<Element>}:
  match bands:
    case Model.Leaf{+input,+first,+rows}: leaf_rule(context,first,rows,input)
    case Model.Fork{+left,+right}:
      %Equal.sym(Bands.Bands(~Element),
        Bands.each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,~leaf,~Bands.same(~Storage.Buffer<Element>),
          Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)},mode,enabled,geometry,context,Partition.Fork{Model.pack(~Element,left),Model.pack(~Element,right)},Unit{}),
        Partition.Fork{Bands.each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,~leaf,~Bands.same(~Storage.Buffer<Element>),
            Model.layout(~Element,left),enabled && Bands.same_image(geometry,Model.layout(~Element,left)),enabled,geometry,context,Model.pack(~Element,left),Unit{}),
          Bands.each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,~leaf,~Bands.same(~Storage.Buffer<Element>),
            Model.layout(~Element,right),enabled && Bands.same_image(geometry,Model.layout(~Element,right)),enabled,geometry,context,Model.pack(~Element,right),Unit{})},
        fork(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,~leaf,~Bands.same(~Storage.Buffer<Element>),
          Model.layout(~Element,left),Model.layout(~Element,right),mode,enabled,geometry,context,Model.pack(~Element,left),Model.pack(~Element,right),Unit{})) :
        {Model.contents(~Element,_) == Model.Joined{content_map(~Element,~Context,~leaf_model,context,left),content_map(~Element,~Context,~leaf_model,context,right)} : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,
        Model.contents(~Element,Bands.each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,~leaf,~Bands.same(~Storage.Buffer<Element>),
          Model.layout(~Element,left),enabled && Bands.same_image(geometry,Model.layout(~Element,left)),enabled,geometry,context,Model.pack(~Element,left),Unit{})),
        content_map(~Element,~Context,~leaf_model,context,left),
        mapped(~Element,~Context,~leaf,~leaf_model,~leaf_rule,left,enabled && Bands.same_image(geometry,Model.layout(~Element,left)),enabled,geometry,context)) :
        {Model.Joined{_,Model.contents(~Element,Bands.each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,~leaf,~Bands.same(~Storage.Buffer<Element>),
            Model.layout(~Element,right),enabled && Bands.same_image(geometry,Model.layout(~Element,right)),enabled,geometry,context,Model.pack(~Element,right),Unit{}))} ==
          Model.Joined{content_map(~Element,~Context,~leaf_model,context,left),content_map(~Element,~Context,~leaf_model,context,right)} : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,
        Model.contents(~Element,Bands.each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,~leaf,~Bands.same(~Storage.Buffer<Element>),
          Model.layout(~Element,right),enabled && Bands.same_image(geometry,Model.layout(~Element,right)),enabled,geometry,context,Model.pack(~Element,right),Unit{})),
        content_map(~Element,~Context,~leaf_model,context,right),
        mapped(~Element,~Context,~leaf,~leaf_model,~leaf_rule,right,enabled && Bands.same_image(geometry,Model.layout(~Element,right)),enabled,geometry,context)) :
        {Model.Joined{content_map(~Element,~Context,~leaf_model,context,left),_} ==
          Model.Joined{content_map(~Element,~Context,~leaf_model,context,left),content_map(~Element,~Context,~leaf_model,context,right)} : Model.Content<Element>}
      {==}

# The contents of an operation on every pair of leaves of two trees of one
# shape; where the shapes differ, the left tree is kept.
def content_zip(~Element: Data,~Context: Data,~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Element> -> Buffer.Model<Element> -> Model.Content<Element>,
  +context: Context,left: Model.Model<Element>,right: Model.Model<Element>) -> Model.Content<Element>:
  match left right:
    case Model.Leaf{left,first,rows} Model.Leaf{right,other_first,other_rows}: leaf_model(context,first,rows,left,right)
    case Model.Fork{left,right} Model.Fork{other_left,other_right}:
      Model.Joined{content_zip(~Element,~Context,~leaf_model,context,left,other_left),content_zip(~Element,~Context,~leaf_model,context,right,other_right)}
    case left right: Model.content(~Element,left)

def ZipRule(~Element: Data,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Unit -> Bands.Bands(~Element),
  ~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Element> -> Buffer.Model<Element> -> Model.Content<Element>) -> Type:
  @+context: Context -> @+first: U32 -> @+rows: U32 -> @+left: Buffer.Model<Element> -> @+right: Buffer.Model<Element> ->
  {Model.contents(~Element,leaf(context,first,rows,Buffer.pack(~Element,left),Buffer.pack(~Element,right),Unit{})) == leaf_model(context,first,rows,left,right) : Model.Content<Element>}

def Zipped(~Element: Data,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Unit -> Bands.Bands(~Element),
  +layout: Bands.Plan,+mode: Bool,+enabled: Bool,+geometry: Bands.Geometry,+context: Context,left: Bands.Bands(~Element),right: Bands.Bands(~Element)) -> Bands.Bands(~Element):
  Bands.each(~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,
    ~Bands.paired(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~leaf),~Bands.left_of(~Storage.Buffer<Element>,~Storage.Buffer<Element>),
    layout,mode,enabled,geometry,context,Bands.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,left,right),Unit{})

# A two-tree operation on the plan of its left tree: the tree induction and
# the shape mismatches, once.
def zipped(~Element: Data,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Unit -> Bands.Bands(~Element),
  ~leaf_model: Context -> U32 -> U32 -> Buffer.Model<Element> -> Buffer.Model<Element> -> Model.Content<Element>,
  ~leaf_rule: ZipRule(~Element,~Context,~leaf,~leaf_model),
  left: Model.Model<Element>,right: Model.Model<Element>,+mode: Bool,+enabled: Bool,+geometry: Bands.Geometry,+context: Context) ->
  {Model.contents(~Element,Zipped(~Element,~Context,~leaf,Model.layout(~Element,left),mode,enabled,geometry,context,Model.pack(~Element,left),Model.pack(~Element,right))) ==
    content_zip(~Element,~Context,~leaf_model,context,left,right) : Model.Content<Element>}:
  match left right:
    case Model.Leaf{+left,+first,+rows} Model.Leaf{+right,other_first,other_rows}: leaf_rule(context,first,rows,left,right)
    case Model.Leaf{+buffer,+first,+rows} Model.Fork{other_left,other_right}: Model.packed_contents(~Element,Model.Leaf{buffer,first,rows})
    case Model.Fork{+left,+right} Model.Leaf{buffer,other_first,other_rows}:
      kept(~Element,Model.Fork{left,right},
        Zipped(~Element,~Context,~leaf,Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)},mode,enabled,geometry,context,
          Model.pack(~Element,Model.Fork{left,right}),Model.pack(~Element,Model.Leaf{buffer,other_first,other_rows})),
        only_kept(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,~leaf,
          Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)},mode,enabled,geometry,context,Model.pack(~Element,Model.Fork{left,right}),Unit{}))
    case Model.Fork{+left,+right} Model.Fork{+other_left,+other_right}:
      %Equal.sym(Bands.Bands(~Element),
        Zipped(~Element,~Context,~leaf,Bands.Split{Model.layout(~Element,left),Model.layout(~Element,right)},mode,enabled,geometry,context,
          Model.pack(~Element,Model.Fork{left,right}),Model.pack(~Element,Model.Fork{other_left,other_right})),
        Partition.Fork{Zipped(~Element,~Context,~leaf,Model.layout(~Element,left),enabled && Bands.same_image(geometry,Model.layout(~Element,left)),enabled,geometry,context,
            Model.pack(~Element,left),Model.pack(~Element,other_left)),
          Zipped(~Element,~Context,~leaf,Model.layout(~Element,right),enabled && Bands.same_image(geometry,Model.layout(~Element,right)),enabled,geometry,context,
            Model.pack(~Element,right),Model.pack(~Element,other_right))},
        fork(~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,
          ~Bands.paired(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~leaf),~Bands.left_of(~Storage.Buffer<Element>,~Storage.Buffer<Element>),
          Model.layout(~Element,left),Model.layout(~Element,right),mode,enabled,geometry,context,
          Bands.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,left),Model.pack(~Element,other_left)),
          Bands.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,Model.pack(~Element,right),Model.pack(~Element,other_right)),Unit{})) :
        {Model.contents(~Element,_) == Model.Joined{content_zip(~Element,~Context,~leaf_model,context,left,other_left),content_zip(~Element,~Context,~leaf_model,context,right,other_right)} : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,
        Model.contents(~Element,Zipped(~Element,~Context,~leaf,Model.layout(~Element,left),enabled && Bands.same_image(geometry,Model.layout(~Element,left)),enabled,geometry,context,
          Model.pack(~Element,left),Model.pack(~Element,other_left))),
        content_zip(~Element,~Context,~leaf_model,context,left,other_left),
        zipped(~Element,~Context,~leaf,~leaf_model,~leaf_rule,left,other_left,enabled && Bands.same_image(geometry,Model.layout(~Element,left)),enabled,geometry,context)) :
        {Model.Joined{_,Model.contents(~Element,Zipped(~Element,~Context,~leaf,Model.layout(~Element,right),enabled && Bands.same_image(geometry,Model.layout(~Element,right)),enabled,geometry,context,
            Model.pack(~Element,right),Model.pack(~Element,other_right)))} ==
          Model.Joined{content_zip(~Element,~Context,~leaf_model,context,left,other_left),content_zip(~Element,~Context,~leaf_model,context,right,other_right)} : Model.Content<Element>}
      %Equal.sym(Model.Content<Element>,
        Model.contents(~Element,Zipped(~Element,~Context,~leaf,Model.layout(~Element,right),enabled && Bands.same_image(geometry,Model.layout(~Element,right)),enabled,geometry,context,
          Model.pack(~Element,right),Model.pack(~Element,other_right))),
        content_zip(~Element,~Context,~leaf_model,context,right,other_right),
        zipped(~Element,~Context,~leaf,~leaf_model,~leaf_rule,right,other_right,enabled && Bands.same_image(geometry,Model.layout(~Element,right)),enabled,geometry,context)) :
        {Model.Joined{content_zip(~Element,~Context,~leaf_model,context,left,other_left),_} ==
          Model.Joined{content_zip(~Element,~Context,~leaf_model,context,left,other_left),content_zip(~Element,~Context,~leaf_model,context,right,other_right)} : Model.Content<Element>}
      {==}
