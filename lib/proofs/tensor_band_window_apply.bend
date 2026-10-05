# The window apply tree for any leaf: a leaf plan runs the leaf, a split plan
# runs both halves in either mode, and trees of another shape keep the cores.
# Window operations use these equations and prove only their own leaf.
import Base
import ../tensor_band.bend as Bands
import ../tensor_band_window.bend as BandWindow
import ../storage_buffer.bend as Storage
import ../traversal_partition.bend as Partition
import ./tensor_band_model.bend as Model
import ./storage_buffer_model.bend as Buffer
import ./tensor_band_halo.bend as BandHalo
import ./tensor_band_each.bend as Each

def leaf(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> BandWindow.Halo<Element> -> Bands.Bands(~Element),
  +first: U32,+rows: U32,+mode: Bool,+enabled: Bool,+geometry: Bands.Geometry,+context: Context,
  +core: Buffer.Model<Element>,+core_first: U32,+core_rows: U32,+above: Buffer.Model<Element>,+below: Buffer.Model<Element>,+window: BandWindow.RowWindow,+halo_first: U32,+halo_rows: U32) ->
  {BandWindow.apply(~Element,~Context,~leaf,Bands.Rows{first,rows},mode,enabled,geometry,context,
      Model.pack(~Element,Model.Leaf{core,core_first,core_rows}),BandHalo.pack_owners(~Element,BandHalo.OwnedLeaf{above,below,window,halo_first,halo_rows})) ==
    leaf(context,first,rows,Buffer.pack(~Element,core),BandWindow.Halo{Buffer.pack(~Element,above),Buffer.pack(~Element,below),window}) : Bands.Bands(~Element)}:
  {==}

def fork(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> BandWindow.Halo<Element> -> Bands.Bands(~Element),
  +left_plan: Bands.Plan,+right_plan: Bands.Plan,+mode: Bool,+enabled: Bool,+geometry: Bands.Geometry,+context: Context,
  +left: Model.Model<Element>,+right: Model.Model<Element>,+left_halo: BandHalo.Owners<Element>,+right_halo: BandHalo.Owners<Element>) ->
  {BandWindow.apply(~Element,~Context,~leaf,Bands.Split{left_plan,right_plan},mode,enabled,geometry,context,
      Model.pack(~Element,Model.Fork{left,right}),BandHalo.pack_owners(~Element,BandHalo.OwnedFork{left_halo,right_halo})) ==
    Partition.Fork{BandWindow.apply(~Element,~Context,~leaf,left_plan,enabled && Bands.same_image(geometry,left_plan),enabled,geometry,context,
        Model.pack(~Element,left),BandHalo.pack_owners(~Element,left_halo)),
      BandWindow.apply(~Element,~Context,~leaf,right_plan,enabled && Bands.same_image(geometry,right_plan),enabled,geometry,context,
        Model.pack(~Element,right),BandHalo.pack_owners(~Element,right_halo))} : Bands.Bands(~Element)}:
  Each.fork(~Bands.Paired<Storage.Buffer<Element>,BandWindow.Halo<Element>>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,
    ~Bands.paired(~Storage.Buffer<Element>,~BandWindow.Halo<Element>,~Unit,~Context,~BandWindow.window_band(~Element,~Context,~leaf)),
    ~Bands.left_of(~Storage.Buffer<Element>,~BandWindow.Halo<Element>),
    left_plan,right_plan,mode,enabled,geometry,context,
    Bands.zip(~Storage.Buffer<Element>,~BandWindow.Halo<Element>,Model.pack(~Element,left),BandHalo.pack_owners(~Element,left_halo)),
    Bands.zip(~Storage.Buffer<Element>,~BandWindow.Halo<Element>,Model.pack(~Element,right),BandHalo.pack_owners(~Element,right_halo)),Unit{})

# One core leaf under a leaf plan whose halos are a fork: the core is kept.
def leaf_forked_halos(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> BandWindow.Halo<Element> -> Bands.Bands(~Element),
  +first: U32,+rows: U32,+mode: Bool,+enabled: Bool,+geometry: Bands.Geometry,+context: Context,
  +core: Buffer.Model<Element>,+core_first: U32,+core_rows: U32,+left_halo: BandHalo.Owners<Element>,+right_halo: BandHalo.Owners<Element>) ->
  {BandWindow.apply(~Element,~Context,~leaf,Bands.Rows{first,rows},mode,enabled,geometry,context,
      Model.pack(~Element,Model.Leaf{core,core_first,core_rows}),BandHalo.pack_owners(~Element,BandHalo.OwnedFork{left_halo,right_halo})) ==
    Model.pack(~Element,Model.Leaf{core,core_first,core_rows}) : Bands.Bands(~Element)}:
  {==}

# Forked cores under a split plan whose halos are one leaf: the cores are kept.
def fork_leaf_halos(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> BandWindow.Halo<Element> -> Bands.Bands(~Element),
  +left_plan: Bands.Plan,+right_plan: Bands.Plan,+mode: Bool,+enabled: Bool,+geometry: Bands.Geometry,+context: Context,
  +left: Model.Model<Element>,+right: Model.Model<Element>,+above: Buffer.Model<Element>,+below: Buffer.Model<Element>,+window: BandWindow.RowWindow,+first: U32,+rows: U32) ->
  {BandWindow.apply(~Element,~Context,~leaf,Bands.Split{left_plan,right_plan},mode,enabled,geometry,context,
      Model.pack(~Element,Model.Fork{left,right}),BandHalo.pack_owners(~Element,BandHalo.OwnedLeaf{above,below,window,first,rows})) ==
    Model.pack(~Element,Model.Fork{left,right}) : Bands.Bands(~Element)}:
  Each.only_kept(~Storage.Buffer<Element>,~BandWindow.Halo<Element>,~Unit,~Context,~Bands.unshared,~BandWindow.window_band(~Element,~Context,~leaf),
    Bands.Split{left_plan,right_plan},mode,enabled,geometry,context,Model.pack(~Element,Model.Fork{left,right}),Unit{})

# Forked cores under a leaf plan are kept, whatever the halos.
def rows_fork(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> BandWindow.Halo<Element> -> Bands.Bands(~Element),
  +first: U32,+rows: U32,+mode: Bool,+enabled: Bool,+geometry: Bands.Geometry,+context: Context,
  +left: Model.Model<Element>,+right: Model.Model<Element>,halos: BandHalo.Owners<Element>) ->
  {BandWindow.apply(~Element,~Context,~leaf,Bands.Rows{first,rows},mode,enabled,geometry,context,
      Model.pack(~Element,Model.Fork{left,right}),BandHalo.pack_owners(~Element,halos)) == Model.pack(~Element,Model.Fork{left,right}) : Bands.Bands(~Element)}:
  match halos:
    case BandHalo.OwnedLeaf{above,below,window,halo_first,halo_rows}:
      Each.only_left(~Storage.Buffer<Element>,~BandWindow.Halo<Element>,Model.pack(~Element,Model.Fork{left,right}))
    case BandHalo.OwnedFork{+left_halo,+right_halo}:
      Each.zip_left(~Storage.Buffer<Element>,~BandWindow.Halo<Element>,Model.pack(~Element,Model.Fork{left,right}),BandHalo.pack_owners(~Element,BandHalo.OwnedFork{left_halo,right_halo}))

# One core leaf under a split plan is kept, whatever the halos.
def split_leaf(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> BandWindow.Halo<Element> -> Bands.Bands(~Element),
  +left_plan: Bands.Plan,+right_plan: Bands.Plan,+mode: Bool,+enabled: Bool,+geometry: Bands.Geometry,+context: Context,
  +core: Buffer.Model<Element>,+first: U32,+rows: U32,halos: BandHalo.Owners<Element>) ->
  {BandWindow.apply(~Element,~Context,~leaf,Bands.Split{left_plan,right_plan},mode,enabled,geometry,context,
      Model.pack(~Element,Model.Leaf{core,first,rows}),BandHalo.pack_owners(~Element,halos)) == Model.pack(~Element,Model.Leaf{core,first,rows}) : Bands.Bands(~Element)}:
  match halos:
    case BandHalo.OwnedLeaf{+above,+below,+window,halo_first,halo_rows}:
      Each.split_leaf(~Bands.Paired<Storage.Buffer<Element>,BandWindow.Halo<Element>>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,
        ~Bands.paired(~Storage.Buffer<Element>,~BandWindow.Halo<Element>,~Unit,~Context,~BandWindow.window_band(~Element,~Context,~leaf)),
        ~Bands.left_of(~Storage.Buffer<Element>,~BandWindow.Halo<Element>),
        left_plan,right_plan,mode,enabled,geometry,context,
        Bands.Both{Buffer.pack(~Element,core),BandWindow.Halo{Buffer.pack(~Element,above),Buffer.pack(~Element,below),window}},first,rows,Unit{},Unit{})
    case BandHalo.OwnedFork{left_halo,right_halo}:
      Each.split_leaf(~Bands.Paired<Storage.Buffer<Element>,BandWindow.Halo<Element>>,~Storage.Buffer<Element>,~Unit,~Context,~Bands.unshared,
        ~Bands.paired(~Storage.Buffer<Element>,~BandWindow.Halo<Element>,~Unit,~Context,~BandWindow.window_band(~Element,~Context,~leaf)),
        ~Bands.left_of(~Storage.Buffer<Element>,~BandWindow.Halo<Element>),
        left_plan,right_plan,mode,enabled,geometry,context,
        Bands.Only{Buffer.pack(~Element,core),first,rows},first,rows,Unit{},Unit{})
