# Packing implementations supply their own duplicable provenance and a sample
# theorem for the output point being read. Padded positions that are never
# stored need no semantic evidence. This interface has no GEMM row/lane shape
# and does not require a globally aligned output offset.
import Base
import ../../lib/proofs/storage_shape.bend as Shape
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/machine_limits.bend as Machine
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./prepared_packing_samples.bend as Prepared
import ./packing_contents.bend as PackingContents
import ../../lib/math_sequence.bend as Sequence
import ./convolution_guard_facts.bend as Guards
import ./packing_tile_bounds.bend as PackingBounds
import ./packing_readback.bend as PackingReadback
import ./packing_sample_refinement.bend as PackingSampleRefinement
import ./reduction_coordinate_bounds.bend as Coordinates
import ./input_storage.bend as Inputs
import ./concrete_support_guard.bend as Support
import ./tile_partition_bounds.bend as TilePartitionBounds
import ../../lib/proofs/boolean_logic.bend as Boolean

def PointSamples(+problem: ConvSpec.Problem,+offset: U32,+span: U32,+tiles: Tree.Tree<Tile.Tile8>,+output_row: Nat,+output_column: Nat) -> Type:
  @+lane: Sequence.Index(8n) -> @+group: Nat -> @+input_channel: Nat -> @+kernel_row: Nat -> @+kernel_column: Nat ->
  @group_inside: {Nat.is_lt(group,PackingBounds.groups(span)) == True{} : Bool} ->
  @position: {PackingReadback.logical_position(offset,group,lane) == Coordinates.position(problem,output_row,output_column) : Nat} ->
  @channel_inside: {Nat.is_lt(input_channel,Guards.input_channel_count(problem)) == True{} : Bool} ->
  @row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool} ->
  @column_inside: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool} ->
  @kernel_row_inside: {Nat.is_lt(kernel_row,Guards.kernel(problem)) == True{} : Bool} ->
  @kernel_column_inside: {Nat.is_lt(kernel_column,Guards.kernel(problem)) == True{} : Bool} ->
  {Tile.element_at(lane,Tree.read(Tile.Tile8,tiles,U32.from_nat(PackingSampleRefinement.packing_index(problem,group,input_channel,kernel_row,kernel_column)))) == ConvSpec.sample(problem,input_channel,output_row,output_column,kernel_row,kernel_column) : F32}

def Evidence() -> Type:
  ConvSpec.Problem -> U32 -> U32 -> Tree.Tree<Tile.Tile8> -> Nat -> Nat -> Data

def Sampling(~Evidence: Evidence()) -> Type:
  @+limits: Machine.Limits -> @+problem: ConvSpec.Problem -> @+offset: U32 -> @+span: U32 -> @+tiles: Tree.Tree<Tile.Tile8> ->
  @+output_row: Nat -> @+output_column: Nat ->
  @valid: {ConvSpec.valid(problem) == True{} : Bool} -> @supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool} ->
  @certificate: Evidence(problem,offset,span,tiles,output_row,output_column) -> PointSamples(problem,offset,span,tiles,output_row,output_column)

def dense_evidence(problem: ConvSpec.Problem,offset: U32,span: U32,tiles: Tree.Tree<Tile.Tile8>,output_row: Nat,output_column: Nat) -> Data:
  Prepared.LeafCertificate(problem,offset,span,tiles)

def dense_samples(+limits: Machine.Limits,+problem: ConvSpec.Problem,+offset: U32,+span: U32,+tiles: Tree.Tree<Tile.Tile8>,+output_row: Nat,+output_column: Nat,
  valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},certificate: dense_evidence(problem,offset,span,tiles,output_row,output_column)) ->
  PointSamples(problem,offset,span,tiles,output_row,output_column):
  lane => group => input_channel => kernel_row => kernel_column => group_inside => position => channel_inside => row_inside => column_inside => kernel_row_inside => kernel_column_inside =>
    Prepared.certified_leaf_samples(limits,problem,offset,span,tiles,valid,supported,certificate)(lane,group,input_channel,output_row,output_column,kernel_row,kernel_column,group_inside,position,channel_inside,row_inside,column_inside,kernel_row_inside,kernel_column_inside)

# A band leaf packs its window at any output offset; its positions end within seven of the window grid.
type BandEvidence<-problem: ConvSpec.Problem,-offset: U32,-span: U32,-tiles: Tree.Tree<Tile.Tile8>> is Data:
  BandEvidence{storage_contract: Inputs.Contract<problem>,span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
    range_bound: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Nat.add(PackingContents.grid(problem),7n)) == True{} : Bool},
    contents: {tiles == Prepared.packed_output(Prepared.allocated_pack(problem,storage_contract,offset,span)) : Tree.Tree<Tile.Tile8>}}

def band_evidence(problem: ConvSpec.Problem,offset: U32,span: U32,tiles: Tree.Tree<Tile.Tile8>,output_row: Nat,output_column: Nat) -> Data:
  BandEvidence<problem,offset,span,tiles>

def band_samples(+limits: Machine.Limits,+problem: ConvSpec.Problem,+offset: U32,+span: U32,+tiles: Tree.Tree<Tile.Tile8>,+output_row: Nat,+output_column: Nat,
  valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},certificate: band_evidence(problem,offset,span,tiles,output_row,output_column)) ->
  PointSamples(problem,offset,span,tiles,output_row,output_column):
  BandEvidence{+storage_contract,+span_bound,+range_bound,+contents} = certificate
  %Equal.sym(Tree.Tree<Tile.Tile8>,tiles,Prepared.packed_output(Prepared.allocated_pack(problem,storage_contract,offset,span)),contents) :
    PointSamples(problem,offset,span,_,output_row,output_column)
  lane => group => input_channel => kernel_row => kernel_column => group_inside => position => channel_inside => row_inside => column_inside => kernel_row_inside => kernel_column_inside =>
    Prepared.leaf_samples(limits,problem,storage_contract,offset,span,valid,supported,span_bound,range_bound)(lane,group,input_channel,output_row,output_column,kernel_row,kernel_column,group_inside,position,channel_inside,row_inside,column_inside,kernel_row_inside,kernel_column_inside)

# A reused workspace records its actual shape and written prefix. GEMM samples
# only that prefix; the untouched suffix may contain tiles from an earlier block.
type WorkspaceEvidence<-problem: ConvSpec.Problem,-offset: U32,-span: U32,-tiles: Tree.Tree<Tile.Tile8>> is Data:
  WorkspaceEvidence{storage_contract: Inputs.Contract<problem>,workspace: Tree.Tree<Tile.Tile8>,depth: Nat,
    balanced: Shape.Balanced(Tile.Tile8,depth,workspace),depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
    room: {Nat.is_le(U32.to_nat(PackingBounds.count(span,problem)),U32.to_nat(Tree.capacity(Tile.Tile8,workspace))) == True{} : Bool},
    span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
    range_bound: {Nat.is_le(Nat.add(U32.to_nat(offset),Nat.mul(PackingBounds.groups(span),8n)),Nat.add(PackingContents.grid(problem),7n)) == True{} : Bool},
    contents: {tiles == Prepared.packed_output(Prepared.workspace_pack(problem,storage_contract,offset,span,workspace)) : Tree.Tree<Tile.Tile8>}}

def workspace_evidence(problem: ConvSpec.Problem,offset: U32,span: U32,tiles: Tree.Tree<Tile.Tile8>,output_row: Nat,output_column: Nat) -> Data:
  WorkspaceEvidence<problem,offset,span,tiles>

def workspace_samples(+limits: Machine.Limits,+problem: ConvSpec.Problem,+offset: U32,+span: U32,+tiles: Tree.Tree<Tile.Tile8>,+output_row: Nat,+output_column: Nat,
  valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},certificate: workspace_evidence(problem,offset,span,tiles,output_row,output_column)) ->
  PointSamples(problem,offset,span,tiles,output_row,output_column):
  WorkspaceEvidence{+storage_contract,+workspace,+depth,+balanced,+depth_ok,+room,+span_bound,+range_bound,+contents} = certificate
  %Equal.sym(Tree.Tree<Tile.Tile8>,tiles,Prepared.packed_output(Prepared.workspace_pack(problem,storage_contract,offset,span,workspace)),contents) :
    PointSamples(problem,offset,span,_,output_row,output_column)
  lane => group => input_channel => kernel_row => kernel_column => group_inside => position => channel_inside => row_inside => column_inside => kernel_row_inside => kernel_column_inside =>
    Prepared.workspace_samples(limits,problem,storage_contract,offset,span,workspace,depth,balanced,depth_ok,room,valid,supported,span_bound,range_bound)(
      lane,group,input_channel,output_row,output_column,kernel_row,kernel_column,group_inside,position,channel_inside,row_inside,column_inside,kernel_row_inside,kernel_column_inside)
