import ../../lib/proofs/storage_shape.bend as Shape
# All prepared tiles: real writes, complete spatial coverage, and exact readback.
import Base
import ./input_storage.bend as Inputs
import ../../lib/traversal_partition.bend as Partition
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../inference.bend as NativeConv
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/write_program.bend as Writes
import ./conditional_write_coverage.bend as Coverage
import ../../lib/proofs/sequence_membership.bend as Members
import ./convolution_guard_facts.bend as Guards
import ./packing_storage_refinement.bend as PackingStorage
import ./prepared_packing_samples.bend as Prepared
import ./tile_layout.bend as TileLayout
import ./tile_partition_coverage.bend as TilePartitionCoverage
import ./reduction_coordinate_bounds.bend as Coordinates
import ./supported_padded_coordinates.bend as Padded
import ./output_values.bend as Values
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./gather_storage_model.bend as GatherModel
import ./gather_write_program.bend as GatherWrites
import ./gather_copy_coordinates.bend as GatherCopyCoordinates
import ./gather_leaf_writes.bend as GatherLeafWrites
import ./gather_block_coverage.bend as GatherBlockCoverage
import ./gather_block_writes.bend as BlockWrites
import ./bounded_write_readback.bend as WriteReadback

def result(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,tiles: PackingStorage.Tiles()) -> GatherModel.ConvResult():
  MicroLeaf.layout_result(problem,storage_contract,TileLayout.tiles(tiles))

def writes(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,tiles: PackingStorage.Tiles()) -> List<&2,Writes.Assignment<F32>>:
  GatherWrites.gather_writes(result(problem,storage_contract,tiles),Padded.spatial(problem))

def tree_evidence(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+tiles: PackingStorage.Tiles(),+channel: Nat,+output_row_index: Nat,+output_column_index: Nat,+capacity: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},certificates: Prepared.AllCertificates(problem,tiles),
  +room: {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),GatherCopyCoordinates.spatial(problem)),U32.to_nat(capacity)) == True{} : Bool},
  +row_inside: {Nat.is_lt(output_row_index,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(output_column_index,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  WriteReadback.WriteEvidence<GatherLeafWrites.query(problem,channel,output_row_index,output_column_index),Values.matrix_value(problem,channel,output_row_index,output_column_index),capacity,writes(problem,storage_contract,tiles)>:
  match tiles:
    case Partition.Leaf{+input,+offset,+span,Unit{}}:
      BlockWrites.block(limits,problem,storage_contract,input,offset,span,channel,output_row_index,output_column_index,capacity,
        valid,supported,certificates,room,row_inside,column_inside)
    case Partition.Fork{+left,+right}:
      Shape.Both{left_certificates,right_certificates} = certificates
      WriteReadback.concatenated_evidence(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index),Values.matrix_value(problem,channel,output_row_index,output_column_index),capacity,writes(problem,storage_contract,left),writes(problem,storage_contract,right),
        tree_evidence(limits,problem,storage_contract,left,channel,output_row_index,output_column_index,capacity,valid,supported,left_certificates,room,row_inside,column_inside),
        tree_evidence(limits,problem,storage_contract,right,channel,output_row_index,output_column_index,capacity,valid,supported,right_certificates,room,row_inside,column_inside))

law tree_coverage:
  for +limits: Machine.Limits
  for +problem: ConvSpec.Problem
  for +storage_contract: Inputs.Contract<problem>
  for +tiles: PackingStorage.Tiles()
  for +channel: Nat
  for +output_row_index: Nat
  for +output_column_index: Nat
  for +valid: {ConvSpec.valid(problem) == True{} : Bool}
  for +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}
  for certificates: Prepared.AllCertificates(problem,tiles)
  for member: Members.Member(Nat,Coordinates.position(problem,output_row_index,output_column_index),TilePartitionCoverage.positions(TileLayout.tiles(tiles)))
  for +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}
  for +row_inside: {Nat.is_lt(output_row_index,ConvSpec.out_height(problem)) == True{} : Bool}
  for +column_inside: {Nat.is_lt(output_column_index,ConvSpec.out_width(problem)) == True{} : Bool}
  {GatherWrites.contains(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index),writes(problem,storage_contract,tiles)) == True{} : Bool}

def leaf_coverage(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+input: Tree.Tree<Tile.Tile8>,+offset: U32,+span: U32,+channel: Nat,+output_row_index: Nat,+output_column_index: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},certificate: Prepared.LeafCertificate(problem,offset,span,input),
  witness: Members.RangeIndex<U32.to_nat(span),U32.to_nat(offset),Coordinates.position(problem,output_row_index,output_column_index)>,
  channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},row_inside: {Nat.is_lt(output_row_index,ConvSpec.out_height(problem)) == True{} : Bool},column_inside: {Nat.is_lt(output_column_index,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {GatherWrites.contains(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index),writes(problem,storage_contract,Partition.Leaf{input,offset,span,Unit{}})) == True{} : Bool}:
  Members.RangeIndex{column,inside,exact} = witness
  GatherBlockCoverage.block(limits,problem,storage_contract,input,offset,span,channel,column,output_row_index,output_column_index,valid,supported,certificate,channel_inside,inside,exact,row_inside,column_inside)

def fork_coverage(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+left: PackingStorage.Tiles(),+right: PackingStorage.Tiles(),+channel: Nat,+output_row_index: Nat,+output_column_index: Nat,
  choice: Either<&2,&2,Members.Member(Nat,Coordinates.position(problem,output_row_index,output_column_index),TilePartitionCoverage.positions(TileLayout.tiles(left))),Members.Member(Nat,Coordinates.position(problem,output_row_index,output_column_index),TilePartitionCoverage.positions(TileLayout.tiles(right)))>,
  left_proof: Members.Member(Nat,Coordinates.position(problem,output_row_index,output_column_index),TilePartitionCoverage.positions(TileLayout.tiles(left))) -> {GatherWrites.contains(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index),writes(problem,storage_contract,left)) == True{} : Bool},
  right_proof: Members.Member(Nat,Coordinates.position(problem,output_row_index,output_column_index),TilePartitionCoverage.positions(TileLayout.tiles(right))) -> {GatherWrites.contains(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index),writes(problem,storage_contract,right)) == True{} : Bool}) ->
  {GatherWrites.contains(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index),writes(problem,storage_contract,Partition.Fork{left,right})) == True{} : Bool}:
  match choice:
    case Inl{member}:
      GatherWrites.concatenation_covered(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index),writes(problem,storage_contract,left),writes(problem,storage_contract,right),Inl{left_proof(member)})
    case Inr{member}:
      GatherWrites.concatenation_covered(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index),writes(problem,storage_contract,left),writes(problem,storage_contract,right),Inr{right_proof(member)})
def tree_coverage(limits,problem,storage_contract,tiles,channel,output_row_index,output_column_index,valid,supported,certificates,member,channel_inside,row_inside,column_inside):
  match tiles:
    case Partition.Leaf{+input,+offset,+span,Unit{}}:
      leaf_coverage(limits,problem,storage_contract,input,offset,span,channel,output_row_index,output_column_index,valid,supported,certificates,
        Members.range_index(U32.to_nat(span),U32.to_nat(offset),Coordinates.position(problem,output_row_index,output_column_index),member),channel_inside,row_inside,column_inside)
    case Partition.Fork{+left,+right}:
      Shape.Both{left_certificates,right_certificates} = certificates
      fork_coverage(problem,storage_contract,left,right,channel,output_row_index,output_column_index,
        Members.split_concatenated(Nat,Coordinates.position(problem,output_row_index,output_column_index),TilePartitionCoverage.positions(TileLayout.tiles(left)),TilePartitionCoverage.positions(TileLayout.tiles(right)),member),
        left_member => tree_coverage(limits,problem,storage_contract,left,channel,output_row_index,output_column_index,valid,supported,left_certificates,left_member,channel_inside,row_inside,column_inside),
        right_member => tree_coverage(limits,problem,storage_contract,right,channel,output_row_index,output_column_index,valid,supported,right_certificates,right_member,channel_inside,row_inside,column_inside))

def readback_with_evidence(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+tiles: PackingStorage.Tiles(),+output: Tree.Tree<F32>,+depth: Nat,+query: U32,+channel: Nat,+output_row_index: Nat,+output_column_index: Nat,
  balanced: Shape.Balanced(F32,depth,output),depth_bound: {Nat.is_le(depth,30n) == True{} : Bool},
  query_inside: {U32.is_lt(query,Tree.capacity(F32,output)) == True{} : Bool},+query_exact: {U32.to_nat(query) == GatherLeafWrites.query(problem,channel,output_row_index,output_column_index) : Nat},
  evidence: WriteReadback.WriteEvidence<GatherLeafWrites.query(problem,channel,output_row_index,output_column_index),Values.matrix_value(problem,channel,output_row_index,output_column_index),Tree.capacity(F32,output),writes(problem,storage_contract,tiles)>,
  covered: {GatherWrites.contains(GatherLeafWrites.query(problem,channel,output_row_index,output_column_index),writes(problem,storage_contract,tiles)) == True{} : Bool}) ->
  {Tree.read(F32,GatherModel.gather(result(problem,storage_contract,tiles),output,Padded.spatial(problem)),query) == Values.matrix_value(problem,channel,output_row_index,output_column_index) : F32}:
  WriteReadback.WriteEvidence{bounds,values} = evidence
  %Equal.sym(Tree.Tree<F32>,GatherModel.gather(result(problem,storage_contract,tiles),output,Padded.spatial(problem)),Writes.pure(F32,writes(problem,storage_contract,tiles),output),GatherWrites.gather_matches_writes(result(problem,storage_contract,tiles),output,Padded.spatial(problem))) :
    {Tree.read(F32,_,query) == Values.matrix_value(problem,channel,output_row_index,output_column_index) : F32}
  WriteReadback.readback_from_matching_covered_writes(writes(problem,storage_contract,tiles),output,depth,query,Values.matrix_value(problem,channel,output_row_index,output_column_index),balanced,depth_bound,bounds,query_inside,
    %Equal.sym(Nat,U32.to_nat(query),GatherLeafWrites.query(problem,channel,output_row_index,output_column_index),query_exact) : Coverage.MatchingWritesHaveValue(_,Values.matrix_value(problem,channel,output_row_index,output_column_index),WriteReadback.natural_write_addresses(writes(problem,storage_contract,tiles)))
    values,
    %Equal.sym(Nat,U32.to_nat(query),GatherLeafWrites.query(problem,channel,output_row_index,output_column_index),query_exact) : {GatherWrites.contains(_,writes(problem,storage_contract,tiles)) == True{} : Bool}
    covered)
