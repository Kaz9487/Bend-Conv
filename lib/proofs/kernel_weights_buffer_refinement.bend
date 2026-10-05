# The Buffer wrapper preserves the existing weight-reorder content theorem.
# No reduction, GEMM loop, or numerical accumulation is opened here.
import Base
import ../kernel_gemm_buffer.bend as GemmBuffer
import ../kernel_weights.bend as Weights
import ../kernel_tile.bend as Tile
import ./kernel_weights_refinement.bend as WeightsLaws
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws

def reordered(+rows: U32,+inner: U32,+weights: Buffer.Model<F32>) ->
  {Buffer.contents(~Tile.Tile8,GemmBuffer.reorder(rows,inner,Buffer.pack(~F32,weights))) ==
    Buffer.Content{WeightsLaws.run(rows,inner,Buffer.tree(~F32,weights)),U32.to_nat((((rows + 7) / 8) * inner : U32))} : Buffer.Content<Tile.Tile8>}:
  Buffer.Model{+tree,length,certificate} = weights
  +count = U32.to_nat((((rows + 7) / 8) * inner : U32))
  %Equal.sym(Array<Tile.Tile8>,Weights.run(rows,inner,Tree.pack(F32,tree)),Tree.pack(Tile.Tile8,WeightsLaws.run(rows,inner,tree)),WeightsLaws.run_matches(rows,inner,tree)) :
    {Buffer.Content{Tree.reflect(Tile.Tile8,_),count} == Buffer.Content{WeightsLaws.run(rows,inner,tree),count} : Buffer.Content<Tile.Tile8>}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Tree.reflect(Tile.Tile8,Tree.pack(Tile.Tile8,WeightsLaws.run(rows,inner,tree))),WeightsLaws.run(rows,inner,tree),StorageLaws.reflect_packed(Tile.Tile8,WeightsLaws.run(rows,inner,tree))) :
    {Buffer.Content{_,count} == Buffer.Content{WeightsLaws.run(rows,inner,tree),count} : Buffer.Content<Tile.Tile8>}
  {==}
