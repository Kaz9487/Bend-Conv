# Three-input selection: shared shape/band helpers and the common lane loop.
import Base
import ./tensor.bend as Tensors
import ./tensor_shape.bend as Shapes
import ./tensor_layout.bend as Layout
import ./tensor_lanes.bend as Lanes
import ./tensor_band.bend as Bands
import ./tensor_band_schedule.bend as Schedule
import ./storage_buffer.bend as Storage
import ./traversal_partition.bend as Partition

def dense(+shape: List<&2,U32>,+count: U32,+workers: U32,mask: Tensors.Tensor<F32>,yes: Tensors.Tensor<F32>,no: Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  Tensors.with_dense(~F32,~Tensors.Tensor<F32>,mask,mask => ms => mf => mt => mw =>
    Tensors.with_dense(~F32,~Tensors.Tensor<F32>,yes,yes => ys => fill => yt => yw =>
      Tensors.with_dense(~F32,~Tensors.Tensor<F32>,no,no => ns => nf => nt => nw =>
        Tensors.Tensor{Lanes.select(U32.to_nat(count),mask,yes,no),shape,fill,Tensors.first_failure(mt,Tensors.first_failure(yt,nt)),workers})))

def expanded(same: Bool,shape: List<&2,U32>,tensor: Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  match same:
    case True{}: tensor
    case False{}: Layout.broadcast_to(~F32,tensor,shape)

type BandPlan is Data:
  BandPlan{geometry: Bands.Geometry,layout: Bands.Plan}

def planned(shape: List<&2,U32>,fill: F32,status: Tensors.Status,workers: U32,+geometry: Bands.Geometry,owners: Bands.Bands(~F32) & Bands.Plan) ->
  Tensors.Tensor<F32> & Maybe<&2,BandPlan>:
  (bands,+layout) = owners
  (Tensors.Banded{bands,geometry,shape,fill,status,workers},Some{BandPlan{geometry,layout}})

def plan(tensor: Tensors.Tensor<F32>) -> Tensors.Tensor<F32> & Maybe<&2,BandPlan>:
  match tensor:
    case Tensors.Tensor{buffer,shape,fill,status,workers}: (Tensors.Tensor{buffer,shape,fill,status,workers},None{})
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}: planned(shape,fill,status,workers,geometry,Bands.plan(~F32,bands))

def common(left: Maybe<&2,BandPlan>,right: Maybe<&2,BandPlan>) -> Maybe<&2,BandPlan>:
  match left right:
    case Some{BandPlan{+geometry,left}} Some{BandPlan{other,right}}: Some{BandPlan{geometry,Schedule.alignment(geometry,left,right)}}
    case Some{left} None{}: Some{left}
    case None{} right: right

def selected_leaf(geometry: Bands.Geometry,first: U32,rows: U32,
  input: Bands.Paired<Storage.Buffer<F32>,Bands.Paired<Storage.Buffer<F32>,Storage.Buffer<F32>>>,resource: Unit) -> Bands.Bands(~F32):
  match input:
    case Bands.Both{yes,Bands.Both{mask,no}}:
      +rows = rows
      Partition.Leaf{Lanes.select(U32.to_nat(Bands.local_count(geometry,rows)),mask,yes,no),first,rows,Unit{}}
    case Bands.Both{yes,Bands.Only{mask,own_first,own_rows}}: Partition.Leaf{yes,first,rows,Unit{}}
    case Bands.Only{yes,own_first,own_rows}: Partition.Leaf{yes,own_first,own_rows,Unit{}}

def selected_bands(+geometry: Bands.Geometry,+layout: Bands.Plan,+parallel: Bool,mask: Bands.Bands(~F32),yes: Bands.Bands(~F32),no: Bands.Bands(~F32)) -> Bands.Bands(~F32):
  Bands.each(~Bands.Paired<Storage.Buffer<F32>,Bands.Paired<Storage.Buffer<F32>,Storage.Buffer<F32>>>,~Storage.Buffer<F32>,~Unit,~Bands.Geometry,
    ~Bands.unshared,~selected_leaf,~Bands.left_of(~Storage.Buffer<F32>,~Bands.Paired<Storage.Buffer<F32>,Storage.Buffer<F32>>),
    layout,parallel && Bands.same_image(geometry,layout),parallel,geometry,geometry,
    Bands.zip(~Storage.Buffer<F32>,~Bands.Paired<Storage.Buffer<F32>,Storage.Buffer<F32>>,yes,
      Bands.zip(~Storage.Buffer<F32>,~Storage.Buffer<F32>,mask,no)),Unit{})

def dispatch(layout: Maybe<&2,BandPlan>,shape: List<&2,U32>,count: U32,fill: F32,workers: U32,
  mask: Tensors.Tensor<F32>,yes: Tensors.Tensor<F32>,no: Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  match layout:
    case None{}: dense(shape,count,workers,mask,yes,no)
    case Some{BandPlan{+geometry,+layout}}:
      +workers = workers
      Tensors.Banded{selected_bands(geometry,layout,Schedule.parallel(geometry,workers,16.0,0.875,layout),
        Tensors.aligned(~F32,geometry,layout,mask),Tensors.aligned(~F32,geometry,layout,yes),Tensors.aligned(~F32,geometry,layout,no)),
        geometry,shape,fill,Tensors.Usable{},workers}

def matching(+shape: List<&2,U32>,+count: U32,+fill: F32,+workers: U32,mask: Tensors.Tensor<F32>,yes: Tensors.Tensor<F32>,no: Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  Tensors.then(~Tensors.Tensor<F32>,~Maybe<&2,BandPlan>,~Tensors.Tensor<F32>,plan(mask),mask => mask_plan =>
    Tensors.then(~Tensors.Tensor<F32>,~Maybe<&2,BandPlan>,~Tensors.Tensor<F32>,plan(yes),yes => yes_plan =>
      Tensors.then(~Tensors.Tensor<F32>,~Maybe<&2,BandPlan>,~Tensors.Tensor<F32>,plan(no),no => no_plan =>
        dispatch(common(mask_plan,common(yes_plan,no_plan)),shape,count,fill,workers,mask,yes,no))))

def shapes(same: Bool,+shape: List<&2,U32>,count: Maybe<&2,U32>,mask_shape: List<&2,U32>,yes_shape: List<&2,U32>,no_shape: List<&2,U32>,fill: F32,workers: U32,
  mask: Tensors.Tensor<F32>,yes: Tensors.Tensor<F32>,no: Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  match same count:
    case True{} Some{count}: matching(shape,count,fill,workers,mask,yes,no)
    case False{} Some{count}:
      dense(shape,count,workers,expanded(Shapes.same_extents(mask_shape,shape),shape,mask),
        expanded(Shapes.same_extents(yes_shape,shape),shape,yes),expanded(Shapes.same_extents(no_shape,shape),shape,no))
    case same None{}: Tensors.failed(~F32,"select",Tensors.InvalidShape{shape},shape,fill,workers)

def masked_shape(common: Maybe<&2,List<&2,U32>>,+mask_shape: List<&2,U32>,+yes_shape: List<&2,U32>,+no_shape: List<&2,U32>,values_shape: List<&2,U32>,fill: F32,workers: U32,
  mask: Tensors.Tensor<F32>,yes: Tensors.Tensor<F32>,no: Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  match common:
    case Some{+shape}:
      shapes(Shapes.same_extents(mask_shape,yes_shape) && Shapes.same_extents(yes_shape,no_shape),shape,Tensors.count_of(shape),mask_shape,yes_shape,no_shape,fill,workers,mask,yes,no)
    case None{}: Tensors.failed(~F32,"select",Tensors.ShapeMismatch{mask_shape,values_shape},mask_shape,fill,workers)

def values_shape(common: Maybe<&2,List<&2,U32>>,+mask_shape: List<&2,U32>,+yes_shape: List<&2,U32>,no_shape: List<&2,U32>,fill: F32,workers: U32,
  mask: Tensors.Tensor<F32>,yes: Tensors.Tensor<F32>,no: Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  match common:
    case Some{+shape}: masked_shape(Shapes.broadcast(mask_shape,shape),mask_shape,yes_shape,no_shape,shape,fill,workers,mask,yes,no)
    case None{}: Tensors.failed(~F32,"select",Tensors.ShapeMismatch{yes_shape,no_shape},yes_shape,fill,workers)

def status(status: Tensors.Status,mask_shape: List<&2,U32>,+yes_shape: List<&2,U32>,+no_shape: List<&2,U32>,fill: F32,workers: U32,
  mask: Tensors.Tensor<F32>,yes: Tensors.Tensor<F32>,no: Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  match status:
    case Tensors.Failed{problem}: Tensors.with_status(~F32,Tensors.Failed{problem},mask)
    case Tensors.Usable{}: values_shape(Shapes.broadcast(yes_shape,no_shape),mask_shape,yes_shape,no_shape,fill,workers,mask,yes,no)

def select(mask: Tensors.Tensor<F32>,yes: Tensors.Tensor<F32>,no: Tensors.Tensor<F32>) -> Tensors.Tensor<F32>:
  Tensors.with_metadata(~F32,~Tensors.Tensor<F32>,mask,mask => mask_shape => mf => mt => mw =>
    Tensors.with_metadata(~F32,~Tensors.Tensor<F32>,yes,yes => yes_shape => fill => yt => yw =>
      Tensors.with_metadata(~F32,~Tensors.Tensor<F32>,no,no => no_shape => nf => nt => nw =>
        status(Tensors.first_failure(mt,Tensors.first_failure(yt,nt)),mask_shape,yes_shape,no_shape,fill,U32.max(mw,U32.max(yw,nw)),mask,yes,no))))
