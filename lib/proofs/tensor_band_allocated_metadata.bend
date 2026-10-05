# Recover metadata from the actual leaf allocation and row descriptor.
# This theorem describes storage ownership; convolution supplies its fields.
import Base
import ../tensor_band.bend as Bands
import ../traversal_partition.bend as Partition
import ../storage_buffer.bend as Storage
import ../tensor.bend as Tensors
import ./storage_buffer_model.bend as Buffer
import ./storage_certificate.bend as Certificate
import ./tensor_band_model.bend as Model
import ./tensor_band_shape_refinement.bend as BandShape
import ./kernel_pipeline_buffer_shape.bend as KernelPipelineBufferShape
import ./kernel_band_buffer_shape.bend as KernelBandBufferShape

type Descriptor is Data:
  Descriptor{geometry: Bands.Geometry,first: U32,rows: U32,count: U32}

def BufferMetadata(context: Descriptor,-buffer: Storage.Buffer<F32>) -> Type:
  Descriptor{+geometry,+first,+rows,+count} = context
  @exact: {KernelPipelineBufferShape.footprint(~F32,buffer) == KernelPipelineBufferShape.footprint(~F32,Tensors.buffer(~F32,0.0,count)) : KernelPipelineBufferShape.Footprint} ->
  @fields: BandShape.Leaf(geometry,first,rows,U32.to_nat(count)) ->
  BandShape.Contents(~F32,geometry,Model.RowBand{Buffer.contents(~F32,buffer),first,rows})

def canonical(+context: Descriptor,+buffer: Buffer.Model<F32>) -> BufferMetadata(context,Buffer.pack(~F32,buffer)):
  Descriptor{+geometry,+first,+rows,+count} = context
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,+certificate}} = buffer
  +buffer = {Buffer.Model{tree,length,Certificate.Certificate{depth,certificate}} : Buffer.Model<F32>}
  exact => fields =>
    +length_exact = Equal.trans(Nat,length,Buffer.content_length(~F32,Buffer.contents(~F32,Buffer.pack(~F32,buffer))),U32.to_nat(count),
      Equal.cong(Buffer.Content<F32>,Nat,Buffer.content_length(~F32),Buffer.content(~F32,buffer),Buffer.contents(~F32,Buffer.pack(~F32,buffer)),
        Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,buffer)),Buffer.content(~F32,buffer),Buffer.packed_contents(~F32,buffer))),
      KernelBandBufferShape.length_canonical(count,buffer,exact))
    %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,buffer)),Buffer.content(~F32,buffer),Buffer.packed_contents(~F32,buffer)) :
      BandShape.Contents(~F32,geometry,Model.RowBand{_,first,rows})
    %Equal.sym(Nat,length,U32.to_nat(count),length_exact) : BandShape.Leaf(geometry,first,rows,_)
    fields

def buffer_metadata(+geometry: Bands.Geometry,+first: U32,+rows: U32,+count: U32,buffer: Storage.Buffer<F32>,
  exact: {KernelPipelineBufferShape.footprint(~F32,buffer) == KernelPipelineBufferShape.footprint(~F32,Tensors.buffer(~F32,0.0,count)) : KernelPipelineBufferShape.Footprint},
  fields: BandShape.Leaf(geometry,first,rows,U32.to_nat(count))) ->
  BandShape.Contents(~F32,geometry,Model.RowBand{Buffer.contents(~F32,buffer),first,rows}):
  +context = {Descriptor{geometry,first,rows,count} : Descriptor}
  Buffer.lift(~F32,~Descriptor,~BufferMetadata,context,buffer,Buffer.observe(~F32,buffer),model => canonical(context,model))(exact,fields)

def first(plan: Bands.Plan) -> U32:
  match plan:
    case Bands.Rows{first,rows}: first
    case Bands.Split{left,right}: 0

def rows(plan: Bands.Plan) -> U32:
  match plan:
    case Bands.Rows{first,rows}: rows
    case Bands.Split{left,right}: 0

def leaf(+geometry: Bands.Geometry,+expected_first: U32,+expected_rows: U32,+count: U32,bands: Bands.Bands(~F32),
  allocated: KernelBandBufferShape.AllocatedLeaf(count,bands),
  +layout: {Model.content_layout(~F32,Model.contents(~F32,bands)) == Bands.Rows{expected_first,expected_rows} : Bands.Plan},
  fields: BandShape.Leaf(geometry,expected_first,expected_rows,U32.to_nat(count))) ->
  BandShape.Contents(~F32,geometry,Model.contents(~F32,bands)):
  match bands:
    case Partition.Leaf{buffer,+actual_first,+actual_rows,details}:
      buffer_metadata(geometry,actual_first,actual_rows,count,buffer,allocated,
        %Equal.sym(U32,actual_first,expected_first,Equal.cong(Bands.Plan,U32,first,Bands.Rows{actual_first,actual_rows},Bands.Rows{expected_first,expected_rows},layout)) : BandShape.Leaf(geometry,_,actual_rows,U32.to_nat(count))
        %Equal.sym(U32,actual_rows,expected_rows,Equal.cong(Bands.Plan,U32,rows,Bands.Rows{actual_first,actual_rows},Bands.Rows{expected_first,expected_rows},layout)) : BandShape.Leaf(geometry,expected_first,_,U32.to_nat(count))
        fields)
    case Partition.Fork{left,right}: match allocated:
