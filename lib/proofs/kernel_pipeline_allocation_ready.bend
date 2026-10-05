# Turn an output's real allocator footprint into content readiness. The
# allocation bound is explicit and must be proved by the shape caller.
import Base
import ../tensor.bend as Tensors
import ./kernel_pipeline_buffer_shape.bend as KernelPipelineBufferShape
import ./storage_buffer_model.bend as Buffer
import ./tensor_allocation_depth.bend as Depth

def from_footprint(~Element: Data,+fill: Element,+count: U32,+model: Buffer.Model<Element>,
  +exact: {KernelPipelineBufferShape.footprint(~Element,Buffer.pack(~Element,model)) == KernelPipelineBufferShape.footprint(~Element,Tensors.buffer(~Element,fill,count)) : KernelPipelineBufferShape.Footprint},
  ready: Depth.Ready<count>) -> Buffer.ContentReady<Element,Buffer.content(~Element,model)>:
  Depth.Ready{depth_ok,room} = ready
  +allocated = {KernelPipelineBufferShape.Footprint{Tensors.depth_for(count),U32.to_nat(count)} : KernelPipelineBufferShape.Footprint}
  +footprint_exact = Equal.trans(KernelPipelineBufferShape.Footprint,KernelPipelineBufferShape.footprint(~Element,Buffer.pack(~Element,model)),KernelPipelineBufferShape.footprint(~Element,Tensors.buffer(~Element,fill,count)),allocated,
    exact,KernelPipelineBufferShape.allocated(~Element,fill,count))
  KernelPipelineBufferShape.content_ready(~Element,model,
    %Equal.sym(KernelPipelineBufferShape.Footprint,KernelPipelineBufferShape.footprint(~Element,Buffer.pack(~Element,model)),allocated,footprint_exact) : {Nat.is_le(KernelPipelineBufferShape.depth(_),30n) == True{} : Bool}
    depth_ok,
    %Equal.sym(KernelPipelineBufferShape.Footprint,KernelPipelineBufferShape.footprint(~Element,Buffer.pack(~Element,model)),allocated,footprint_exact) :
      {Nat.is_le(KernelPipelineBufferShape.length(_),U32.to_nat(U32.shln(1,KernelPipelineBufferShape.depth(_)))) == True{} : Bool}
    room)

def bounded(~Element: Data,+fill: Element,+count: U32,+model: Buffer.Model<Element>,
  exact: {KernelPipelineBufferShape.footprint(~Element,Buffer.pack(~Element,model)) == KernelPipelineBufferShape.footprint(~Element,Tensors.buffer(~Element,fill,count)) : KernelPipelineBufferShape.Footprint},
  within: {U32.is_le(count,1073741824) == True{} : Bool}) -> Buffer.ContentReady<Element,Buffer.content(~Element,model)>:
  from_footprint(~Element,fill,count,model,exact,Depth.bounded(count,within))
