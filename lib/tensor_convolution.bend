# CHW / NCHW entry. Geometry and allocation checks finish before planning.
# Every image uses the same reordered weights; the batch is processed in order.
import Base
import ./tensor.bend as Tensors
import ./storage_buffer.bend as Storage
import ./storage_clone.bend as Clone
import ./storage_copy.bend as Copy
import ./kernel_shape.bend as KernelShape
import ./kernel_pipeline.bend as Pipeline
import ./kernel_gemm_buffer.bend as GemmBuffer
import ./kernel_tile.bend as Tile
import ./kernel_planning.bend as KernelPlanning
import ./default_kernel_cost.bend as DefaultCost
import ./kernel_cost.bend as Cost
import ./default_machine_limits.bend as Default
import ./machine_limits.bend as Machine
import ./proofs/storage_certificate.bend as Certificate
import ./tensor_band.bend as Bands
import ./tensor_band_window.bend as BandWindow
import ./kernel_band_pipeline.bend as BandPipeline
import ./kernel_band_planning.bend as BandPlanning
import ./math_activation.bend as Activations

type InputShape is Data:
  InputShape{batch: U32,batched: Bool,channels: U32,height: U32,width: U32}

type Geometry is Data:
  Geometry{batch: U32,batched: Bool,dimensions: KernelShape.Dimensions}

type DenseArguments is Data:
  DenseArguments{image_count: U32,work: Pipeline.Work,input_size: U32,output_size: U32,output_extents: List<&2,U32>,selected_depth: U32}

def input_shape(shape: List<&2,U32>) -> Maybe<&2,InputShape>:
  match shape:
    case [channels,height,width]: Some{InputShape{1,False{},channels,height,width}}
    case [batch,channels,height,width]: Some{InputShape{batch,True{},channels,height,width}}
    case shape: None{}

def matched(matches: Bool,geometry: Geometry) -> Maybe<&2,Geometry>:
  match matches:
    case True{}: Some{geometry}
    case False{}: None{}

def geometry(input: Maybe<&2,InputShape>,weights: List<&2,U32>,bias: List<&2,U32>,stride: U32,padding: U32) -> Maybe<&2,Geometry>:
  match input weights bias:
    case Some{InputShape{batch,batched,+channels,height,width}} Con{+rows,Con{weight_channels,Con{+kernel,Con{kernel_width,Nil{}}}}} Con{bias_channels,Nil{}}:
      matched(U32.is_eq(channels,weight_channels) && U32.is_eq(kernel,kernel_width) && U32.is_eq(rows,bias_channels),
        Geometry{batch,batched,KernelShape.Dimensions{U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding)}})
    case input weights bias: None{}

def output_shape(batched: Bool,batch: U32,rows: U32,height: U32,width: U32) -> List<&2,U32>:
  match batched:
    case False{}: [rows,height,width]
    case True{}: [batch,rows,height,width]

def selected_depth(plan: KernelPlanning.Plan) -> U32:
  KernelPlanning.Plan{packed,depth,workspace,time} = plan
  depth

def configured_depth(work: Pipeline.Work,input_capacity: U32,bias_capacity: U32,workers: U32,+activation: Activations.Activation,+config: Cost.Configuration) -> U32:
  Pipeline.Work{rows,height,width,+kernel,stride,padding,output_width,positions,+inner} = work
  # As in the pipeline split, input channels are the reduction length over kernel taps.
  +geometry = {KernelPlanning.Geometry{(inner / (kernel * kernel) : U32),height,width,kernel,stride,padding,output_width} : KernelPlanning.Geometry}
  selected_depth(KernelPlanning.select(KernelPlanning.Work{positions,inner,rows,KernelPlanning.Owners{input_capacity,bias_capacity},geometry,activation},workers,None{},config))

# The input tensor's workers bound the split, as a thread pool bounds PyTorch's parallel_for.
def depth(work: Pipeline.Work,input_capacity: U32,bias_capacity: U32,workers: U32,+activation: Activations.Activation) -> U32:
  configured_depth(work,input_capacity,bias_capacity,workers,activation,DefaultCost.configuration())

def copied_output(pair: Storage.Buffer<F32> & Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  (source,output) = pair
  output

def image(~custom: F32 -> F32,+activation: Activations.Activation,depth: U32,work: Pipeline.Work,index: U32,+input_size: U32,input: Storage.Buffer<F32>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>) ->
  Storage.Buffer<F32> & Storage.Buffer<F32>:
  Storage.bind_pair(~Storage.Buffer<F32>,~Storage.Buffer<F32>,~(Storage.Buffer<F32> & Storage.Buffer<F32>),
    Copy.range(~F32,input_size,(index * input_size : U32),0,input,Tensors.buffer(~F32,0.0,input_size)),source => image =>
      (source,Pipeline.run_packed(~custom,activation,depth,work,image,weights,bias)))

def batch(~custom: F32 -> F32,+activation: Activations.Activation,count: Nat,+index: U32,+depth: U32,+work: Pipeline.Work,+input_size: U32,+output_size: U32,
  input: Storage.Buffer<F32>,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,output: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  match count:
    case 0n: output
    case 1n:
      Storage.bind_pair(~Storage.Buffer<F32>,~Storage.Buffer<F32>,~Storage.Buffer<F32>,
        image(~custom,activation,depth,work,index,input_size,input,weights,bias),source => result =>
          copied_output(Copy.range(~F32,output_size,0,(index * output_size : U32),result,output)))
    case 2n+rest:
      Storage.bind_pair(~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<Tile.Tile8>,~Storage.Buffer<F32>,Clone.clone(~Tile.Tile8,weights),current_weights => remaining_weights =>
        Storage.bind_pair(~Storage.Buffer<F32>,~Storage.Buffer<F32>,~Storage.Buffer<F32>,Clone.clone(~F32,bias),current_bias => remaining_bias =>
          Storage.bind_pair(~Storage.Buffer<F32>,~Storage.Buffer<F32>,~Storage.Buffer<F32>,
            image(~custom,activation,depth,work,index,input_size,input,current_weights,current_bias),source => result =>
              batch(~custom,activation,1n+rest,(index + 1 : U32),depth,work,input_size,output_size,source,remaining_weights,remaining_bias,
                copied_output(Copy.range(~F32,output_size,0,(index * output_size : U32),result,output))))))

def images_selected(~custom: F32 -> F32,+activation: Activations.Activation,empty: Bool,single: Bool,+count: U32,depth: U32,work: Pipeline.Work,input_size: U32,+output_size: U32,
  input: Storage.Buffer<F32>,weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  match empty:
    case True{}: Tensors.buffer(~F32,0.0,0)
    case False{}:
      match single:
        case True{}: Pipeline.run(~custom,activation,depth,work,input,weights,bias)
        case False{}:
          Pipeline.Work{+rows,h,w,k,s,p,ow,positions,+inner} = work
          batch(~custom,activation,U32.to_nat(count),0,depth,Pipeline.Work{rows,h,w,k,s,p,ow,positions,inner},input_size,output_size,
            input,GemmBuffer.reorder(rows,inner,weights),bias,Tensors.buffer(~F32,0.0,(count * output_size : U32)))

def images(~custom: F32 -> F32,+activation: Activations.Activation,+count: U32,depth: U32,work: Pipeline.Work,input_size: U32,+output_size: U32,
  input: Storage.Buffer<F32>,weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>) -> Storage.Buffer<F32>:
  images_selected(~custom,activation,U32.is_eq(count,0),U32.is_eq(count,1),count,depth,work,input_size,output_size,input,weights,bias)

def execute_dense_at_depth(~custom: F32 -> F32,+activation: Activations.Activation,+image_count: U32,+selected_depth: U32,+work: Pipeline.Work,+input_size: U32,+output_size: U32,+output_extents: List<&2,U32>,
  input: Storage.Buffer<F32>,weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,+workers: U32) -> Tensors.Tensor<F32>:
  Tensors.Tensor{images(~custom,activation,image_count,selected_depth,work,input_size,output_size,input,weights,bias),
    output_extents,0.0,Tensors.Ready{},workers}

def dense_arguments(geometry: Geometry,+bias_depth: Nat,+workers: U32,+activation: Activations.Activation) -> DenseArguments:
  Geometry{+count,batched,KernelShape.Dimensions{+ci,+rows,+height,+width,+kernel,+stride,+padding}} = geometry
  +oh = U32.from_nat(KernelShape.axis_size(height,kernel,stride,padding))
  +ow = U32.from_nat(KernelShape.axis_size(width,kernel,stride,padding))
  +input_size = U32.from_nat(Nat.mul(ci,Nat.mul(height,width)))
  +output_size = (U32.from_nat(rows) * oh * ow : U32)
  +work = {Pipeline.Work{U32.from_nat(rows),U32.from_nat(height),U32.from_nat(width),U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),ow,(oh * ow : U32),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel)))} : Pipeline.Work}
  DenseArguments{count,work,input_size,output_size,output_shape(batched,count,U32.from_nat(rows),oh,ow),
    depth(work,U32.from_nat(KernelPlanning.capacity(input_size)),U32.shln(1,bias_depth),workers,activation)}

def execute_dense_arguments(~custom: F32 -> F32,+activation: Activations.Activation,arguments: DenseArguments,input: Storage.Buffer<F32>,weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,+workers: U32) -> Tensors.Tensor<F32>:
  DenseArguments{count,work,input_size,output_size,output_extents,selected_depth} = arguments
  execute_dense_at_depth(~custom,activation,count,selected_depth,work,input_size,output_size,output_extents,input,weights,bias,workers)

def execute_dense(~custom: F32 -> F32,+activation: Activations.Activation,geometry: Geometry,input: Storage.Buffer<F32>,weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,+workers: U32) -> Tensors.Tensor<F32>:
  Storage.Buffer{bias,bias_length,Certificate.Certificate{+bias_depth,bias_equation}} = bias
  execute_dense_arguments(~custom,activation,dense_arguments(geometry,bias_depth,workers,activation),input,weights,Storage.Buffer{bias,bias_length,Certificate.Certificate{bias_depth,bias_equation}},workers)

def band_context(geometry: Geometry,parallel: Bool,+activation: Activations.Activation) -> BandPipeline.Context:
  Geometry{count,batched,KernelShape.Dimensions{+ci,rows,+height,+width,+kernel,+stride,+padding}} = geometry
  BandPipeline.Context{BandWindow.Geometry{Bands.Geometry{count,U32.from_nat(ci),U32.from_nat(height),U32.from_nat(width)},U32.from_nat(kernel),U32.from_nat(stride),U32.from_nat(padding),
    U32.from_nat(KernelShape.axis_size(height,kernel,stride,padding)),U32.from_nat(KernelShape.axis_size(width,kernel,stride,padding))},U32.from_nat(rows),U32.from_nat(Nat.mul(ci,Nat.mul(kernel,kernel))),parallel,activation}

def band_work(context: BandPipeline.Context,bias_capacity: U32) -> KernelPlanning.Work:
  BandPipeline.Context{BandWindow.Geometry{Bands.Geometry{batch,+channels,+height,+width},kernel,stride,padding,oh,+ow},rows,inner,parallel,activation} = context
  KernelPlanning.Work{(oh * ow : U32),inner,rows,KernelPlanning.Owners{U32.from_nat(KernelPlanning.capacity((channels * height * width : U32))),bias_capacity},
    KernelPlanning.Geometry{channels,height,width,kernel,stride,padding,ow},activation}

def context_window(context: BandPipeline.Context) -> BandWindow.Geometry:
  BandPipeline.Context{window,rows,inner,parallel,activation} = context
  window

def band_output(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: Geometry,parallel: Bool,input: Bands.Bands(~F32),weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,workers: U32) -> Tensors.Tensor<F32>:
  Geometry{+count,batched,KernelShape.Dimensions{ci,+rows,height,width,+kernel,+stride,+padding}} = geometry
  +oh = U32.from_nat(KernelShape.axis_size(height,kernel,stride,padding))
  +ow = U32.from_nat(KernelShape.axis_size(width,kernel,stride,padding))
  Tensors.Banded{BandPipeline.execute(~custom,band_context(geometry,parallel,activation),input,weights,bias),
    Bands.Geometry{count,U32.from_nat(rows),oh,ow},output_shape(batched,count,U32.from_nat(rows),oh,ow),0.0,Tensors.Ready{},workers}

def start_bands(~custom: F32 -> F32,+activation: Activations.Activation,depth: U32,parallel: Bool,+geometry: Geometry,input: Storage.Buffer<F32>,weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,workers: U32) -> Tensors.Tensor<F32>:
  Geometry{count,batched,KernelShape.Dimensions{ci,rows,h,w,k,s,p}} = geometry
  band_output(~custom,activation,geometry,parallel,Bands.split(~F32,U32.to_nat(depth),Bands.Geometry{count,U32.from_nat(ci),U32.from_nat(h),U32.from_nat(w)},0.0,input),weights,bias,workers)

def initial_layout_selected(~custom: F32 -> F32,+activation: Activations.Activation,zero: Bool,depth: U32,parallel: Bool,+geometry: Geometry,input: Storage.Buffer<F32>,weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,workers: U32) -> Tensors.Tensor<F32>:
  match zero:
    case True{}: Tensors.with_workers(~F32,execute_dense(~custom,activation,geometry,input,weights,bias,1),workers)
    case False{}: start_bands(~custom,activation,depth,parallel,geometry,input,weights,bias,workers)

def initial_layout(~custom: F32 -> F32,+activation: Activations.Activation,plan: BandPlanning.Plan,+geometry: Geometry,input: Storage.Buffer<F32>,weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,workers: U32) -> Tensors.Tensor<F32>:
  BandPlanning.Plan{+depth,parallel,time} = plan
  initial_layout_selected(~custom,activation,U32.is_eq(depth,0),depth,parallel,geometry,input,weights,bias,workers)

def execute_many(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: Geometry,input: Storage.Buffer<F32>,weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,+workers: U32) -> Tensors.Tensor<F32>:
  Storage.Buffer{bias,length,Certificate.Certificate{+depth,equation}} = bias
  +context = band_context(geometry,True{},activation)
  initial_layout(~custom,activation,BandPlanning.select(band_work(context,U32.shln(1,depth)),context_window(context),workers,DefaultCost.configuration()),
    geometry,input,weights,Storage.Buffer{bias,length,Certificate.Certificate{depth,equation}},workers)

# Padding-only output windows and an empty batch use the existing dense path.
# This is a representation choice, not a restriction on valid convolution shapes.
def uses_band_windows(geometry: Geometry) -> Bool:
  Geometry{count,batched,KernelShape.Dimensions{ci,rows,h,w,k,s,p}} = geometry
  U32.is_lt(0,count) && Nat.is_lt(p,k)

def execute_layout(~custom: F32 -> F32,+activation: Activations.Activation,band_windows: Bool,+geometry: Geometry,input: Storage.Buffer<F32>,
  weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,workers: U32) -> Tensors.Tensor<F32>:
  match band_windows:
    case False{}: execute_dense(~custom,activation,geometry,input,weights,bias,workers)
    case True{}: execute_many(~custom,activation,geometry,input,weights,bias,workers)

def execute_selected(~custom: F32 -> F32,+activation: Activations.Activation,single_worker: Bool,+geometry: Geometry,input: Storage.Buffer<F32>,weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,workers: U32) -> Tensors.Tensor<F32>:
  match single_worker:
    case True{}: execute_dense(~custom,activation,geometry,input,weights,bias,1)
    case False{}: execute_layout(~custom,activation,uses_band_windows(geometry),geometry,input,weights,bias,workers)

def execute(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: Geometry,input: Storage.Buffer<F32>,weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,+workers: U32) -> Tensors.Tensor<F32>:
  execute_selected(~custom,activation,U32.is_eq(workers,1),geometry,input,weights,bias,workers)

def machine(~custom: F32 -> F32,+activation: Activations.Activation,supported: Bool,geometry: Geometry,input: Storage.Buffer<F32>,weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,+shape: List<&2,U32>,fill: F32,+workers: U32) -> Tensors.Tensor<F32>:
  match supported:
    case True{}: execute(~custom,activation,geometry,input,weights,bias,workers)
    case False{}: Tensors.Tensor{input,shape,fill,Tensors.Failed{Tensors.Problem{"conv2d",Tensors.InvalidShape{shape}}},workers}

def machine_support(geometry: Geometry,+limits: Machine.Limits) -> Bool:
  Geometry{+count,batched,+dimensions} = geometry
  KernelShape.Dimensions{+ci,+rows,+h,+w,+k,+s,+p} = dimensions
  KernelShape.supports_packed(limits,dimensions,Nat.mul(ci,Nat.mul(h,w)),Nat.mul(rows,Nat.mul(ci,Nat.mul(k,k)))) &&
    Nat.is_le(Nat.mul(U32.to_nat(count),Nat.mul(rows,Nat.mul(KernelShape.axis_size(h,k,s,p),KernelShape.axis_size(w,k,s,p)))),U32.to_nat(Machine.capacity(limits)))

def supported(~custom: F32 -> F32,+activation: Activations.Activation,valid: Bool,+geometry: Geometry,input: Storage.Buffer<F32>,weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,+shape: List<&2,U32>,fill: F32,+workers: U32) -> Tensors.Tensor<F32>:
  match valid:
    case False{}: Tensors.Tensor{input,shape,fill,Tensors.Failed{Tensors.Problem{"conv2d",Tensors.InvalidArgument{shape}}},workers}
    case True{}:
      machine(~custom,activation,machine_support(geometry,Default.limits()),geometry,input,weights,bias,shape,fill,workers)

def checked(~custom: F32 -> F32,+activation: Activations.Activation,geometry: Maybe<&2,Geometry>,input: Storage.Buffer<F32>,weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,+shape: List<&2,U32>,fill: F32,+workers: U32) -> Tensors.Tensor<F32>:
  match geometry:
    case None{}: Tensors.Tensor{input,shape,fill,Tensors.Failed{Tensors.Problem{"conv2d",Tensors.InvalidArgument{shape}}},workers}
    case Some{+geometry}:
      Geometry{+count,batched,+dimensions} = geometry
      KernelShape.Dimensions{+ci,rows,+h,+w,k,s,p} = dimensions
      Storage.Buffer{input,+input_length,Certificate.Certificate{+input_depth,input_equation}} = input
      Storage.Buffer{weights,+weight_length,Certificate.Certificate{+weight_depth,weight_equation}} = weights
      Storage.Buffer{bias,+bias_length,Certificate.Certificate{+bias_depth,bias_equation}} = bias
      supported(~custom,activation,
        KernelShape.valid(dimensions,Nat.mul(ci,Nat.mul(h,w)),weight_length,bias_length) &&
        Nat.is_eq(input_length,Nat.mul(U32.to_nat(count),Nat.mul(ci,Nat.mul(h,w)))) &&
        Nat.is_le(input_length,U32.to_nat(U32.shln(1,input_depth))) &&
        Nat.is_le(weight_length,U32.to_nat(U32.shln(1,weight_depth))) && Nat.is_le(bias_length,U32.to_nat(U32.shln(1,bias_depth))),geometry,
        Storage.Buffer{input,input_length,Certificate.Certificate{input_depth,input_equation}},
        Storage.Buffer{weights,weight_length,Certificate.Certificate{weight_depth,weight_equation}},
        Storage.Buffer{bias,bias_length,Certificate.Certificate{bias_depth,bias_equation}},shape,fill,workers)

def status(~custom: F32 -> F32,+activation: Activations.Activation,status: Tensors.Status,input: Storage.Buffer<F32>,+input_extents: List<&2,U32>,fill: F32,
  weights: Storage.Buffer<F32>,weight_shape: List<&2,U32>,bias: Storage.Buffer<F32>,bias_shape: List<&2,U32>,stride: U32,padding: U32,+workers: U32) -> Tensors.Tensor<F32>:
  match status:
    case Tensors.Failed{problem}: Tensors.Tensor{input,input_extents,fill,Tensors.Failed{problem},workers}
    case Tensors.Ready{}: checked(~custom,activation,geometry(input_shape(input_extents),weight_shape,bias_shape,stride,padding),input,weights,bias,input_extents,fill,workers)

def replanned(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: Geometry,selection: BandPlanning.LayoutPlan,input: Bands.Bands(~F32),weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,workers: U32) -> Tensors.Tensor<F32>:
  BandPlanning.LayoutPlan{layout,+parallel,time,repartition} = selection
  band_output(~custom,activation,geometry,parallel,Bands.align(~F32,BandWindow.input_geometry(context_window(band_context(geometry,parallel,Activations.Identity{}))),0.0,layout,input),weights,bias,workers)

def existing_layout(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: Geometry,weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,+workers: U32,
  owners: Bands.Bands(~F32) & Bands.Plan) -> Tensors.Tensor<F32>:
  Storage.Buffer{bias,length,Certificate.Certificate{+depth,equation}} = bias
  (input,plan) = owners
  +context = band_context(geometry,True{},activation)
  replanned(~custom,activation,geometry,BandPlanning.replan(plan,band_work(context,U32.shln(1,depth)),context_window(context),workers,DefaultCost.configuration()),
    input,weights,Storage.Buffer{bias,length,Certificate.Certificate{depth,equation}},workers)

def execute_existing_layout(~custom: F32 -> F32,+activation: Activations.Activation,band_windows: Bool,+geometry: Geometry,+layout: Bands.Geometry,
  input: Bands.Bands(~F32),weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,workers: U32) -> Tensors.Tensor<F32>:
  match band_windows:
    case True{}: existing_layout(~custom,activation,geometry,weights,bias,workers,Bands.plan(~F32,input))
    case False{}:
      Bands.Geometry{+images,+channels,+height,+width} = layout
      execute_dense(~custom,activation,geometry,
        Bands.gather(~F32,layout,input,Tensors.buffer(~F32,0.0,U32.from_nat(Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))))))),
        weights,bias,workers)

def band_machine(~custom: F32 -> F32,+activation: Activations.Activation,supported: Bool,+geometry: Geometry,layout: Bands.Geometry,input: Bands.Bands(~F32),
  weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,+shape: List<&2,U32>,fill: F32,workers: U32) -> Tensors.Tensor<F32>:
  match supported:
    case True{}: execute_existing_layout(~custom,activation,uses_band_windows(geometry),geometry,layout,input,weights,bias,workers)
    case False{}: Tensors.Banded{input,layout,shape,fill,Tensors.Failed{Tensors.Problem{"conv2d",Tensors.InvalidShape{shape}}},workers}

def band_supported(~custom: F32 -> F32,+activation: Activations.Activation,valid: Bool,+geometry: Geometry,layout: Bands.Geometry,input: Bands.Bands(~F32),
  weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,+shape: List<&2,U32>,fill: F32,workers: U32) -> Tensors.Tensor<F32>:
  match valid:
    case False{}: Tensors.Banded{input,layout,shape,fill,Tensors.Failed{Tensors.Problem{"conv2d",Tensors.InvalidArgument{shape}}},workers}
    case True{}:
      band_machine(~custom,activation,machine_support(geometry,Default.limits()),geometry,layout,input,weights,bias,shape,fill,workers)

def band_inspected(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: Geometry,+layout: Bands.Geometry,weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,shape: List<&2,U32>,fill: F32,workers: U32,
  inspected: Bands.Bands(~F32) & Bands.Inspection) -> Tensors.Tensor<F32>:
  Geometry{+count,batched,+dimensions} = geometry
  KernelShape.Dimensions{ci,rows,+h,w,k,s,p} = dimensions
  Bands.Geometry{batch,channels,height,width} = layout
  Storage.Buffer{weights,+weight_length,Certificate.Certificate{+weight_depth,weight_equation}} = weights
  Storage.Buffer{bias,+bias_length,Certificate.Certificate{+bias_depth,bias_equation}} = bias
  (input,Bands.Inspection{valid,first,end,leaves}) = inspected
  band_supported(~custom,activation,valid && U32.is_zero(first) && Nat.is_eq(U32.to_nat(end),Nat.mul(U32.to_nat(count),h)) &&
    U32.is_eq(batch,count) && Nat.is_eq(U32.to_nat(channels),ci) && Nat.is_eq(U32.to_nat(height),h) && Nat.is_eq(U32.to_nat(width),w) &&
    KernelShape.valid(dimensions,Nat.mul(ci,Nat.mul(h,w)),weight_length,bias_length) &&
    Nat.is_le(weight_length,U32.to_nat(U32.shln(1,weight_depth))) && Nat.is_le(bias_length,U32.to_nat(U32.shln(1,bias_depth))),
    geometry,layout,input,Storage.Buffer{weights,weight_length,Certificate.Certificate{weight_depth,weight_equation}},
    Storage.Buffer{bias,bias_length,Certificate.Certificate{bias_depth,bias_equation}},shape,fill,workers)

def band_checked(~custom: F32 -> F32,+activation: Activations.Activation,geometry: Maybe<&2,Geometry>,+layout: Bands.Geometry,input: Bands.Bands(~F32),
  weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,+shape: List<&2,U32>,fill: F32,workers: U32) -> Tensors.Tensor<F32>:
  match geometry:
    case None{}: Tensors.Banded{input,layout,shape,fill,Tensors.Failed{Tensors.Problem{"conv2d",Tensors.InvalidArgument{shape}}},workers}
    case Some{geometry}: band_inspected(~custom,activation,geometry,layout,weights,bias,shape,fill,workers,Bands.inspect(~F32,layout,input))

def band_status(~custom: F32 -> F32,+activation: Activations.Activation,status: Tensors.Status,layout: Bands.Geometry,input: Bands.Bands(~F32),+shape: List<&2,U32>,fill: F32,
  weights: Storage.Buffer<F32>,weight_shape: List<&2,U32>,bias: Storage.Buffer<F32>,bias_shape: List<&2,U32>,stride: U32,padding: U32,workers: U32) -> Tensors.Tensor<F32>:
  match status:
    case Tensors.Failed{problem}: Tensors.Banded{input,layout,shape,fill,Tensors.Failed{problem},workers}
    case Tensors.Ready{}: band_checked(~custom,activation,geometry(input_shape(shape),weight_shape,bias_shape,stride,padding),layout,input,weights,bias,shape,fill,workers)

def input_layout(~custom: F32 -> F32,+activation: Activations.Activation,input: Tensors.Tensor<F32>,weights: Storage.Buffer<F32>,weight_shape: List<&2,U32>,bias: Storage.Buffer<F32>,bias_shape: List<&2,U32>,
  parameter_status: Tensors.Status,parameter_workers: U32,stride: U32,padding: U32) -> Tensors.Tensor<F32>:
  match input:
    case Tensors.Tensor{input,shape,fill,input_status,workers}:
      status(~custom,activation,Tensors.first_failure(input_status,parameter_status),input,shape,fill,weights,weight_shape,bias,bias_shape,stride,padding,U32.max(workers,parameter_workers))
    case Tensors.Banded{input,layout,shape,fill,input_status,workers}:
      band_status(~custom,activation,Tensors.first_failure(input_status,parameter_status),layout,input,shape,fill,weights,weight_shape,bias,bias_shape,stride,padding,U32.max(workers,parameter_workers))

def input_selected(~custom: F32 -> F32,+activation: Activations.Activation,dense: Bool,input: Tensors.Tensor<F32>,weights: Storage.Buffer<F32>,weight_shape: List<&2,U32>,bias: Storage.Buffer<F32>,bias_shape: List<&2,U32>,
  +parameter_status: Tensors.Status,+parameter_workers: U32,+stride: U32,+padding: U32) -> Tensors.Tensor<F32>:
  match dense:
    case False{}: input_layout(~custom,activation,input,weights,weight_shape,bias,bias_shape,parameter_status,parameter_workers,stride,padding)
    case True{}:
      Tensors.with_dense(~F32,~Tensors.Tensor<F32>,input,input => shape => fill => input_status => workers =>
        status(~custom,activation,Tensors.first_failure(input_status,parameter_status),input,shape,fill,weights,weight_shape,bias,bias_shape,stride,padding,U32.max(workers,parameter_workers)))

# Parameter representations select one public path before any element work.
# Dense parameters retain the existing incoming-band convolution leaf path.
def conv2d_selected(~custom: F32 -> F32,+activation: Activations.Activation,+dense: Bool,input: Tensors.Tensor<F32>,weights: Tensors.Tensor<F32>,bias: Tensors.Tensor<F32>,+stride: U32,+padding: U32) -> Tensors.Tensor<F32>:
  Tensors.with_dense(~F32,~Tensors.Tensor<F32>,weights,weights => weight_shape => weight_fill => weight_status => weight_workers =>
    Tensors.with_dense(~F32,~Tensors.Tensor<F32>,bias,bias => bias_shape => bias_fill => bias_status => bias_workers =>
      input_selected(~custom,activation,dense,input,weights,weight_shape,bias,bias_shape,Tensors.first_failure(weight_status,bias_status),U32.max(weight_workers,bias_workers),stride,padding)))

def conv2d_bias_layout(~custom: F32 -> F32,+activation: Activations.Activation,dense: Bool,input: Tensors.Tensor<F32>,weights: Tensors.Tensor<F32>,bias: Tensors.Tensor<F32>,stride: U32,padding: U32) -> Tensors.Tensor<F32>:
  match bias:
    case Tensors.Tensor{storage,shape,fill,status,workers}:
      conv2d_selected(~custom,activation,dense,input,weights,Tensors.Tensor{storage,shape,fill,status,workers},stride,padding)
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}:
      conv2d_selected(~custom,activation,True{},input,weights,Tensors.Banded{bands,geometry,shape,fill,status,workers},stride,padding)

def conv2d(~custom: F32 -> F32,+activation: Activations.Activation,input: Tensors.Tensor<F32>,weights: Tensors.Tensor<F32>,bias: Tensors.Tensor<F32>,stride: U32,padding: U32) -> Tensors.Tensor<F32>:
  match weights:
    case Tensors.Tensor{storage,shape,fill,status,workers}:
      conv2d_bias_layout(~custom,activation,False{},input,Tensors.Tensor{storage,shape,fill,status,workers},bias,stride,padding)
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}:
      conv2d_bias_layout(~custom,activation,True{},input,Tensors.Banded{bands,geometry,shape,fill,status,workers},bias,stride,padding)
