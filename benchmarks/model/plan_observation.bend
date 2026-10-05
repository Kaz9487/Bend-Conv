# Diagnostic metadata replay: no tensor allocation, convolution or timing.
# Choices and costs are read from the same public caller/planners as the model.
import Base
import ../../lib/tensor_band.bend as Bands
import ../../lib/tensor_band_window.bend as BandWindow
import ../../lib/tensor_band_pool.bend as BandPool
import ../../lib/tensor_band_concat.bend as BandConcat
import ../../lib/tensor_shape.bend as Shapes
import ../../lib/tensor_layout.bend as Layout
import ../../lib/tensor_operations.bend as Operations
import ../../lib/tensor_view.bend as Views
import ../../lib/tensor_window.bend as Window
import ../../lib/tensor_access.bend as Access
import ../../lib/tensor_band_schedule.bend as BandSchedule
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../../lib/kernel_planning.bend as KernelPlanning
import ../../lib/kernel_band_planning.bend as BandPlanning
import ../../lib/default_kernel_cost.bend as DefaultCost
import ../../lib/math_activation.bend as Activations
import ../../lib/storage_buffer.bend as Storage
import ../../lib/tensor.bend as Tensors
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/default_machine_limits.bend as Default

type Receipt is Data:
  Receipt{layout: Maybe<&2,Bands.Plan>,route: String,depth: U32,parallel: Bool,estimate: Maybe<&2,F32>,repartition: Maybe<&2,F32>,gather: Bool}

def dense() -> Receipt:
  Receipt{None{},"dense",0,False{},None{},None{},False{}}

def layout(receipt: Receipt) -> Maybe<&2,Bands.Plan>:
  Receipt{layout,route,depth,parallel,estimate,repartition,gather} = receipt
  layout

def output_rows(rows: BandWindow.RowWindow) -> Bands.Plan:
  BandWindow.RowWindow{BandWindow.Rows{first,count},above,below,window} = rows
  Bands.Rows{first,count}

def output_plan(plan: Bands.Plan,+window: BandWindow.Geometry) -> Bands.Plan:
  match plan:
    case Bands.Rows{first,count}: output_rows(BandWindow.window(window,first,count))
    case Bands.Split{left,right}: Bands.Split{output_plan(left,window),output_plan(right,window)}

def selected_initial(+window: BandWindow.Geometry,selection: BandPlanning.Plan) -> Receipt:
  BandWindow.Geometry{Bands.Geometry{+images,channels,+height,width},k,s,p,oh,ow} = window
  BandPlanning.Plan{depth,parallel,time} = selection
  match depth:
    case 0: Receipt{None{},"dense_initial",0,False{},Some{time},Some{0.0},False{}}
    case +depth:
      Receipt{Some{output_plan(Bands.image_plan(U32.to_nat(images),0,U32.to_nat(depth),height),window)},"band_initial",depth,parallel,Some{time},Some{0.0},False{}}

def selected_existing(+window: BandWindow.Geometry,selection: BandPlanning.LayoutPlan) -> Receipt:
  BandPlanning.LayoutPlan{+plan,parallel,time,repartition} = selection
  Receipt{Some{output_plan(plan,window)},"band_retained_or_repartitioned",0,parallel,Some{time},Some{repartition},False{}}

def dense_selected(+images: U32,gather: Bool,selection: KernelPlanning.Plan) -> Receipt:
  KernelPlanning.Plan{packed,+depth,workspace,time} = selection
  Receipt{None{},"dense_convolution",depth,U32.is_lt(0,depth),Bool.pick(Maybe<&2,F32>,U32.is_eq(images,1),Some{time},None{}),None{},gather}

def dense_convolution(+geometry: TensorConvolution.Geometry,+work: KernelPlanning.Work,workers: U32,gather: Bool) -> Receipt:
  TensorConvolution.Geometry{images,batched,dimensions} = geometry
  dense_selected(images,gather,KernelPlanning.select(work,workers,None{},DefaultCost.configuration()))

def existing_allowed(allowed: Bool,+geometry: TensorConvolution.Geometry,+work: KernelPlanning.Work,+window: BandWindow.Geometry,+workers: U32,plan: Bands.Plan) -> Receipt:
  match allowed:
    case True{}: selected_existing(window,BandPlanning.replan(plan,work,window,workers,DefaultCost.configuration()))
    case False{}: dense_convolution(geometry,work,workers,True{})

def initial_allowed(allowed: Bool,+geometry: TensorConvolution.Geometry,+work: KernelPlanning.Work,+window: BandWindow.Geometry,+workers: U32) -> Receipt:
  match allowed:
    case True{}: selected_initial(window,BandPlanning.select(work,window,workers,DefaultCost.configuration()))
    case False{}: dense_convolution(geometry,work,workers,False{})

def convolution_layout(previous: Maybe<&2,Bands.Plan>,+workers: U32,+geometry: TensorConvolution.Geometry,+work: KernelPlanning.Work,+window: BandWindow.Geometry) -> Receipt:
  match previous:
    case Some{plan}: existing_allowed(TensorConvolution.uses_band_windows(geometry),geometry,work,window,workers,plan)
    case None{}:
      match workers:
        case 1: dense_convolution(geometry,work,1,False{})
        case +workers: initial_allowed(TensorConvolution.uses_band_windows(geometry),geometry,work,window,workers)

def convolution(+geometry: TensorConvolution.Geometry,+activation: Activations.Activation,+workers: U32,+bias_count: U32,previous: Receipt) -> Receipt:
  +context = TensorConvolution.band_context(geometry,True{},activation)
  convolution_layout(layout(previous),workers,geometry,TensorConvolution.band_work(context,U32.shln(1,Storage.depth_for(bias_count))),TensorConvolution.context_window(context))

def convolution_valid(valid: Bool,+geometry: TensorConvolution.Geometry,activation: Activations.Activation,workers: U32,bias_count: U32,previous: Receipt) -> Receipt:
  match valid:
    case True{}: convolution(geometry,activation,workers,bias_count,previous)
    case False{}: Receipt{None{},"invalid_convolution",0,False{},None{},None{},False{}}

def convolution_geometry(parsed: Maybe<&2,TensorConvolution.Geometry>,+weight_count: U32,+bias_count: U32,activation: Activations.Activation,workers: U32,previous: Receipt) -> Receipt:
  match parsed:
    case None{}: Receipt{None{},"invalid_geometry",0,False{},None{},None{},False{}}
    case Some{+geometry}:
      TensorConvolution.Geometry{images,batched,+dimensions} = geometry
      KernelShape.Dimensions{+ci,co,+h,+w,k,s,p} = dimensions
      convolution_valid(KernelShape.valid(dimensions,Nat.mul(ci,Nat.mul(h,w)),U32.to_nat(weight_count),U32.to_nat(bias_count)) && TensorConvolution.machine_support(geometry,Default.limits()),geometry,activation,workers,bias_count,previous)

def checked_convolution(input_shape: List<&2,U32>,weight_shape: List<&2,U32>,bias_shape: List<&2,U32>,stride: U32,padding: U32,
  weight_count: U32,bias_count: U32,activation: Activations.Activation,workers: U32,previous: Receipt) -> Receipt:
  convolution_geometry(TensorConvolution.geometry(TensorConvolution.input_shape(input_shape),weight_shape,bias_shape,stride,padding),weight_count,bias_count,activation,workers,previous)

def combined_plan(+geometry: Bands.Geometry,+workers: U32,left: Maybe<&2,Bands.Plan>,right: Maybe<&2,Bands.Plan>) -> Receipt:
  match left right:
    case None{} None{}: dense()
    case Some{+plan} None{}: Receipt{Some{plan},"binary_align_dense",0,BandSchedule.parallel(geometry,workers,12.0,1.0,plan),None{},None{},False{}}
    case None{} Some{+plan}: Receipt{Some{plan},"binary_align_dense",0,BandSchedule.parallel(geometry,workers,12.0,1.0,plan),None{},None{},False{}}
    case Some{left} Some{right}:
      +plan = BandSchedule.alignment(geometry,left,right)
      Receipt{Some{plan},"binary_align",0,BandSchedule.parallel(geometry,workers,12.0,1.0,plan),None{},None{},False{}}

def combined(geometry: Bands.Geometry,workers: U32,left: Receipt,right: Receipt) -> Receipt:
  combined_plan(geometry,workers,layout(left),layout(right))

# The public compatible pass compares every input plan before its Bool branch.
# These Data receipts observe the existing comparator, without tensor owners.
type ConcatLayout is Data:
  ConcatLayout{selected: Maybe<&2,Bands.Plan>,compatible: Bool,gather: Bool}

def single_concat_layout(layout: Maybe<&2,Bands.Plan>) -> ConcatLayout:
  match layout:
    case Some{plan}: ConcatLayout{Some{plan},True{},True{}}
    case None{}: ConcatLayout{None{},False{},False{}}

def cons_concat_layout(+first: Maybe<&2,Bands.Plan>,rest: ConcatLayout) -> ConcatLayout:
  ConcatLayout{+selected,compatible,gather} = rest
  ConcatLayout{BandConcat.first_plan(first,selected),compatible && BandConcat.compatible_plans(first,selected),
    gather || Maybe.is_some(&2,Bands.Plan,first)}

def concat_layout(layouts: List<&2,Maybe<&2,Bands.Plan>>) -> ConcatLayout:
  match layouts:
    case []: ConcatLayout{None{},False{},False{}}
    case [first]: single_concat_layout(first)
    case first <> rest: cons_concat_layout(first,concat_layout(rest))

def concat_selected(valid: Bool,geometry: Maybe<&2,Bands.Geometry>,count: Maybe<&2,U32>,discovered: ConcatLayout) -> Receipt:
  match valid geometry count discovered:
    case True{} Some{geometry} Some{count} ConcatLayout{Some{plan},compatible,gather}:
      Receipt{Some{plan},"channel_concat",0,False{},None{},None{},False{}}
    case valid geometry count ConcatLayout{selected,compatible,gather}:
      Receipt{None{},"dense_concat",0,False{},None{},None{},gather}

def concat_discovered(+shape: List<&2,U32>,axis: U32,discovered: ConcatLayout) -> Receipt:
  ConcatLayout{selected,+compatible,gather} = discovered
  concat_selected(compatible && Bool.not(Shapes.has_zero(shape)),BandConcat.geometry(shape,axis),Tensors.count_of(shape),
    ConcatLayout{selected,compatible,gather})

def concatenated(shape: List<&2,U32>,axis: U32,layouts: List<&2,Maybe<&2,Bands.Plan>>) -> Receipt:
  concat_discovered(shape,axis,concat_layout(layouts))

def append_plan(+geometry: Bands.Geometry,+workers: U32,input: Maybe<&2,Bands.Plan>,target: Maybe<&2,Bands.Plan>) -> Bool:
  match input target:
    case Some{input} Some{+target}: Bands.same_plan(input,target) && BandSchedule.parallel(geometry,workers,8.0,0.0,target)
    case input target: False{}

def append_parallel(geometry: Bands.Geometry,workers: U32,input: Receipt,target: Receipt) -> Bool:
  append_plan(geometry,workers,layout(input),layout(target))

def pooled(+window: BandWindow.Geometry,workers: U32,previous: Receipt) -> Receipt:
  Receipt{plan,route,depth,parallel,estimate,repartition,gather} = previous
  match plan:
    case None{}: dense()
    case Some{+plan}: Receipt{Some{output_plan(plan,window)},"band_pool",0,BandPool.parallel(window,workers,BandPool.plan_size(plan,window)),None{},None{},False{}}

def pool_dimensions(pool: Access.Pool,+geometry: Bands.Geometry,workers: U32,previous: Receipt) -> Receipt:
  Access.Pool{height,width,kernel,stride,padding,oh,ow} = pool
  pooled(BandWindow.Geometry{geometry,kernel,stride,padding,oh,ow},workers,previous)

def dense_fallback(route: String,previous: Receipt) -> Receipt:
  Receipt{plan,old_route,depth,parallel,estimate,repartition,gather} = previous
  Receipt{None{},route,0,False{},None{},None{},Maybe.is_some(&2,Bands.Plan,plan)}

def pool_selected(valid: Bool,planes: Maybe<&2,U32>,count: Maybe<&2,U32>,pool: Access.Pool,geometry: Bands.Geometry,workers: U32,previous: Receipt) -> Receipt:
  match valid planes count:
    case True{} Some{planes} Some{count}: pool_dimensions(pool,geometry,workers,previous)
    case valid planes count: dense_fallback("dense_pool",previous)

def pool_public_geometry(+shape: List<&2,U32>,+geometry: Bands.Geometry,+pool: Access.Pool,workers: U32,previous: Receipt) -> Receipt:
  Access.Pool{+height,+width,+kernel,+stride,+padding,+oh,+ow} = pool
  +pool = {Access.Pool{height,width,kernel,stride,padding,oh,ow} : Access.Pool}
  +out_shape = List.append(&2,U32,List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n)),[oh,ow])
  pool_selected(Window.band_shape(shape,geometry) && Operations.pool_geometry(pool) && U32.is_lt(0,height) &&
    U32.is_lt(0,width) && Bool.not(Shapes.has_zero(out_shape)),Tensors.count_of(Window.middle_shape(shape,pool)),Tensors.count_of(out_shape),pool,geometry,workers,previous)

def pooled_public(+shape: List<&2,U32>,geometry: Bands.Geometry,kernel: U32,stride: U32,padding: U32,workers: U32,previous: Receipt) -> Receipt:
  pool_public_geometry(shape,geometry,Window.geometry(shape,kernel,stride,padding),workers,previous)

def scaled_plan(plan: Bands.Plan,+scale: U32) -> Bands.Plan:
  match plan:
    case Bands.Rows{first,count}: Bands.Rows{(first * scale : U32),(count * scale : U32)}
    case Bands.Split{left,right}: Bands.Split{scaled_plan(left,scale),scaled_plan(right,scale)}

def upsampled(+geometry: Bands.Geometry,+workers: U32,+scale: U32,previous: Receipt) -> Receipt:
  Receipt{plan,route,depth,parallel,estimate,repartition,gather} = previous
  match plan:
    case None{}: dense()
    case Some{+plan}: Receipt{Some{scaled_plan(plan,scale)},"band_upsample",0,BandSchedule.parallel(geometry,workers,(U32.to_f32(scale) * U32.to_f32(scale) * 12.0 : F32),0.0,plan),None{},None{},False{}}

def upsample_selected(valid: Bool,geometry: Bands.Geometry,workers: U32,scale: U32,previous: Receipt) -> Receipt:
  match valid:
    case True{}: upsampled(geometry,workers,scale,previous)
    case False{}: dense_fallback("dense_upsample",previous)

def upsample_public(+shape: List<&2,U32>,geometry: Bands.Geometry,workers: U32,+scale: U32,previous: Receipt) -> Receipt:
  Bands.Geometry{batch,channels,+height,+width} = geometry
  upsample_selected(Bool.not(U32.is_zero(scale)) && Views.has_count(Shapes.checked_multiply(height,scale)) &&
    Views.has_count(Shapes.checked_multiply(width,scale)) && Views.has_count(Tensors.count_of(Layout.scaled_extents(shape,scale))) &&
    Bool.not(Shapes.has_zero(Layout.scaled_extents(shape,scale))),Bands.Geometry{batch,channels,height,width},workers,scale,previous)

# The indexed decode keeps the logits' bands; the reshape after it gathers.
def decoded(geometry: Bands.Geometry,workers: U32,operations: F32,previous: Receipt) -> Receipt:
  Receipt{plan,route,depth,parallel,estimate,repartition,gather} = previous
  match plan:
    case None{}: Receipt{None{},"dense_decode",0,False{},None{},None{},False{}}
    case Some{plan}: Receipt{None{},"band_decode",0,BandSchedule.parallel(geometry,workers,12.0,operations,plan),None{},None{},True{}}

def emit_plan(plan: Bands.Plan) -> IO(Unit):
  match plan:
    case Bands.Rows{first,count}:
      do IO<Unit>:
        IO.write("{\"first\":")
        IO.write(Nat.show(U32.to_nat(first)))
        IO.write(",\"rows\":")
        IO.write(Nat.show(U32.to_nat(count)))
        IO.write("}")
    case Bands.Split{left,right}:
      do IO<Unit>:
        IO.write("{\"left\":")
        emit_plan(left)
        IO.write(",\"right\":")
        emit_plan(right)
        IO.write("}")

def emit_layout(plan: Maybe<&2,Bands.Plan>) -> IO(Unit):
  match plan:
    case None{}: IO.write("null")
    case Some{plan}: emit_plan(plan)

def emit_estimate(estimate: Maybe<&2,F32>) -> IO(Unit):
  match estimate:
    case None{}: IO.write("null")
    case Some{time}: IO.write(F32.show(time))

def emit(index: U32,receipt: Receipt) -> IO(Unit):
  Receipt{plan,route,depth,parallel,estimate,repartition,gather} = receipt
  do IO<Unit>:
    IO.write("{\"operation\":")
    IO.write(Nat.show(U32.to_nat(index)))
    IO.write(",\"route\":\"")
    IO.write(route)
    IO.write("\",\"output_layout\":")
    emit_layout(plan)
    IO.write(",\"initial_or_dense_depth\":")
    IO.write(Nat.show(U32.to_nat(depth)))
    IO.write(",\"parallel\":")
    IO.write(Bool.pick(String,parallel,"true","false"))
    IO.write(",\"estimated_ns\":")
    emit_estimate(estimate)
    IO.write(",\"repartition_ns\":")
    emit_estimate(repartition)
    IO.write(",\"gather_input\":")
    IO.write(Bool.pick(String,gather,"true","false"))
    IO.write("}\n")

def emit_append(index: U32,part: U32,parallel: Bool) -> IO(Unit):
  do IO<Unit>:
    IO.write("{\"operation\":")
    IO.write(Nat.show(U32.to_nat(index)))
    IO.write(",\"part\":")
    IO.write(Nat.show(U32.to_nat(part)))
    IO.write(",\"append_parallel\":")
    IO.write(Bool.pick(String,parallel,"true","false"))
    IO.write("}\n")

def emit_count(index: U32,count: Maybe<&2,U32>) -> IO(Unit):
  match count:
    case None{}:
      do IO<Unit>:
        IO.write("{\"tensor\":")
        IO.write(Nat.show(U32.to_nat(index)))
        IO.write(",\"count\":null}\n")
    case Some{count}:
      do IO<Unit>:
        IO.write("{\"tensor\":")
        IO.write(Nat.show(U32.to_nat(index)))
        IO.write(",\"count\":")
        IO.write(Nat.show(U32.to_nat(count)))
        IO.write("}\n")

def emit_tensor_count(index: U32,shape: List<&2,U32>) -> IO(Unit):
  emit_count(index,Tensors.count_of(shape))
