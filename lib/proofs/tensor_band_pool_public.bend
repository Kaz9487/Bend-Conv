# Public pool callers use the actual selector and count computation: the
# planes swept along the columns and the output are both counted. The
# incoming Normal state supplies the input count and all source ownership.
# Both routes end in the same contract: the band route from its eligibility,
# the dense fallback from a Ready result of the actual call.
import Base
import ../tensor.bend as Tensors
import ../tensor_window.bend as Window
import ../tensor_band.bend as Bands
import ../tensor_band_window.bend as BandWindow
import ../tensor_band_pool.bend as BandPool
import ../tensor_access.bend as Access
import ../tensor_view.bend as Views
import ../tensor_operations.bend as Operations
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_model.bend as Model
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ./tensor_band_pool_execution.bend as BandPoolExecution
import ./tensor_band_pool_samples.bend as BandPoolSamples
import ./tensor_band_pool_window_values.bend as BandPoolWindowValues
import ./tensor_band_pool_output_domain.bend as OutputDomain
import ./tensor_pool_scalar_bounds.bend as Scalar
import ./tensor_dense_contract.bend as Dense
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_dense_observation.bend as DenseObservation
import ./tensor_band_concat_execution.bend as BandConcatExecution
import ./tensor_ready_densification.bend as ReadyDensification
import ./tensor_pool_dense_values.bend as PoolDense
import ./tensor_sweep_values.bend as Sweeps
import ./tensor_dense_normal.bend as DenseNormal
import ./storage_shape.bend as Shape
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding

def window_geometry(input: Bands.Geometry,+kernel: U32,+stride: U32,+padding: U32) -> BandWindow.Geometry:
  Bands.Geometry{images,channels,+height,+width} = input
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,
    Window.out_extent(height,kernel,stride,padding),Window.out_extent(width,kernel,stride,padding)}

def both_true(+left: Bool,+right: Bool,left_true: {left == True{} : Bool},right_true: {right == True{} : Bool}) ->
  {left && right == True{} : Bool}:
  %Equal.sym(Bool,left,True{},left_true) : {_ && right == True{} : Bool}
  %Equal.sym(Bool,right,True{},right_true) : {True{} && _ == True{} : Bool}
  {==}

# Output axes replace only the last two input axes. Both leading axes keep
# their original meanings; CHW retains its actual implicit batch of one.
def OutputCoherent(+shape: List<&2,U32>,+geometry: BandWindow.Geometry) -> Data:
  {Window.band_shape(OutputDomain.output_shape(shape,geometry),BandPoolExecution.values_output_geometry(geometry)) == True{} : Bool}

def output_coherent(+shape: List<&2,U32>,+geometry: BandWindow.Geometry,
  +matched: {Window.band_shape(shape,BandWindow.input_geometry(geometry)) == True{} : Bool}) -> OutputCoherent(shape,geometry):
  match shape geometry:
    case [+c,+h,+w] BandWindow.Geometry{Bands.Geometry{+images,+channels,+height,+width},kernel,stride,padding,+oh,+ow}:
      +leading = Boolean.left((U32.is_eq(images,1) && U32.is_eq(c,channels) : Bool),U32.is_eq(h,height),
        Boolean.left((U32.is_eq(images,1) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),matched))
      both_true((U32.is_eq(images,1) && U32.is_eq(c,channels) && U32.is_eq(oh,oh) : Bool),U32.is_eq(ow,ow),
        both_true((U32.is_eq(images,1) && U32.is_eq(c,channels) : Bool),U32.is_eq(oh,oh),leading,BandShape.equal_words(oh,oh,{==})),BandShape.equal_words(ow,ow,{==}))
    case [+n,+c,+h,+w] BandWindow.Geometry{Bands.Geometry{+images,+channels,+height,+width},kernel,stride,padding,+oh,+ow}:
      +leading = Boolean.left((U32.is_eq(n,images) && U32.is_eq(c,channels) : Bool),U32.is_eq(h,height),
        Boolean.left((U32.is_eq(n,images) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),matched))
      both_true((U32.is_eq(n,images) && U32.is_eq(c,channels) && U32.is_eq(oh,oh) : Bool),U32.is_eq(ow,ow),
        both_true((U32.is_eq(n,images) && U32.is_eq(c,channels) : Bool),U32.is_eq(oh,oh),leading,BandShape.equal_words(oh,oh,{==})),BandShape.equal_words(ow,ow,{==}))
    case Nil{} BandWindow.Geometry{input,kernel,stride,padding,oh,ow}: Empty.absurd(OutputCoherent(shape,geometry),WordEncoding.false_true(matched))
    case Con{a,Nil{}} BandWindow.Geometry{input,kernel,stride,padding,oh,ow}: Empty.absurd(OutputCoherent(shape,geometry),WordEncoding.false_true(matched))
    case Con{a,Con{b,Nil{}}} BandWindow.Geometry{input,kernel,stride,padding,oh,ow}: Empty.absurd(OutputCoherent(shape,geometry),WordEncoding.false_true(matched))
    case Con{a,Con{b,Con{c,Con{d,Con{e,rest}}}}} BandWindow.Geometry{input,kernel,stride,padding,oh,ow}: Empty.absurd(OutputCoherent(shape,geometry),WordEncoding.false_true(matched))

# Decode the actual public last-axis reader from full shape coherence.
# No pool geometry equality is supplied by a caller.
def actual_pool(+shape: List<&2,U32>,+input: Bands.Geometry,+kernel: U32,+stride: U32,+padding: U32,
  +matched: {Window.band_shape(shape,input) == True{} : Bool}) ->
  {Window.geometry(shape,kernel,stride,padding) == OutputDomain.pool(window_geometry(input,kernel,stride,padding)) : Access.Pool}:
  match shape input:
    case [+c,+h,+w] Bands.Geometry{+images,+channels,+height,+width}:
      +height_equal = ShapeLaws.native_equal(h,height,Boolean.right((U32.is_eq(images,1) && U32.is_eq(c,channels) : Bool),U32.is_eq(h,height),
        Boolean.left((U32.is_eq(images,1) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),matched)))
      +width_equal = ShapeLaws.native_equal(w,width,Boolean.right((U32.is_eq(images,1) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),matched))
      %Equal.sym(U32,h,height,height_equal) : {Window.geometry([c,_,w],kernel,stride,padding) == OutputDomain.pool(window_geometry(input,kernel,stride,padding)) : Access.Pool}
      %Equal.sym(U32,w,width,width_equal) : {Window.geometry([c,height,_],kernel,stride,padding) == OutputDomain.pool(window_geometry(input,kernel,stride,padding)) : Access.Pool}
      {==}
    case [+n,+c,+h,+w] Bands.Geometry{+images,+channels,+height,+width}:
      +height_equal = ShapeLaws.native_equal(h,height,Boolean.right((U32.is_eq(n,images) && U32.is_eq(c,channels) : Bool),U32.is_eq(h,height),
        Boolean.left((U32.is_eq(n,images) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),matched)))
      +width_equal = ShapeLaws.native_equal(w,width,Boolean.right((U32.is_eq(n,images) && U32.is_eq(c,channels) && U32.is_eq(h,height) : Bool),U32.is_eq(w,width),matched))
      %Equal.sym(U32,h,height,height_equal) : {Window.geometry([n,c,_,w],kernel,stride,padding) == OutputDomain.pool(window_geometry(input,kernel,stride,padding)) : Access.Pool}
      %Equal.sym(U32,w,width,width_equal) : {Window.geometry([n,c,height,_],kernel,stride,padding) == OutputDomain.pool(window_geometry(input,kernel,stride,padding)) : Access.Pool}
      {==}
    case Nil{} Bands.Geometry{images,channels,height,width}: Empty.absurd({Window.geometry(shape,kernel,stride,padding) == OutputDomain.pool(window_geometry(input,kernel,stride,padding)) : Access.Pool},WordEncoding.false_true(matched))
    case Con{a,Nil{}} Bands.Geometry{images,channels,height,width}: Empty.absurd({Window.geometry(shape,kernel,stride,padding) == OutputDomain.pool(window_geometry(input,kernel,stride,padding)) : Access.Pool},WordEncoding.false_true(matched))
    case Con{a,Con{b,Nil{}}} Bands.Geometry{images,channels,height,width}: Empty.absurd({Window.geometry(shape,kernel,stride,padding) == OutputDomain.pool(window_geometry(input,kernel,stride,padding)) : Access.Pool},WordEncoding.false_true(matched))
    case Con{a,Con{b,Con{c,Con{d,Con{e,rest}}}}} Bands.Geometry{images,channels,height,width}: Empty.absurd({Window.geometry(shape,kernel,stride,padding) == OutputDomain.pool(window_geometry(input,kernel,stride,padding)) : Access.Pool},WordEncoding.false_true(matched))

def planes_shape(+shape: List<&2,U32>,+geometry: BandWindow.Geometry) -> List<&2,U32>:
  Window.middle_shape(shape,OutputDomain.pool(geometry))

def eligible(+shape: List<&2,U32>,+input: Bands.Geometry,+kernel: U32,+stride: U32,+padding: U32) -> Bool:
  +geometry = window_geometry(input,kernel,stride,padding)
  OutputDomain.enabled(shape,geometry) && Views.has_count(Tensors.count_of(planes_shape(shape,geometry))) && Views.has_count(Tensors.count_of(OutputDomain.output_shape(shape,geometry)))

def selected(+shape: List<&2,U32>,+input: Bands.Geometry,+kernel: U32,+stride: U32,+padding: U32,
  truth: {eligible(shape,input,kernel,stride,padding) == True{} : Bool}) ->
  {OutputDomain.enabled(shape,window_geometry(input,kernel,stride,padding)) == True{} : Bool}:
  +geometry = window_geometry(input,kernel,stride,padding)
  +planes = Views.has_count(Tensors.count_of(planes_shape(shape,geometry)))
  Boolean.left(OutputDomain.enabled(shape,geometry),planes,
    Boolean.left((OutputDomain.enabled(shape,geometry) && planes : Bool),Views.has_count(Tensors.count_of(OutputDomain.output_shape(shape,geometry))),truth))

def counted_result(+shape: List<&2,U32>,result: Maybe<&2,U32>,accepted: {Tensors.count_of(shape) == result : Maybe<&2,U32>},
  found: {Views.has_count(result) == True{} : Bool}) -> Global.Counted<shape>:
  match result:
    case Some{count}: Global.Counted{count,accepted}
    case None{}: Empty.absurd(Global.Counted<shape>,WordEncoding.false_true(found))

def planes_counted(+shape: List<&2,U32>,+input: Bands.Geometry,+kernel: U32,+stride: U32,+padding: U32,
  truth: {eligible(shape,input,kernel,stride,padding) == True{} : Bool}) ->
  Global.Counted<planes_shape(shape,window_geometry(input,kernel,stride,padding))>:
  +geometry = window_geometry(input,kernel,stride,padding)
  +middle_shape = planes_shape(shape,geometry)
  counted_result(middle_shape,Tensors.count_of(middle_shape),{==},Boolean.right(OutputDomain.enabled(shape,geometry),Views.has_count(Tensors.count_of(middle_shape)),
    Boolean.left((OutputDomain.enabled(shape,geometry) && Views.has_count(Tensors.count_of(middle_shape)) : Bool),Views.has_count(Tensors.count_of(OutputDomain.output_shape(shape,geometry))),truth)))

def output_counted(+shape: List<&2,U32>,+input: Bands.Geometry,+kernel: U32,+stride: U32,+padding: U32,
  truth: {eligible(shape,input,kernel,stride,padding) == True{} : Bool}) ->
  Global.Counted<OutputDomain.output_shape(shape,window_geometry(input,kernel,stride,padding))>:
  +geometry = window_geometry(input,kernel,stride,padding)
  +out_shape = OutputDomain.output_shape(shape,geometry)
  counted_result(out_shape,Tensors.count_of(out_shape),{==},
    Boolean.right((OutputDomain.enabled(shape,geometry) && Views.has_count(Tensors.count_of(planes_shape(shape,geometry))) : Bool),Views.has_count(Tensors.count_of(out_shape)),truth))

# The result is Normal and Ready, and every element is the pool of its pixel:
# the fold, from pad, of the k lines of its window, each the fold, from pad,
# of its k samples.
def Contract(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,
  +logical: Logical,+geometry: BandWindow.Geometry,+pad: Element,-tensor: Tensors.Tensor<Element>) -> Data:
  Shape.Both<Global.Normal(~Element,tensor),Shape.Both<{Dense.tensor_ready(~Element,tensor) == True{} : Bool},
    Dense.TensorValues(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),BandPoolExecution.global_context(~Element,~Logical,logical,geometry,pad),tensor)>>

def invocation(~Element: Data,~operation: Element -> Element -> Element,+name: String,+pad: Element,+input: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,
  -bands: Bands.Bands(~Element),+planes: U32,+count: U32,+matched: {Window.band_shape(shape,input) == True{} : Bool},
  selection: {OutputDomain.enabled(shape,window_geometry(input,kernel,stride,padding)) == True{} : Bool},
  planes_counted: {Tensors.count_of(planes_shape(shape,window_geometry(input,kernel,stride,padding))) == Some{planes} : Maybe<&2,U32>},
  counted: {Tensors.count_of(OutputDomain.output_shape(shape,window_geometry(input,kernel,stride,padding))) == Some{count} : Maybe<&2,U32>}) ->
  {Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding) ==
    Tensors.Banded{BandPool.execute(~Element,~operation,BandPool.Context{window_geometry(input,kernel,stride,padding),pad,fill,workers},bands),
      BandPoolExecution.values_output_geometry(window_geometry(input,kernel,stride,padding)),OutputDomain.output_shape(shape,window_geometry(input,kernel,stride,padding)),fill,Tensors.Usable{},workers} : Tensors.Tensor<Element>}:
  Bands.Geometry{images,channels,height,width} = input
  +geometry = window_geometry(input,kernel,stride,padding)
  +out_shape = OutputDomain.output_shape(shape,geometry)
  %Equal.sym(Access.Pool,Window.geometry(shape,kernel,stride,padding),OutputDomain.pool(geometry),actual_pool(shape,input,kernel,stride,padding,matched)) :
    {Window.band_geometry(~Element,~operation,name,pad,_,input,bands,shape,fill,workers) ==
      Tensors.Banded{BandPool.execute(~Element,~operation,BandPool.Context{geometry,pad,fill,workers},bands),BandPoolExecution.values_output_geometry(geometry),out_shape,fill,Tensors.Usable{},workers} : Tensors.Tensor<Element>}
  %Equal.sym(Bool,OutputDomain.enabled(shape,geometry),True{},selection) :
    {Window.band_count(~Element,~operation,name,_,Tensors.count_of(planes_shape(shape,geometry)),Tensors.count_of(out_shape),pad,OutputDomain.pool(geometry),input,bands,shape,out_shape,fill,workers) ==
      Tensors.Banded{BandPool.execute(~Element,~operation,BandPool.Context{geometry,pad,fill,workers},bands),BandPoolExecution.values_output_geometry(geometry),out_shape,fill,Tensors.Usable{},workers} : Tensors.Tensor<Element>}
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(planes_shape(shape,geometry)),Some{planes},planes_counted) :
    {Window.band_count(~Element,~operation,name,True{},_,Tensors.count_of(out_shape),pad,OutputDomain.pool(geometry),input,bands,shape,out_shape,fill,workers) ==
      Tensors.Banded{BandPool.execute(~Element,~operation,BandPool.Context{geometry,pad,fill,workers},bands),BandPoolExecution.values_output_geometry(geometry),out_shape,fill,Tensors.Usable{},workers} : Tensors.Tensor<Element>}
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(out_shape),Some{count},counted) :
    {Window.band_count(~Element,~operation,name,True{},Some{planes},_,pad,OutputDomain.pool(geometry),input,bands,shape,out_shape,fill,workers) ==
      Tensors.Banded{BandPool.execute(~Element,~operation,BandPool.Context{geometry,pad,fill,workers},bands),BandPoolExecution.values_output_geometry(geometry),out_shape,fill,Tensors.Usable{},workers} : Tensors.Tensor<Element>}
  {==}

def band_counted_machine(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +machine: Scalar.Observation,+logical: Logical,+input: Bands.Geometry,+shape: List<&2,U32>,+pad: Element,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,+source: Model.Model<Element>,
  +truth: {eligible(shape,input,kernel,stride,padding) == True{} : Bool},
  normal: Global.Normal(~Element,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers}),
  source_values: BandValues.Values(~Element,~Logical,~original,logical,input,Model.contents(~Element,Model.pack(~Element,source))),
  planes_output: Global.Counted<planes_shape(shape,window_geometry(input,kernel,stride,padding))>,
  output: Global.Counted<OutputDomain.output_shape(shape,window_geometry(input,kernel,stride,padding))>) ->
  Contract(~Element,~Logical,~original,~operation,logical,window_geometry(input,kernel,stride,padding),pad,
    Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)):
  Bands.Geometry{images,channels,height,width} = input
  Shape.Both{source_well_formed,Shape.Both{input_counted,+coherent}} = normal
  Global.Counted{+planes,+planes_accepted} = planes_output
  Global.Counted{+count,+accepted} = output
  +geometry = window_geometry(input,kernel,stride,padding)
  +selection = selected(shape,input,kernel,stride,padding,truth)
  +source_well_formed = {%Model.packed_contents(~Element,source) : BandShape.WellFormed<Element,input,_> source_well_formed : BandShape.WellFormed<Element,input,Model.content(~Element,source)>}
  +source_values = {%Model.packed_contents(~Element,source) : BandValues.Values(~Element,~Logical,~original,logical,input,_) source_values : BandValues.Values(~Element,~Logical,~original,logical,input,Model.content(~Element,source))}
  %Equal.sym(Tensors.Tensor<Element>,Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding),
    Tensors.Banded{BandPool.execute(~Element,~operation,BandPool.Context{geometry,pad,fill,workers},Model.pack(~Element,source)),BandPoolExecution.values_output_geometry(geometry),OutputDomain.output_shape(shape,geometry),fill,Tensors.Usable{},workers},
    invocation(~Element,~operation,name,pad,input,shape,fill,workers,kernel,stride,padding,Model.pack(~Element,source),planes,count,coherent,selection,planes_accepted,accepted)) :
    Contract(~Element,~Logical,~original,~operation,logical,geometry,pad,_)
  Shape.Both{
    Shape.Both{BandPoolSamples.execute_well_formed(~Element,~operation,machine,shape,geometry,pad,fill,workers,count,source,selection,accepted,source_well_formed),
      Shape.Both{Global.Counted{count,accepted},output_coherent(shape,geometry,coherent)}},
    Shape.Both{{==},BandPoolSamples.execute_values_all(~Element,~Logical,~original,~operation,machine,logical,shape,geometry,pad,fill,workers,planes,count,source,selection,planes_accepted,accepted,source_well_formed,source_values,input_counted)}}

# The public caller constructs its machine receipt. Its result contains the
# complete tensor contract, so no fixed Default Nat bound is exposed here.
def band_counted(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +logical: Logical,+input: Bands.Geometry,+shape: List<&2,U32>,+pad: Element,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,+source: Model.Model<Element>,
  +truth: {eligible(shape,input,kernel,stride,padding) == True{} : Bool},
  normal: Global.Normal(~Element,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers}),
  source_values: BandValues.Values(~Element,~Logical,~original,logical,input,Model.contents(~Element,Model.pack(~Element,source))),
  planes_output: Global.Counted<planes_shape(shape,window_geometry(input,kernel,stride,padding))>,
  output: Global.Counted<OutputDomain.output_shape(shape,window_geometry(input,kernel,stride,padding))>) ->
  Contract(~Element,~Logical,~original,~operation,logical,window_geometry(input,kernel,stride,padding),pad,
    Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)):
  band_counted_machine(~Element,~Logical,~original,~operation,name,Scalar.observe_envelope(),logical,input,shape,pad,fill,workers,kernel,stride,padding,source,truth,normal,source_values,planes_output,output)

def band_modeled(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +logical: Logical,+input: Bands.Geometry,+shape: List<&2,U32>,+pad: Element,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,+source: Model.Model<Element>,
  +truth: {eligible(shape,input,kernel,stride,padding) == True{} : Bool},
  normal: Global.Normal(~Element,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers}),
  source_values: BandValues.Values(~Element,~Logical,~original,logical,input,Model.contents(~Element,Model.pack(~Element,source)))) ->
  Contract(~Element,~Logical,~original,~operation,logical,window_geometry(input,kernel,stride,padding),pad,
    Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)):
  band_counted(~Element,~Logical,~original,~operation,name,logical,input,shape,pad,fill,workers,kernel,stride,padding,source,truth,normal,source_values,planes_counted(shape,input,kernel,stride,padding,truth),output_counted(shape,input,kernel,stride,padding,truth))

def band_observed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +logical: Logical,+input: Bands.Geometry,+shape: List<&2,U32>,+pad: Element,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,-bands: Bands.Bands(~Element),
  +truth: {eligible(shape,input,kernel,stride,padding) == True{} : Bool},observation: Model.Observation<Element,bands>,
  normal: Global.Normal(~Element,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers}),
  source_values: BandValues.Values(~Element,~Logical,~original,logical,input,Model.contents(~Element,bands))) ->
  Contract(~Element,~Logical,~original,~operation,logical,window_geometry(input,kernel,stride,padding),pad,
    Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)):
  Model.Observation{+source,+source_exact} = observation
  %source_exact : Contract(~Element,~Logical,~original,~operation,logical,window_geometry(input,kernel,stride,padding),pad,
    Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{_,input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding))
  band_modeled(~Element,~Logical,~original,~operation,name,logical,input,shape,pad,fill,workers,kernel,stride,padding,source,truth,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,source),bands,source_exact) : Global.Normal(~Element,Tensors.Banded{_,input,shape,fill,Tensors.Usable{},workers}) normal,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,source),bands,source_exact) : BandValues.Values(~Element,~Logical,~original,logical,input,Model.contents(~Element,_)) source_values)

def band_ready(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +logical: Logical,+input: Bands.Geometry,+shape: List<&2,U32>,+pad: Element,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,bands: Bands.Bands(~Element),
  truth: {eligible(shape,input,kernel,stride,padding) == True{} : Bool},
  normal: Global.Normal(~Element,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers}),
  source_values: BandValues.Values(~Element,~Logical,~original,logical,input,Model.contents(~Element,bands))) ->
  Contract(~Element,~Logical,~original,~operation,logical,window_geometry(input,kernel,stride,padding),pad,
    Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)):
  band_observed(~Element,~Logical,~original,~operation,name,logical,input,shape,pad,fill,workers,kernel,stride,padding,bands,truth,Model.observe(~Element,bands),normal,source_values)

def band_status_ready(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +logical: Logical,+input: Bands.Geometry,+shape: List<&2,U32>,+pad: Element,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,
  status: Tensors.Status,bands: Bands.Bands(~Element),truth: {eligible(shape,input,kernel,stride,padding) == True{} : Bool},
  source_ready: {Dense.status_ready(status) == True{} : Bool},
  normal: Global.Normal(~Element,Tensors.Banded{bands,input,shape,fill,status,workers}),
  source_values: BandValues.Values(~Element,~Logical,~original,logical,input,Model.contents(~Element,bands))) ->
  Contract(~Element,~Logical,~original,~operation,logical,window_geometry(input,kernel,stride,padding),pad,
    Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{bands,input,shape,fill,status,workers},kernel,stride,padding)):
  match status:
    case Tensors.Usable{}:
      band_ready(~Element,~Logical,~original,~operation,name,logical,input,shape,pad,fill,workers,kernel,stride,padding,bands,truth,normal,source_values)
    case Tensors.Failed{problem}:
      Empty.absurd(Contract(~Element,~Logical,~original,~operation,logical,window_geometry(input,kernel,stride,padding),pad,
        Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{bands,input,shape,fill,status,workers},kernel,stride,padding)),WordEncoding.false_true(source_ready))

# The rejected selector runs exactly one gather and the existing dense pool.
# Its value and Normal contract can therefore use the shared densification
# family instead of separate proofs for every rejected spatial boundary.
def rejected_count(~Element: Data,~operation: Element -> Element -> Element,+name: String,
  valid: Bool,planes: Maybe<&2,U32>,count: Maybe<&2,U32>,+pool: Access.Pool,+input: Bands.Geometry,-bands: Bands.Bands(~Element),
  +pad: Element,+shape: List<&2,U32>,+out_shape: List<&2,U32>,+fill: Element,+workers: U32,
  rejected: {valid && Views.has_count(planes) && Views.has_count(count) == False{} : Bool}) ->
  {Window.band_count(~Element,~operation,name,valid,planes,count,pad,pool,input,bands,shape,out_shape,fill,workers) ==
    Window.pool2d(~Element,~operation,name,pad,DenseObservation.dense_tensor(~Element,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers}),
      BandPoolWindowValues.pool_kernel(pool),BandPoolWindowValues.pool_stride(pool),BandPoolWindowValues.pool_padding(pool)) : Tensors.Tensor<Element>}:
  match valid planes count pool:
    case True{} Some{planes} Some{count} Access.Pool{height,width,kernel,stride,padding,oh,ow}:
      Empty.absurd({Window.band_count(~Element,~operation,name,True{},Some{planes},Some{count},pad,pool,input,bands,shape,out_shape,fill,workers) ==
        Window.pool2d(~Element,~operation,name,pad,DenseObservation.dense_tensor(~Element,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers}),kernel,stride,padding) : Tensors.Tensor<Element>},
        WordEncoding.false_true(Equal.sym(Bool,True{},False{},rejected)))
    case False{} planes count Access.Pool{height,width,kernel,stride,padding,oh,ow}: {==}
    case True{} None{} count Access.Pool{height,width,kernel,stride,padding,oh,ow}: {==}
    case True{} Some{planes} None{} Access.Pool{height,width,kernel,stride,padding,oh,ow}: {==}

def rejected_result(~Element: Data,~operation: Element -> Element -> Element,+name: String,
  +pad: Element,+input: Bands.Geometry,-bands: Bands.Bands(~Element),+shape: List<&2,U32>,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,
  +coherent: {Window.band_shape(shape,input) == True{} : Bool},rejected: {eligible(shape,input,kernel,stride,padding) == False{} : Bool}) ->
  {Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding) ==
    Window.pool2d(~Element,~operation,name,pad,DenseObservation.dense_tensor(~Element,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers}),kernel,stride,padding) : Tensors.Tensor<Element>}:
  Bands.Geometry{images,channels,height,width} = input
  +geometry = window_geometry(input,kernel,stride,padding)
  %Equal.sym(Access.Pool,Window.geometry(shape,kernel,stride,padding),OutputDomain.pool(geometry),actual_pool(shape,input,kernel,stride,padding,coherent)) :
    {Window.band_geometry(~Element,~operation,name,pad,_,input,bands,shape,fill,workers) ==
      Window.pool2d(~Element,~operation,name,pad,DenseObservation.dense_tensor(~Element,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers}),kernel,stride,padding) : Tensors.Tensor<Element>}
  rejected_count(~Element,~operation,name,OutputDomain.enabled(shape,geometry),Tensors.count_of(planes_shape(shape,geometry)),Tensors.count_of(OutputDomain.output_shape(shape,geometry)),OutputDomain.pool(geometry),input,bands,pad,shape,OutputDomain.output_shape(shape,geometry),fill,workers,rejected)

# The rejected selector gathers once and runs the dense pool on the same pool
# geometry. A Ready result of that call therefore has the complete contract:
# the shared preparation supplies the dense contract of the gathered tensor
# and the dense pool theorem supplies the values.
def rejected_completed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +logical: Logical,+input: Bands.Geometry,+source: Model.Model<Element>,+shape: List<&2,U32>,+pad: Element,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,
  prepared: ReadyDensification.Prepared(~Element,~Logical,~original,logical,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers}),
  +result_equal: {Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding) == Window.pool2d(~Element,~operation,name,pad,DenseObservation.dense_tensor(~Element,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers}),kernel,stride,padding) : Tensors.Tensor<Element>},
  +coherent: {Window.band_shape(shape,input) == True{} : Bool},
  accepted: {Dense.tensor_ready(~Element,Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)) == True{} : Bool}) ->
  Contract(~Element,~Logical,~original,~operation,logical,window_geometry(input,kernel,stride,padding),pad,Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)):
  Bands.Geometry{images,channels,height,width} = input
  Shape.Both{dense_normal,dense_contract} = prepared
  %Equal.sym(Tensors.Tensor<Element>,Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding),Window.pool2d(~Element,~operation,name,pad,DenseObservation.dense_tensor(~Element,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers}),kernel,stride,padding),result_equal) :
    Contract(~Element,~Logical,~original,~operation,logical,window_geometry(input,kernel,stride,padding),pad,_)
  %actual_pool(shape,input,kernel,stride,padding,coherent) :
    Shape.Both<Global.Normal(~Element,Window.pool2d(~Element,~operation,name,pad,DenseObservation.dense_tensor(~Element,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers}),kernel,stride,padding)),Shape.Both<{Dense.tensor_ready(~Element,Window.pool2d(~Element,~operation,name,pad,DenseObservation.dense_tensor(~Element,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers}),kernel,stride,padding)) == True{} : Bool},
      Dense.TensorValues(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),PoolDense.context(~Element,~Logical,logical,_,pad),Window.pool2d(~Element,~operation,name,pad,DenseObservation.dense_tensor(~Element,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers}),kernel,stride,padding))>>
  DenseNormal.from_shaped(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),PoolDense.context(~Element,~Logical,logical,Window.geometry(shape,kernel,stride,padding),pad),PoolDense.result_shape(shape,kernel,stride,padding),Window.pool2d(~Element,~operation,name,pad,DenseObservation.dense_tensor(~Element,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers}),kernel,stride,padding),
    PoolDense.dense_shaped(~Element,~Logical,~original,~operation,name,logical,pad,kernel,stride,padding,
      Bands.gather(~Element,input,Model.pack(~Element,source),Tensors.buffer(~Element,fill,Bands.elements(input))),shape,fill,workers,dense_contract,
      %result_equal : {Dense.tensor_ready(~Element,_) == True{} : Bool}
      accepted))

def rejected_modeled(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +logical: Logical,+input: Bands.Geometry,+source: Model.Model<Element>,+shape: List<&2,U32>,+pad: Element,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,
  +normal: Global.Normal(~Element,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers}),values: BandValues.Values(~Element,~Logical,~original,logical,input,Model.contents(~Element,Model.pack(~Element,source))),
  rejected: {eligible(shape,input,kernel,stride,padding) == False{} : Bool},accepted: {Dense.tensor_ready(~Element,Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)) == True{} : Bool}) ->
  Contract(~Element,~Logical,~original,~operation,logical,window_geometry(input,kernel,stride,padding),pad,Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{Model.pack(~Element,source),input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)):
  Shape.Both{source_well_formed,Shape.Both{input_counted,+coherent}} = normal
  rejected_completed(~Element,~Logical,~original,~operation,name,logical,input,source,shape,pad,fill,workers,kernel,stride,padding,
    ReadyDensification.prepared_owner(~Element,~Logical,~original,logical,
      {BandConcatExecution.BandedOwner{source,input,shape,fill,Tensors.Usable{},workers} : BandConcatExecution.TensorOwnerModel<Element>},normal,{==},values),
    rejected_result(~Element,~operation,name,pad,input,Model.pack(~Element,source),shape,fill,workers,kernel,stride,padding,coherent,rejected),coherent,accepted)

def rejected_observed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +logical: Logical,+input: Bands.Geometry,-bands: Bands.Bands(~Element),+shape: List<&2,U32>,+pad: Element,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,
  observation: Model.Observation<Element,bands>,normal: Global.Normal(~Element,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers}),values: BandValues.Values(~Element,~Logical,~original,logical,input,Model.contents(~Element,bands)),
  rejected: {eligible(shape,input,kernel,stride,padding) == False{} : Bool},accepted: {Dense.tensor_ready(~Element,Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)) == True{} : Bool}) ->
  Contract(~Element,~Logical,~original,~operation,logical,window_geometry(input,kernel,stride,padding),pad,Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)):
  Model.Observation{+source,+equation} = observation
  %equation : Contract(~Element,~Logical,~original,~operation,logical,window_geometry(input,kernel,stride,padding),pad,Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{_,input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding))
  rejected_modeled(~Element,~Logical,~original,~operation,name,logical,input,source,shape,pad,fill,workers,kernel,stride,padding,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,source),bands,equation) : Global.Normal(~Element,Tensors.Banded{_,input,shape,fill,Tensors.Usable{},workers}) normal,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,source),bands,equation) : BandValues.Values(~Element,~Logical,~original,logical,input,Model.contents(~Element,_)) values,rejected,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,source),bands,equation) : {Dense.tensor_ready(~Element,Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{_,input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)) == True{} : Bool} accepted)

def rejected_ready(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +logical: Logical,+input: Bands.Geometry,bands: Bands.Bands(~Element),+shape: List<&2,U32>,+pad: Element,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,
  normal: Global.Normal(~Element,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers}),values: BandValues.Values(~Element,~Logical,~original,logical,input,Model.contents(~Element,bands)),
  rejected: {eligible(shape,input,kernel,stride,padding) == False{} : Bool},accepted: {Dense.tensor_ready(~Element,Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)) == True{} : Bool}) ->
  Contract(~Element,~Logical,~original,~operation,logical,window_geometry(input,kernel,stride,padding),pad,Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)):
  rejected_observed(~Element,~Logical,~original,~operation,name,logical,input,bands,shape,pad,fill,workers,kernel,stride,padding,Model.observe(~Element,bands),normal,values,rejected,accepted)

def classified(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  choice: Bool,+logical: Logical,+input: Bands.Geometry,+shape: List<&2,U32>,+pad: Element,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,
  selection: {eligible(shape,input,kernel,stride,padding) == choice : Bool},bands: Bands.Bands(~Element),
  normal: Global.Normal(~Element,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers}),values: BandValues.Values(~Element,~Logical,~original,logical,input,Model.contents(~Element,bands)),accepted: {Dense.tensor_ready(~Element,Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)) == True{} : Bool}) ->
  Contract(~Element,~Logical,~original,~operation,logical,window_geometry(input,kernel,stride,padding),pad,Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)):
  match choice:
    case False{}: rejected_ready(~Element,~Logical,~original,~operation,name,logical,input,bands,shape,pad,fill,workers,kernel,stride,padding,normal,values,selection,accepted)
    case True{}: band_ready(~Element,~Logical,~original,~operation,name,logical,input,shape,pad,fill,workers,kernel,stride,padding,bands,selection,normal,values)

# The public pool of a Ready banded tensor with Normal state and complete
# values. The theorem computes the actual selector. When the call returns
# Ready, on either route, the result is Normal and holds the ordered pool of
# the logical input at every index.
def band_result(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+name: String,
  +logical: Logical,+input: Bands.Geometry,bands: Bands.Bands(~Element),+shape: List<&2,U32>,+pad: Element,+fill: Element,+workers: U32,+kernel: U32,+stride: U32,+padding: U32,
  normal: Global.Normal(~Element,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers}),values: BandValues.Values(~Element,~Logical,~original,logical,input,Model.contents(~Element,bands)),accepted: {Dense.tensor_ready(~Element,Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)) == True{} : Bool}) ->
  Contract(~Element,~Logical,~original,~operation,logical,window_geometry(input,kernel,stride,padding),pad,Window.pool2d(~Element,~operation,name,pad,Tensors.Banded{bands,input,shape,fill,Tensors.Usable{},workers},kernel,stride,padding)):
  classified(~Element,~Logical,~original,~operation,name,eligible(shape,input,kernel,stride,padding),logical,input,shape,pad,fill,workers,kernel,stride,padding,{==},bands,normal,values,accepted)
