# Channel concatenation writes each input range directly into its final band.
# Different input boundaries need no temporary dense tensor or aligned copy.
import Base
import ./tensor.bend as Tensors
import ./tensor_band.bend as Bands
import ./storage_buffer.bend as Storage
import ./traversal_partition.bend as Partition
import ./storage_copy.bend as Copy
import ./tensor_band_schedule.bend as BandSchedule

def geometry(shape: List<&2,U32>,axis: U32) -> Maybe<&2,Bands.Geometry>:
  match shape axis:
    case [channels,height,width] 0: Some{Bands.Geometry{1,channels,height,width}}
    case [batch,channels,height,width] 1: Some{Bands.Geometry{batch,channels,height,width}}
    case shape axis: None{}

def tensor_planned(~Element: Data,geometry: Bands.Geometry,shape: List<&2,U32>,fill: Element,status: Tensors.Status,workers: U32,
  owners: Bands.Bands(~Element) & Bands.Plan) -> Tensors.Tensor<Element> & Maybe<&2,Bands.Plan>:
  (bands,layout) = owners
  (Tensors.Banded{bands,geometry,shape,fill,status,workers},Some{layout})

def tensor_plan(~Element: Data,tensor: Tensors.Tensor<Element>) -> Tensors.Tensor<Element> & Maybe<&2,Bands.Plan>:
  match tensor:
    case Tensors.Tensor{storage,shape,fill,status,workers}: (Tensors.Tensor{storage,shape,fill,status,workers},None{})
    case Tensors.Banded{bands,geometry,shape,fill,status,workers}: tensor_planned(~Element,geometry,shape,fill,status,workers,Bands.plan(~Element,bands))

def first_plan(first: Maybe<&2,Bands.Plan>,rest: Maybe<&2,Bands.Plan>) -> Maybe<&2,Bands.Plan>:
  match first:
    case Some{layout}: Some{layout}
    case None{}: rest

def single_plan(~Element: Data,first: Tensors.Tensor<Element> & Maybe<&2,Bands.Plan>) -> Tensors.Parts<Element> & Maybe<&2,Bands.Plan>:
  (tensor,layout) = first
  (Tensors.Single{tensor},layout)

def cons_plan(~Element: Data,first: Tensors.Tensor<Element> & Maybe<&2,Bands.Plan>,rest: Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) -> Tensors.Parts<Element> & Maybe<&2,Bands.Plan>:
  (tensor,first) = first
  (rest,other) = rest
  (Tensors.Cons{tensor,rest},first_plan(first,other))

def discover(~Element: Data,parts: Tensors.Parts<Element>) -> Tensors.Parts<Element> & Maybe<&2,Bands.Plan>:
  match parts:
    case Tensors.Single{tensor}: single_plan(~Element,tensor_plan(~Element,tensor))
    case Tensors.Cons{tensor,rest}: cons_plan(~Element,tensor_plan(~Element,tensor),discover(~Element,rest))

# Reuse the existing metadata traversal to classify the common band path.
# Mixed representations and different plans use the dense concat caller.
def compatible_plans(first: Maybe<&2,Bands.Plan>,rest: Maybe<&2,Bands.Plan>) -> Bool:
  match first rest:
    case Some{first} Some{rest}: Bands.same_plan(first,rest)
    case first rest: False{}

def single_compatible_plan(~Element: Data,tensor: Tensors.Tensor<Element>,layout: Maybe<&2,Bands.Plan>) ->
  (Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool:
  match layout:
    case Some{layout}: ((Tensors.Single{tensor},Some{layout}),True{})
    case None{}: ((Tensors.Single{tensor},None{}),False{})

def single_compatible(~Element: Data,first: Tensors.Tensor<Element> & Maybe<&2,Bands.Plan>) ->
  (Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool:
  (tensor,layout) = first
  single_compatible_plan(~Element,tensor,layout)

def cons_compatible(~Element: Data,first: Tensors.Tensor<Element> & Maybe<&2,Bands.Plan>,
  rest: (Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool) -> (Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool:
  (tensor,first_layout) = first
  (rest_and_layout,compatible) = rest
  (rest_parts,rest_layout) = rest_and_layout
  +first_layout = {first_layout : Maybe<&2,Bands.Plan>}
  +rest_layout = {rest_layout : Maybe<&2,Bands.Plan>}
  ((Tensors.Cons{tensor,rest_parts},first_plan(first_layout,rest_layout)),compatible && compatible_plans(first_layout,rest_layout))

def discover_compatible(~Element: Data,parts: Tensors.Parts<Element>) -> (Tensors.Parts<Element> & Maybe<&2,Bands.Plan>) & Bool:
  match parts:
    case Tensors.Single{tensor}: single_compatible(~Element,tensor_plan(~Element,tensor))
    case Tensors.Cons{tensor,rest}: cons_compatible(~Element,tensor_plan(~Element,tensor),discover_compatible(~Element,rest))

def allocate(~Element: Data,layout: Bands.Plan,+geometry: Bands.Geometry,+fill: Element) -> Bands.Bands(~Element):
  match layout:
    case Bands.Rows{first,+rows}: Partition.Leaf{Tensors.buffer(~Element,fill,Bands.local_count(geometry,rows)),first,rows,Unit{}}
    case Bands.Split{left,right}: Partition.Fork{allocate(~Element,left,geometry,fill),allocate(~Element,right,geometry,fill)}

def dense_rows(~Element: Data,geometry: Bands.Geometry,+first: U32,+rows: U32,channel: U32,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  Bands.Geometry{batch,+channels,+height,+width} = geometry
  Bands.planes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),
    ((first / height * channels * height + first % height) * width : U32),(channel * rows * width : U32),(input,output))

def channel_offset(geometry: Bands.Geometry,channel: U32,rows: U32) -> U32:
  Bands.Geometry{batch,channels,height,width} = geometry
  (channel * rows * width : U32)

def rows(~Element: Data,geometry: Bands.Geometry,+first: U32,+rows: U32,channel: U32,tensor: Tensors.Tensor<Element>,output: Storage.Buffer<Element>) -> Tensors.Tensor<Element> & Storage.Buffer<Element>:
  match tensor:
    case Tensors.Tensor{input,shape,fill,status,workers}:
      Storage.bind_pair(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~(Tensors.Tensor<Element> & Storage.Buffer<Element>),
        dense_rows(~Element,geometry,first,rows,channel,input,output),input => output => (Tensors.Tensor{input,shape,fill,status,workers},output))
    case Tensors.Banded{bands,+geometry,shape,fill,status,workers}:
      Storage.bind_pair(~Bands.Bands(~Element),~Storage.Buffer<Element>,~(Tensors.Tensor<Element> & Storage.Buffer<Element>),
        Bands.range_into(~Element,geometry,first,rows,channel_offset(geometry,channel,rows),bands,output),bands => output =>
          (Tensors.Banded{bands,geometry,shape,fill,status,workers},output))

def append_tree(~Element: Data,output: Bands.Bands(~Element),+geometry: Bands.Geometry,+channel: U32,input: Tensors.Tensor<Element>) -> Tensors.Tensor<Element> & Bands.Bands(~Element):
  match output:
    case Partition.Leaf{output,+first,+count,details}:
      Storage.bind_pair(~Tensors.Tensor<Element>,~Storage.Buffer<Element>,~(Tensors.Tensor<Element> & Bands.Bands(~Element)),
        rows(~Element,geometry,first,count,channel,input,output),input => output => (input,Partition.Leaf{output,first,count,Unit{}}))
    case Partition.Fork{left,right}:
      Storage.bind_pair(~Tensors.Tensor<Element>,~Bands.Bands(~Element),~(Tensors.Tensor<Element> & Bands.Bands(~Element)),
        append_tree(~Element,left,geometry,channel,input),input => left =>
          Storage.bind_pair(~Tensors.Tensor<Element>,~Bands.Bands(~Element),~(Tensors.Tensor<Element> & Bands.Bands(~Element)),
            append_tree(~Element,right,geometry,channel,input),input => right => (input,Partition.Fork{left,right})))

def channels(shape: List<&2,U32>) -> U32:
  match shape:
    case [channels,height,width]: channels
    case [batch,channels,height,width]: channels
    case shape: 0

type Channel is Data:
  Channel{geometry: Bands.Geometry,channel: U32}

def append_band(~Element: Data,context: Channel,first: U32,rows: U32,output: Storage.Buffer<Element>,input: Storage.Buffer<Element>,resource: Unit) -> Bands.Bands(~Element):
  Channel{+geometry,channel} = context
  +rows = rows
  Partition.Leaf{Bands.destination(~Element,Copy.range(~Element,Bands.local_count(geometry,rows),0,channel_offset(geometry,channel,rows),input,output)),first,rows,Unit{}}

def append_planned(~Element: Data,+parallel: Bool,+geometry: Bands.Geometry,channel: U32,input: Bands.Bands(~Element),planned: Bands.Bands(~Element) & Bands.Plan) -> Bands.Bands(~Element):
  (output,+layout) = planned
  Bands.each(~Bands.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Storage.Buffer<Element>,~Unit,~Channel,~Bands.unshared,
    ~Bands.paired(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Channel,~append_band(~Element)),
    ~Bands.left_of(~Storage.Buffer<Element>,~Storage.Buffer<Element>),
    layout,parallel && Bands.same_image(geometry,layout),parallel,geometry,Channel{geometry,channel},Bands.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,output,input),Unit{})

# Every output leaf receives the same rows of the input, through the shared
# band recursion on the plan of the output. Trees of another shape keep the output.
def append_aligned(~Element: Data,parallel: Bool,geometry: Bands.Geometry,channel: U32,input: Bands.Bands(~Element),output: Bands.Bands(~Element)) -> Bands.Bands(~Element):
  append_planned(~Element,parallel,geometry,channel,input,Bands.plan(~Element,output))

def appended_rows(~Element: Data,owners: Tensors.Tensor<Element> & Bands.Bands(~Element)) -> Bands.Bands(~Element):
  (source,output) = owners
  output

def selected_rows(~Element: Data,same: Bool,+geometry: Bands.Geometry,channel: U32,workers: U32,layout: Bands.Plan,input: Bands.Bands(~Element),output: Bands.Bands(~Element),shape: List<&2,U32>,fill: Element,status: Tensors.Status,previous: U32) -> Bands.Bands(~Element):
  match same:
    case True{}: append_aligned(~Element,BandSchedule.parallel(geometry,workers,8.0,0.0,layout),geometry,channel,input,output)
    case False{}:
      appended_rows(~Element,append_tree(~Element,output,geometry,channel,Tensors.Banded{input,geometry,shape,fill,status,previous}))

def aligned_or_rows(~Element: Data,+geometry: Bands.Geometry,channel: U32,workers: U32,
  input: Bands.Bands(~Element) & Bands.Plan,output: Bands.Bands(~Element) & Bands.Plan,shape: List<&2,U32>,fill: Element,status: Tensors.Status,previous: U32) -> Bands.Bands(~Element):
  (input,source_plan) = input
  (output,+target_plan) = output
  selected_rows(~Element,Bands.same_plan(source_plan,target_plan),geometry,channel,workers,target_plan,input,output,shape,fill,status,previous)

def append_storage(~Element: Data,+geometry: Bands.Geometry,channel: U32,workers: U32,input: Tensors.Tensor<Element>,output: Bands.Bands(~Element)) -> Bands.Bands(~Element):
  match input:
    case Tensors.Tensor{storage,shape,fill,status,previous}:
      appended_rows(~Element,append_tree(~Element,output,geometry,channel,Tensors.Tensor{storage,shape,fill,status,previous}))
    case Tensors.Banded{bands,layout,shape,fill,status,previous}:
      aligned_or_rows(~Element,geometry,channel,workers,Bands.plan(~Element,bands),Bands.plan(~Element,output),shape,fill,status,previous)

def append_input(~Element: Data,geometry: Bands.Geometry,+channel: U32,+count: U32,workers: U32,input: Tensors.Tensor<Element>,output: Bands.Bands(~Element)) -> Bands.Bands(~Element) & U32:
  Bands.Geometry{batch,channels,height,width} = geometry
  (append_storage(~Element,Bands.Geometry{batch,count,height,width},channel,workers,input,output),(channel + count : U32))

def append_part(~Element: Data,geometry: Bands.Geometry,workers: U32,input: Tensors.Tensor<Element>,state: Bands.Bands(~Element) & U32) -> Bands.Bands(~Element) & U32:
  (output,channel) = state
  Tensors.with_metadata(~Element,~(Bands.Bands(~Element) & U32),input,input => shape => fill => status => previous =>
    append_input(~Element,geometry,channel,channels(shape),workers,input,output))

def output(~Element: Data,state: Bands.Bands(~Element) & U32) -> Bands.Bands(~Element):
  (output,channel) = state
  output

def append_parts(~Element: Data,parts: Tensors.Parts<Element>,+geometry: Bands.Geometry,+workers: U32,state: Bands.Bands(~Element) & U32) -> Bands.Bands(~Element):
  match parts:
    case Tensors.Single{input}: output(~Element,append_part(~Element,geometry,workers,input,state))
    case Tensors.Cons{input,rest}: append_parts(~Element,rest,geometry,workers,append_part(~Element,geometry,workers,input,state))

def join(~Element: Data,layout: Bands.Plan,+geometry: Bands.Geometry,parts: Tensors.Parts<Element>,shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  Tensors.Banded{append_parts(~Element,parts,geometry,workers,(allocate(~Element,layout,geometry,fill),0)),geometry,shape,fill,Tensors.Usable{},workers}
