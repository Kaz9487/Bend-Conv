# Canonical proof representation of the real band owners and their metadata.
# This representation is used only by refinement proofs, never by execution.
import Base
import ../tensor_band.bend as Bands
import ../traversal_partition.bend as Partition
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape

type Model<-Element: Data> is Data:
  Leaf{buffer: Buffer.Model<Element>,first: U32,rows: U32}
  Fork{left: Model<Element>,right: Model<Element>}

type Content<-Element: Data> is Data:
  RowBand{buffer: Buffer.Content<Element>,first: U32,rows: U32}
  Joined{left: Content<Element>,right: Content<Element>}

def contents(~Element: Data,bands: Bands.Bands(~Element)) -> Content<Element>:
  match bands:
    case Partition.Leaf{buffer,first,rows,details}: RowBand{Buffer.contents(~Element,buffer),first,rows}
    case Partition.Fork{left,right}: Joined{contents(~Element,left),contents(~Element,right)}

def content(~Element: Data,model: Model<Element>) -> Content<Element>:
  match model:
    case Leaf{buffer,first,rows}: RowBand{Buffer.content(~Element,buffer),first,rows}
    case Fork{left,right}: Joined{content(~Element,left),content(~Element,right)}

def Ready(~Element: Data,model: Model<Element>) -> Data:
  match model:
    case Leaf{input,first,rows}: Buffer.Ready<Element,input>
    case Fork{left,right}: Shape.Both<Ready(~Element,left),Ready(~Element,right)>

def ContentReady(~Element: Data,content: Content<Element>) -> Data:
  match content:
    case RowBand{buffer,first,rows}: Buffer.ContentReady<Element,buffer>
    case Joined{left,right}: Shape.Both<ContentReady(~Element,left),ContentReady(~Element,right)>

def ready(~Element: Data,model: Model<Element>,evidence: ContentReady(~Element,content(~Element,model))) -> Ready(~Element,model):
  match model:
    case Leaf{input,first,rows}: Buffer.ready_content(~Element,input,evidence)
    case Fork{+left,+right}:
      Shape.Both{left_ready,right_ready} = evidence
      Shape.Both{ready(~Element,left,left_ready),ready(~Element,right,right_ready)}

def pack(~Element: Data,model: Model<Element>) -> Bands.Bands(~Element):
  match model:
    case Leaf{buffer,first,rows}: Partition.Leaf{Buffer.pack(~Element,buffer),first,rows,Unit{}}
    case Fork{left,right}: Partition.Fork{pack(~Element,left),pack(~Element,right)}

def packed_contents(~Element: Data,+model: Model<Element>) -> {contents(~Element,pack(~Element,model)) == content(~Element,model) : Content<Element>}:
  match model:
    case Leaf{buffer,+first,+rows}:
      Equal.cong(Buffer.Content<Element>,Content<Element>,buffer => RowBand{buffer,first,rows},
        Buffer.contents(~Element,Buffer.pack(~Element,buffer)),Buffer.content(~Element,buffer),Buffer.packed_contents(~Element,buffer))
    case Fork{+left,+right}:
      %Equal.sym(Content<Element>,contents(~Element,pack(~Element,left)),content(~Element,left),packed_contents(~Element,left)) :
        {Joined{_,contents(~Element,pack(~Element,right))} == Joined{content(~Element,left),content(~Element,right)} : Content<Element>}
      %Equal.sym(Content<Element>,contents(~Element,pack(~Element,right)),content(~Element,right),packed_contents(~Element,right)) :
        {Joined{content(~Element,left),_} == Joined{content(~Element,left),content(~Element,right)} : Content<Element>}
      {==}

def layout(~Element: Data,model: Model<Element>) -> Bands.Plan:
  match model:
    case Leaf{buffer,first,rows}: Bands.Rows{first,rows}
    case Fork{left,right}: Bands.Split{layout(~Element,left),layout(~Element,right)}

def content_layout(~Element: Data,content: Content<Element>) -> Bands.Plan:
  match content:
    case RowBand{buffer,first,rows}: Bands.Rows{first,rows}
    case Joined{left,right}: Bands.Split{content_layout(~Element,left),content_layout(~Element,right)}

def layout_matches_content(~Element: Data,+model: Model<Element>) -> {layout(~Element,model) == content_layout(~Element,content(~Element,model)) : Bands.Plan}:
  match model:
    case Leaf{buffer,first,rows}: {==}
    case Fork{+left,+right}:
      %layout_matches_content(~Element,left) : {Bands.Split{layout(~Element,left),layout(~Element,right)} == Bands.Split{_,content_layout(~Element,content(~Element,right))} : Bands.Plan}
      %layout_matches_content(~Element,right) : {Bands.Split{layout(~Element,left),layout(~Element,right)} == Bands.Split{layout(~Element,left),_} : Bands.Plan}
      {==}

def plan(~Element: Data,+model: Model<Element>) ->
  {Bands.plan(~Element,pack(~Element,model)) == (pack(~Element,model),layout(~Element,model)) : Bands.Bands(~Element) & Bands.Plan}:
  match model:
    case Leaf{buffer,first,rows}: {==}
    case Fork{+left,+right}:
      %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,pack(~Element,left)),(pack(~Element,left),layout(~Element,left)),plan(~Element,left)) :
        {Bands.plan_join(~Element,_,Bands.plan(~Element,pack(~Element,right))) == (pack(~Element,Fork{left,right}),layout(~Element,Fork{left,right})) : Bands.Bands(~Element) & Bands.Plan}
      %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,pack(~Element,right)),(pack(~Element,right),layout(~Element,right)),plan(~Element,right)) :
        {Bands.plan_join(~Element,(pack(~Element,left),layout(~Element,left)),_) == (pack(~Element,Fork{left,right}),layout(~Element,Fork{left,right})) : Bands.Bands(~Element) & Bands.Plan}
      {==}

type Observation<-Element: Data,-bands: Bands.Bands(~Element)> is Data:
  Observation{model: Model<Element>,equation: {pack(~Element,model) == bands : Bands.Bands(~Element)}}

def leaf(~Element: Data,+first: U32,+rows: U32,-buffer: Storage.Buffer<Element>,observation: Buffer.Observation<Element,buffer>) ->
  Observation<Element,Partition.Leaf{buffer,first,rows,Unit{}}>:
  Buffer.Observation{+model,equation} = observation
  Observation{Leaf{model,first,rows},Equal.cong(Storage.Buffer<Element>,Bands.Bands(~Element),
    owner => Partition.Leaf{owner,first,rows,Unit{}},Buffer.pack(~Element,model),buffer,equation)}

def fork(~Element: Data,-left: Bands.Bands(~Element),-right: Bands.Bands(~Element),
  left_observation: Observation<Element,left>,right_observation: Observation<Element,right>) -> Observation<Element,Partition.Fork{left,right}>:
  Observation{+left_model,left_equal} = left_observation
  Observation{+right_model,right_equal} = right_observation
  Observation{Fork{left_model,right_model},
    %left_equal : {Partition.Fork{pack(~Element,left_model),pack(~Element,right_model)} == Partition.Fork{_,right} : Bands.Bands(~Element)}
    %right_equal : {Partition.Fork{pack(~Element,left_model),pack(~Element,right_model)} == Partition.Fork{pack(~Element,left_model),_} : Bands.Bands(~Element)}
    {==}}

def observe(~Element: Data,bands: Bands.Bands(~Element)) -> Observation<Element,bands>:
  match bands:
    case Partition.Leaf{buffer,+first,+rows,Unit{}}: leaf(~Element,first,rows,buffer,Buffer.observe(~Element,buffer))
    case Partition.Fork{left,right}: fork(~Element,left,right,observe(~Element,left),observe(~Element,right))

def lift(~Element: Data,~Context: Data,~Proposition: Context -> @-bands: Bands.Bands(~Element) -> Type,+context: Context,-bands: Bands.Bands(~Element),
  observation: Observation<Element,bands>,canonical: @+model: Model<Element> -> Proposition(context,pack(~Element,model))) -> Proposition(context,bands):
  Observation{model,equation} = observation
  %equation : Proposition(context,_)
  canonical(model)
