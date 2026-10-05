# Ordered source paths retain the original Parts and their natural prefix.
# Every sample proof is reconstructed from Data; paths contain no callbacks.
import Base
import ./tensor_band_concat_execution.bend as BandConcatExecution
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition

def ordered_value(~Element: Data,~ExtentContext: Data,~extent: ExtentContext -> BandConcatExecution.TensorContent<Element> -> Nat,
  ~SampleContext: Data,~sample: SampleContext -> BandConcatExecution.TensorContent<Element> -> Nat -> Nat -> Nat -> Nat -> Element,
  +context: ExtentContext,+sample_context: SampleContext,parts: BandConcatExecution.PartContents<Element>,+image: Nat,+channel: Nat,+row: Nat,+column: Nat) -> Element:
  match parts:
    case BandConcatExecution.SingleContent{input}: sample(sample_context,input,image,channel,row,column)
    case BandConcatExecution.ConsContent{+input,+rest}:
      Bool.pick(Element,Nat.is_lt(channel,extent(context,input)),sample(sample_context,input,image,channel,row,column),
        ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,rest,image,Nat.sub(channel,extent(context,input)),row,column))

def ordered_head(~Element: Data,~ExtentContext: Data,~extent: ExtentContext -> BandConcatExecution.TensorContent<Element> -> Nat,
  ~SampleContext: Data,~sample: SampleContext -> BandConcatExecution.TensorContent<Element> -> Nat -> Nat -> Nat -> Nat -> Element,
  +context: ExtentContext,+sample_context: SampleContext,+input: BandConcatExecution.TensorContent<Element>,+rest: BandConcatExecution.PartContents<Element>,+image: Nat,+channel: Nat,+row: Nat,+column: Nat,
  inside: {Nat.is_lt(channel,extent(context,input)) == True{} : Bool}) ->
  {ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,BandConcatExecution.ConsContent{input,rest},image,channel,row,column) == sample(sample_context,input,image,channel,row,column) : Element}:
  %Equal.sym(Bool,Nat.is_lt(channel,extent(context,input)),True{},inside) :
    {Bool.pick(Element,_,sample(sample_context,input,image,channel,row,column),ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,rest,image,Nat.sub(channel,extent(context,input)),row,column)) == sample(sample_context,input,image,channel,row,column) : Element}
  {==}

def ordered_tail(~Element: Data,~ExtentContext: Data,~extent: ExtentContext -> BandConcatExecution.TensorContent<Element> -> Nat,
  ~SampleContext: Data,~sample: SampleContext -> BandConcatExecution.TensorContent<Element> -> Nat -> Nat -> Nat -> Nat -> Element,
  +context: ExtentContext,+sample_context: SampleContext,+input: BandConcatExecution.TensorContent<Element>,+rest: BandConcatExecution.PartContents<Element>,+image: Nat,+channel: Nat,+row: Nat,+column: Nat,
  later: {Nat.is_lt(channel,extent(context,input)) == False{} : Bool}) ->
  {ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,BandConcatExecution.ConsContent{input,rest},image,channel,row,column) == ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,rest,image,Nat.sub(channel,extent(context,input)),row,column) : Element}:
  %Equal.sym(Bool,Nat.is_lt(channel,extent(context,input)),False{},later) :
    {Bool.pick(Element,_,sample(sample_context,input,image,channel,row,column),ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,rest,image,Nat.sub(channel,extent(context,input)),row,column)) == ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,rest,image,Nat.sub(channel,extent(context,input)),row,column) : Element}
  {==}

def ordered_shifted_tail(~Element: Data,~ExtentContext: Data,~extent: ExtentContext -> BandConcatExecution.TensorContent<Element> -> Nat,
  ~SampleContext: Data,~sample: SampleContext -> BandConcatExecution.TensorContent<Element> -> Nat -> Nat -> Nat -> Nat -> Element,
  +context: ExtentContext,+sample_context: SampleContext,+input: BandConcatExecution.TensorContent<Element>,+rest: BandConcatExecution.PartContents<Element>,+image: Nat,+channel: Nat,+row: Nat,+column: Nat) ->
  {ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,BandConcatExecution.ConsContent{input,rest},image,Nat.add(extent(context,input),channel),row,column) == ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,rest,image,channel,row,column) : Element}:
  %Order.subtract_shift(extent(context,input),channel) : {ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,BandConcatExecution.ConsContent{input,rest},image,Nat.add(extent(context,input),channel),row,column) == ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,rest,image,_,row,column) : Element}
  ordered_tail(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,input,rest,image,Nat.add(extent(context,input),channel),row,column,
    Order.not_below(extent(context,input),Nat.add(extent(context,input),channel),Order.le_add(extent(context,input),channel)))

type SourcePosition<-Element: Data,-ExtentContext: Data,-extent: ExtentContext -> BandConcatExecution.TensorContent<Element> -> Nat,
  -context: ExtentContext,-input: BandConcatExecution.TensorContent<Element>,-root: BandConcatExecution.PartContents<Element>,-prefix: Nat> is Data:
  SingleSource{root_equal: {root == BandConcatExecution.SingleContent{input} : BandConcatExecution.PartContents<Element>},prefix_zero: {prefix == 0n : Nat}}
  HeadSource{rest: BandConcatExecution.PartContents<Element>,root_equal: {root == BandConcatExecution.ConsContent{input,rest} : BandConcatExecution.PartContents<Element>},prefix_zero: {prefix == 0n : Nat}}
  LaterSource{head: BandConcatExecution.TensorContent<Element>,rest: BandConcatExecution.PartContents<Element>,prior: Nat,
    root_equal: {root == BandConcatExecution.ConsContent{head,rest} : BandConcatExecution.PartContents<Element>},
    position: SourcePosition<Element,ExtentContext,extent,context,input,rest,prior>,prefix_equal: {prefix == Nat.add(extent(context,head),prior) : Nat}}

type PartsPosition<-Element: Data,-ExtentContext: Data,-extent: ExtentContext -> BandConcatExecution.TensorContent<Element> -> Nat,
  -context: ExtentContext,-parts: BandConcatExecution.PartContents<Element>,-root: BandConcatExecution.PartContents<Element>,-prefix: Nat> is Data:
  WholeParts{root_equal: {root == parts : BandConcatExecution.PartContents<Element>},prefix_zero: {prefix == 0n : Nat}}
  LaterParts{head: BandConcatExecution.TensorContent<Element>,rest: BandConcatExecution.PartContents<Element>,prior: Nat,
    root_equal: {root == BandConcatExecution.ConsContent{head,rest} : BandConcatExecution.PartContents<Element>},
    position: PartsPosition<Element,ExtentContext,extent,context,parts,rest,prior>,prefix_equal: {prefix == Nat.add(extent(context,head),prior) : Nat}}

def first_source(~Element: Data,parts: BandConcatExecution.PartContents<Element>) -> BandConcatExecution.TensorContent<Element>:
  match parts:
    case BandConcatExecution.SingleContent{input}: input
    case BandConcatExecution.ConsContent{input,rest}: input

def root_position(~Element: Data,~ExtentContext: Data,~extent: ExtentContext -> BandConcatExecution.TensorContent<Element> -> Nat,
  +context: ExtentContext,+parts: BandConcatExecution.PartContents<Element>) -> PartsPosition<Element,ExtentContext,extent,context,parts,parts,0n>:
  WholeParts{{==},{==}}

def head_position(~Element: Data,~ExtentContext: Data,~extent: ExtentContext -> BandConcatExecution.TensorContent<Element> -> Nat,
  +context: ExtentContext,parts: BandConcatExecution.PartContents<Element>,-root: BandConcatExecution.PartContents<Element>,-prefix: Nat,
  position: PartsPosition<Element,ExtentContext,extent,context,parts,root,prefix>) -> SourcePosition<Element,ExtentContext,extent,context,first_source(~Element,parts),root,prefix>:
  match parts:
    case BandConcatExecution.SingleContent{+input}:
      match position:
        case WholeParts{root_equal,prefix_zero}: SingleSource{root_equal,prefix_zero}
        case LaterParts{+head,+rest,+prior,root_equal,previous,prefix_equal}:
          LaterSource{head,rest,prior,root_equal,head_position(~Element,~ExtentContext,~extent,context,BandConcatExecution.SingleContent{input},rest,prior,previous),prefix_equal}
    case BandConcatExecution.ConsContent{+input,+remaining}:
      match position:
        case WholeParts{root_equal,prefix_zero}: HeadSource{remaining,root_equal,prefix_zero}
        case LaterParts{+head,+rest,+prior,root_equal,previous,prefix_equal}:
          LaterSource{head,rest,prior,root_equal,head_position(~Element,~ExtentContext,~extent,context,BandConcatExecution.ConsContent{input,remaining},rest,prior,previous),prefix_equal}

def tail_position(~Element: Data,~ExtentContext: Data,~extent: ExtentContext -> BandConcatExecution.TensorContent<Element> -> Nat,
  +context: ExtentContext,+input: BandConcatExecution.TensorContent<Element>,+rest: BandConcatExecution.PartContents<Element>,-root: BandConcatExecution.PartContents<Element>,-prefix: Nat,
  position: PartsPosition<Element,ExtentContext,extent,context,BandConcatExecution.ConsContent{input,rest},root,prefix>) ->
  PartsPosition<Element,ExtentContext,extent,context,rest,root,Nat.add(prefix,extent(context,input))>:
  match position:
    case WholeParts{root_equal,+prefix_zero}:
      LaterParts{input,rest,0n,root_equal,WholeParts{{==},{==}},
        Equal.trans(Nat,Nat.add(prefix,extent(context,input)),extent(context,input),Nat.add(extent(context,input),0n),
          %Equal.sym(Nat,prefix,0n,prefix_zero) : {Nat.add(_,extent(context,input)) == extent(context,input) : Nat} {==},
          Equal.sym(Nat,Nat.add(extent(context,input),0n),extent(context,input),Addition.zero_right(extent(context,input))))}
    case LaterParts{+head,+remaining,+prior,root_equal,previous,+prefix_equal}:
      LaterParts{head,remaining,Nat.add(prior,extent(context,input)),root_equal,tail_position(~Element,~ExtentContext,~extent,context,input,rest,remaining,prior,previous),
        Equal.trans(Nat,Nat.add(prefix,extent(context,input)),Nat.add(Nat.add(extent(context,head),prior),extent(context,input)),Nat.add(extent(context,head),Nat.add(prior,extent(context,input))),
          Equal.cong(Nat,Nat,value => Nat.add(value,extent(context,input)),prefix,Nat.add(extent(context,head),prior),prefix_equal),Addition.associative(extent(context,head),prior,extent(context,input)))}

def source_value(~Element: Data,~ExtentContext: Data,~extent: ExtentContext -> BandConcatExecution.TensorContent<Element> -> Nat,
  ~SampleContext: Data,~sample: SampleContext -> BandConcatExecution.TensorContent<Element> -> Nat -> Nat -> Nat -> Nat -> Element,
  +context: ExtentContext,+sample_context: SampleContext,+input: BandConcatExecution.TensorContent<Element>,-root: BandConcatExecution.PartContents<Element>,-prefix: Nat,+image: Nat,+channel: Nat,+row: Nat,+column: Nat,
  +inside: {Nat.is_lt(channel,extent(context,input)) == True{} : Bool},position: SourcePosition<Element,ExtentContext,extent,context,input,root,prefix>) ->
  {sample(sample_context,input,image,channel,row,column) == ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,root,image,Nat.add(prefix,channel),row,column) : Element}:
  match position:
    case SingleSource{+root_equal,+prefix_zero}:
      %Equal.sym(Nat,prefix,0n,prefix_zero) : {sample(sample_context,input,image,channel,row,column) == ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,root,image,Nat.add(_,channel),row,column) : Element}
      %Equal.sym(BandConcatExecution.PartContents<Element>,root,BandConcatExecution.SingleContent{input},root_equal) : {sample(sample_context,input,image,channel,row,column) == ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,_,image,channel,row,column) : Element}
      {==}
    case HeadSource{+rest,+root_equal,+prefix_zero}:
      %Equal.sym(Nat,prefix,0n,prefix_zero) : {sample(sample_context,input,image,channel,row,column) == ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,root,image,Nat.add(_,channel),row,column) : Element}
      %Equal.sym(BandConcatExecution.PartContents<Element>,root,BandConcatExecution.ConsContent{input,rest},root_equal) : {sample(sample_context,input,image,channel,row,column) == ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,_,image,channel,row,column) : Element}
      Equal.sym(Element,ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,BandConcatExecution.ConsContent{input,rest},image,channel,row,column),sample(sample_context,input,image,channel,row,column),
        ordered_head(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,input,rest,image,channel,row,column,inside))
    case LaterSource{+head,+rest,+prior,+root_equal,previous,+prefix_equal}:
      %Equal.sym(Nat,prefix,Nat.add(extent(context,head),prior),prefix_equal) : {sample(sample_context,input,image,channel,row,column) == ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,root,image,Nat.add(_,channel),row,column) : Element}
      %Equal.sym(BandConcatExecution.PartContents<Element>,root,BandConcatExecution.ConsContent{head,rest},root_equal) : {sample(sample_context,input,image,channel,row,column) == ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,_,image,Nat.add(Nat.add(extent(context,head),prior),channel),row,column) : Element}
      Equal.trans(Element,sample(sample_context,input,image,channel,row,column),ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,rest,image,Nat.add(prior,channel),row,column),ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,BandConcatExecution.ConsContent{head,rest},image,Nat.add(Nat.add(extent(context,head),prior),channel),row,column),
        source_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,input,rest,prior,image,channel,row,column,inside,previous),
        %Equal.sym(Nat,Nat.add(Nat.add(extent(context,head),prior),channel),Nat.add(extent(context,head),Nat.add(prior,channel)),Addition.associative(extent(context,head),prior,channel)) : {ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,rest,image,Nat.add(prior,channel),row,column) == ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,BandConcatExecution.ConsContent{head,rest},image,_,row,column) : Element}
        Equal.sym(Element,ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,BandConcatExecution.ConsContent{head,rest},image,Nat.add(extent(context,head),Nat.add(prior,channel)),row,column),ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,rest,image,Nat.add(prior,channel),row,column),
          ordered_shifted_tail(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,head,rest,image,Nat.add(prior,channel),row,column)))

def parts_value(~Element: Data,~ExtentContext: Data,~extent: ExtentContext -> BandConcatExecution.TensorContent<Element> -> Nat,
  ~SampleContext: Data,~sample: SampleContext -> BandConcatExecution.TensorContent<Element> -> Nat -> Nat -> Nat -> Nat -> Element,
  +context: ExtentContext,+sample_context: SampleContext,+parts: BandConcatExecution.PartContents<Element>,-root: BandConcatExecution.PartContents<Element>,-prefix: Nat,+image: Nat,+channel: Nat,+row: Nat,+column: Nat,
  position: PartsPosition<Element,ExtentContext,extent,context,parts,root,prefix>) ->
  {ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,parts,image,channel,row,column) == ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,root,image,Nat.add(prefix,channel),row,column) : Element}:
  match position:
    case WholeParts{+root_equal,+prefix_zero}:
      %Equal.sym(Nat,prefix,0n,prefix_zero) : {ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,parts,image,channel,row,column) == ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,root,image,Nat.add(_,channel),row,column) : Element}
      %Equal.sym(BandConcatExecution.PartContents<Element>,root,parts,root_equal) : {ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,parts,image,channel,row,column) == ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,_,image,channel,row,column) : Element}
      {==}
    case LaterParts{+head,+rest,+prior,+root_equal,previous,+prefix_equal}:
      %Equal.sym(Nat,prefix,Nat.add(extent(context,head),prior),prefix_equal) : {ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,parts,image,channel,row,column) == ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,root,image,Nat.add(_,channel),row,column) : Element}
      %Equal.sym(BandConcatExecution.PartContents<Element>,root,BandConcatExecution.ConsContent{head,rest},root_equal) : {ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,parts,image,channel,row,column) == ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,_,image,Nat.add(Nat.add(extent(context,head),prior),channel),row,column) : Element}
      Equal.trans(Element,ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,parts,image,channel,row,column),ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,rest,image,Nat.add(prior,channel),row,column),ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,BandConcatExecution.ConsContent{head,rest},image,Nat.add(Nat.add(extent(context,head),prior),channel),row,column),
        parts_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,parts,rest,prior,image,channel,row,column,previous),
        %Equal.sym(Nat,Nat.add(Nat.add(extent(context,head),prior),channel),Nat.add(extent(context,head),Nat.add(prior,channel)),Addition.associative(extent(context,head),prior,channel)) : {ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,rest,image,Nat.add(prior,channel),row,column) == ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,BandConcatExecution.ConsContent{head,rest},image,_,row,column) : Element}
        Equal.sym(Element,ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,BandConcatExecution.ConsContent{head,rest},image,Nat.add(extent(context,head),Nat.add(prior,channel)),row,column),ordered_value(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,rest,image,Nat.add(prior,channel),row,column),
          ordered_shifted_tail(~Element,~ExtentContext,~extent,~SampleContext,~sample,context,sample_context,head,rest,image,Nat.add(prior,channel),row,column)))
