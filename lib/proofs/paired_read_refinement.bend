import Base
import ../kernel_block.bend as Block

def read_at_refines(
  ~Native: Type,~Context: Data,~Element: Data,~Index: Data,
  ~read: Native -> Context -> Index -> Native & Element,
  -source: Native,+context: Context,+first: Index,+second: Index,+left: Element,+right: Element,
  first_read: {read(source,context,first) == (source,left) : Native & Element},
  second_read: {read(source,context,second) == (source,right) : Native & Element}
) -> {Block.read_at(~Native,~Context,~Element,~Index,~read,source,context,first,second) ==
  (source,Block.Twin{left,right}) : Native & Block.Twin<Element>}:
  %Equal.sym(Native & Element,read(source,context,first),(source,left),first_read) :
    {Block.read_second(~Native,~Context,~Element,~Index,~read,_,context,second) ==
      (source,Block.Twin{left,right}) : Native & Block.Twin<Element>}
  %Equal.sym(Native & Element,read(source,context,second),(source,right),second_read) :
    {Block.finish_pair(~Native,~Element,left,_) == (source,Block.Twin{left,right}) : Native & Block.Twin<Element>}
  {==}

def select(~Element: Data,second: Bool,pair: Block.Twin<Element>) -> Element:
  Block.Twin{left,right} = pair
  Bool.pick(Element,second,right,left)

def zip_preserves_selection(~Element: Data,~Context: Data,~combine: Element -> Element -> Context -> Element,
  +second: Bool,+left: Block.Twin<Element>,+right: Block.Twin<Element>,+context: Context) ->
  {select(~Element,second,Block.zip_map(~Element,~Context,~combine,left,right,context)) ==
    combine(select(~Element,second,left),select(~Element,second,right),context) : Element}:
  match second left right:
    case False{} Block.Twin{left_first,left_second} Block.Twin{right_first,right_second}: {==}
    case True{} Block.Twin{left_first,left_second} Block.Twin{right_first,right_second}: {==}

# Pair composition is independent of element shape and block width.
def pair_sample(
  ~Model: Data,~Context: Data,~Element: Data,
  ~sample: Model -> Context -> Nat -> Element,~stride: Nat,
  model: Model,context: Context,index: Nat
) -> Block.Twin<Element>:
  +model = model
  +context = context
  +index = index
  Block.Twin{sample(model,context,index),sample(model,context,Nat.add(index,stride))}

def read_pair_refines(
  ~Native: Type,~Model: Data,~Context: Data,~Element: Data,
  ~pack: Model -> Native,
  ~read: Native -> Context -> Nat -> Native & Element,
  ~sample: Model -> Context -> Nat -> Element,
  ~stride: Nat,+model: Model,+context: Context,+index: Nat,
  first_read: {read(pack(model),context,index) == (pack(model),sample(model,context,index)) : Native & Element},
  second_read: {read(pack(model),context,Nat.add(index,stride)) == (pack(model),sample(model,context,Nat.add(index,stride))) : Native & Element}
) -> {Block.read_pair(~Native,~Context,~Element,~read,~stride,pack(model),context,index) == (pack(model),pair_sample(~Model,~Context,
    ~Element,~sample,~stride,model,context,index)) : Native & Block.Twin<Element>}:
  read_at_refines(~Native,~Context,~Element,~Nat,~read,pack(model),context,index,Nat.add(index,stride),
    sample(model,context,index),sample(model,context,Nat.add(index,stride)),first_read,second_read)
