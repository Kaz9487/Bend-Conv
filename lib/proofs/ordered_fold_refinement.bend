# A left fold of samples, preserving their exact order and the reader owner.
# Both dense and band window readers instantiate this common proof.
import Base
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop

def values(~Source: Data,~Element: Data,~Accumulator: Data,~Context: Data,
  ~sample: Source -> Context -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,
  count: Nat,+position: U32,+source: Source,+context: Context,accumulator: Accumulator) -> Accumulator:
  match count:
    case 0n: accumulator
    case 1n+rest:
      values(~Source,~Element,~Accumulator,~Context,~sample,~operation,rest,(position + 1 : U32),source,context,operation(accumulator,sample(source,context,position)))

def fold(~Reader: Type,~Source: Data,~Element: Data,~Accumulator: Data,~Context: Data,
  ~pack: Source -> Reader,~read: Reader -> Context -> U32 -> Reader & Element,
  ~sample: Source -> Context -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,
  ~read_matches: @+source: Source -> @+context: Context -> @+position: U32 -> {read(pack(source),context,position) == (pack(source),sample(source,context,position)) : Reader & Element},
  count: Nat,+position: U32,+source: Source,+context: Context,+initial: Accumulator) ->
  {Traversal.fold(~Reader,~Element,~Accumulator,~Context,~read,~operation,count,position,pack(source),initial,context) ==
    (pack(source),values(~Source,~Element,~Accumulator,~Context,~sample,~operation,count,position,source,context,initial)) : Reader & Accumulator}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Reader & Element,read(pack(source),context,position),(pack(source),sample(source,context,position)),read_matches(source,context,position)) :
        {Traversal.finish_fold(~Reader,~Accumulator,~Context,
          Loop.run(~Traversal.State<&1,&2,Reader,Accumulator,Context>,~Traversal.fold_step(~Reader,~Element,~Accumulator,~Context,~read,~operation),rest,
            Traversal.fold_value(~Reader,~Element,~Accumulator,~Context,~operation,initial,position,context,_))) ==
          (pack(source),values(~Source,~Element,~Accumulator,~Context,~sample,~operation,1n+rest,position,source,context,initial)) : Reader & Accumulator}
      fold(~Reader,~Source,~Element,~Accumulator,~Context,~pack,~read,~sample,~operation,~read_matches,rest,(position + 1 : U32),source,context,
        operation(initial,sample(source,context,position)))
