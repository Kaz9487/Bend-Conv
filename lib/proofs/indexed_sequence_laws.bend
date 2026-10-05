import Base
import ./natural_addition.bend as Addition
import ../math_sequence.bend as Sequence
import ./nat_division.bend as Division
import ./word_encoding.bend as WordEncoding
import ./binary_radix.bend as Radix

def get_tabulated(~Element: Data,~Context: Data,~sample: Context -> Nat -> Element,+length: Nat,+index: Sequence.Index(length),+offset: Nat,
  +context: Context) -> {Sequence.get(Element,length,index,Sequence.tabulate(~Element,~Context,~sample,length,offset,context)) == sample(context,
    Nat.add(Sequence.index_value(length,index),offset)) : Element}:
  match length:
    case 0n:
      match index:
    case 1n+ +rest:
      match index:
        case Sequence.First{}: {==}
        case Sequence.Next{+previous}:
          %Addition.successor_right(Sequence.index_value(rest,previous),offset) : {Sequence.get(Element,rest,previous,Sequence.tabulate(~Element,~Context,
                ~sample,rest,1n+offset,context)) == sample(context,_) : Element}
          get_tabulated(~Element,~Context,~sample,rest,previous,1n+offset,context)

law get_replicated:
  for -Element: Data
  for +length: Nat
  for +index: Sequence.Index(length)
  for +value: Element
  {Sequence.get(Element,length,index,Sequence.replicate(Element,length,value)) == value : Element}
def get_replicated(Element,length,index,value):
  match length:
    case 0n:
      match index:
    case 1n+ +rest:
      match index:
        case Sequence.First{}: {==}
        case Sequence.Next{+index}: get_replicated(Element,rest,index,value)

def zip_map_at_index(
  ~Element: Data,~Context: Data,~combine: Context -> Element -> Element -> Element,
  +length: Nat,+index: Sequence.Index(length),+context: Context,
  +left: Sequence.Values(Element,length),+right: Sequence.Values(Element,length)
) -> {Sequence.get(Element,length,index,Sequence.zip_map(~Element,~Context,~combine,length,context,left,right)) ==
  combine(context,Sequence.get(Element,length,index,left),Sequence.get(Element,length,index,right)) : Element}:
  match length:
    case 0n:
      match index:
    case 1n+ +rest:
      match index:
        case Sequence.First{}:
          Sequence.Elements{left_head,left_tail} = left
          Sequence.Elements{right_head,right_tail} = right
          {==}
        case Sequence.Next{+previous}:
          Sequence.Elements{left_head,+left_tail} = left
          Sequence.Elements{right_head,+right_tail} = right
          zip_map_at_index(~Element,~Context,~combine,rest,previous,context,left_tail,right_tail)
