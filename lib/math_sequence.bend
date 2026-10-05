# Element- and length-indexed sequence model. Random get is O(index);
# runtime dense storage is Array<Element>, not this sequential observation.
import Base

law Values:
  for -Element: Data
  for length: Nat
  Data

type Elements<-Element: Data,-length: Nat> is Data:
  Elements{head: Element,tail: Values(Element,length)}

def Values(Element,length):
  match length:
    case 0n: Unit
    case 1n+rest: Elements<Element,rest>

law Index:
  for length: Nat
  Data

type Position<-length: Nat> is Data:
  First{}
  Next{index: Index(length)}

def Index(length):
  match length:
    case 0n: Empty
    case 1n+rest: Position<rest>

def index_value(length: Nat,index: Index(length)) -> Nat:
  match length:
    case 0n:
      match index:
    case 1n+rest:
      match index:
        case First{}: 0n
        case Next{index}: 1n+index_value(rest,index)

def tabulate(~Element: Data,~Context: Data,~sample: Context -> Nat -> Element,length: Nat,+offset: Nat,+context: Context) -> Values(Element,length):
  match length:
    case 0n: Unit{}
    case 1n+rest:
      Elements{sample(context,offset),tabulate(~Element,~Context,~sample,rest,1n+offset,context)}

def get(-Element: Data,length: Nat,index: Index(length),vector: Values(Element,length)) -> Element:
  match length:
    case 0n:
      match index:
    case 1n+rest:
      match index:
        case First{}:
          Elements{head,tail} = vector
          head
        case Next{index}:
          Elements{head,tail} = vector
          get(Element,rest,index,tail)

def replicate(-Element: Data,length: Nat,+value: Element) -> Values(Element,length):
  match length:
    case 0n: Unit{}
    case 1n+rest: Elements{value,replicate(Element,rest,value)}


def zip_map(~Element: Data,~Context: Data,~combine: Context -> Element -> Element -> Element,
  length: Nat,+context: Context,left: Values(Element,length),right: Values(Element,length)) -> Values(Element,length):
  match length:
    case 0n: Unit{}
    case 1n+rest:
      Elements{left_head,left_tail} = left
      Elements{right_head,right_tail} = right
      Elements{combine(context,left_head,right_head),zip_map(~Element,~Context,~combine,rest,context,left_tail,right_tail)}
