# A reusable finite-sequence witness; range membership yields a bounded index.
import Base
import ../ordered_matrix.bend as OrderedMatrix
import ./ordered_ranges.bend as Ranges
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding
import ./storage_shape.bend as Shape

def All(-Element: Data,-Property: Element -> Data,items: List<&2,Element>) -> Data:
  match items:
    case Nil{}: Unit
    case item <> rest: Shape.Both<Property(item),All(Element,Property,rest)>

def all_concatenated(-Element: Data,-Property: Element -> Data,+left: List<&2,Element>,+right: List<&2,Element>,
  left_proof: All(Element,Property,left),right_proof: All(Element,Property,right)) -> All(Element,Property,OrderedMatrix.cat(Element,left,right)):
  match left:
    case Nil{}: right_proof
    case +item <> +rest:
      Shape.Both{head,tail} = left_proof
      Shape.Both{head,all_concatenated(Element,Property,rest,right,tail,right_proof)}

def transform_all(~Element: Data,~Context: Data,~Before: @+context: Context -> @+element: Element -> Data,~After: @+context: Context -> @+element: Element -> Data,
  ~transform: @+context: Context -> @+element: Element -> @+evidence: Before(context,element) -> After(context,element),
  +items: List<&2,Element>,+context: Context,evidence: All(Element,{element => Before(context,element) : Element -> Data},items)) ->
  All(Element,{element => After(context,element) : Element -> Data},items):
  match items:
    case Nil{}: Unit{}
    case +element <> +rest:
      Shape.Both{head,tail} = evidence
      Shape.Both{transform(context,element,head),transform_all(~Element,~Context,~Before,~After,~transform,rest,context,tail)}

def Member(-A: Data,+value: A,values: List<&2,A>) -> Data:
  match values:
    case Nil{}: Empty
    case head <> rest: Either<&2,&2,{value == head : A},Member(A,value,rest)>

def all_member(-Element: Data,-Property: Element -> Data,+value: Element,items: List<&2,Element>,
  all: All(Element,Property,items),member: Member(Element,value,items)) -> Property(value):
  match items:
    case []: Empty.absurd(Property(value),member)
    case +head <> +rest:
      Shape.Both{head_proof,tail_proof} = all
      match member:
        case Inl{equal}:
          %Equal.sym(Element,value,head,equal) : Property(_)
          head_proof
        case Inr{tail}: all_member(Element,Property,value,rest,tail_proof,tail)

def lift_split(-A: Data,+value: A,+head: A,+left: List<&2,A>,+right: List<&2,A>,
  choice: Either<&2,&2,Member(A,value,left),Member(A,value,right)>) -> Either<&2,&2,Member(A,value,head <> left),Member(A,value,right)>:
  match choice:
    case Inl{member}: Inl{Inr{member}}
    case Inr{member}: Inr{member}

def split_concatenated(-A: Data,+value: A,+left: List<&2,A>,+right: List<&2,A>,member: Member(A,value,OrderedMatrix.cat(A,left,right))) ->
  Either<&2,&2,Member(A,value,left),Member(A,value,right)>:
  match left:
    case Nil{}: Inr{member}
    case +head <> +rest:
      match member:
        case Inl{equal}: Inl{Inl{equal}}
        case Inr{tail}: lift_split(A,value,head,rest,right,split_concatenated(A,value,rest,right,tail))

def range_member(+count: Nat,+start: Nat,+index: Nat,inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  Member(Nat,Nat.add(start,index),Ranges.range(count,start)):
  match count index:
    case 0n 0n: Empty.absurd(Member(Nat,Nat.add(start,0n),[]),WordEncoding.false_true(inside))
    case 0n 1n+index: Empty.absurd(Member(Nat,Nat.add(start,1n+index),[]),WordEncoding.false_true(inside))
    case 1n+rest 0n: Inl{Addition.zero_right(start)}
    case 1n+ +rest 1n+ +previous:
      Inr{
        %Equal.sym(Nat,Nat.add(start,1n+previous),1n+Nat.add(start,previous),Addition.successor_right(start,previous)) : Member(Nat,_,Ranges.range(rest,1n+start))
        range_member(rest,1n+start,previous,inside)}

type RangeIndex<-count: Nat,-start: Nat,-value: Nat> is Data:
  RangeIndex{index: Nat,inside: {Nat.is_lt(index,count) == True{} : Bool},exact: {Nat.add(start,index) == value : Nat}}

def lift_range_index(+count: Nat,+start: Nat,+value: Nat,proof: RangeIndex<count,1n+start,value>) -> RangeIndex<1n+count,start,value>:
  RangeIndex{+index,inside,exact} = proof
  RangeIndex{1n+index,inside,
    Equal.trans(Nat,Nat.add(start,1n+index),1n+Nat.add(start,index),value,Addition.successor_right(start,index),exact)}

def range_index(+count: Nat,+start: Nat,+value: Nat,member: Member(Nat,value,Ranges.range(count,start))) -> RangeIndex<count,start,value>:
  match count:
    case 0n: Empty.absurd(RangeIndex<0n,start,value>,member)
    case 1n+ +rest:
      match member:
        case Inl{equal}:
          RangeIndex{0n,{==},Equal.trans(Nat,Nat.add(start,0n),start,value,Addition.zero_right(start),Equal.sym(Nat,value,start,equal))}
        case Inr{tail}: lift_range_index(rest,start,value,range_index(rest,1n+start,value,tail))
