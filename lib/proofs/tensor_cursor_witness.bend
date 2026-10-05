import Base
import ../tensor_cursor.bend as Cursor
import ../traversal_array.bend as Traversal
import ./storage_witness.bend as Witness
import ./traversal_witness.bend as TraversalWitness
import ./traversal_visit_witness.bend as VisitWitness

def read_value(~Element: Data,axes: Cursor.Levels(),address: U32,-storage: Array<Element>,-result: Array<Element> & Element,
  observed: Witness.Read<Element,storage,result>) ->
  VisitWitness.Sample<Array<Element>,Element,Cursor.Cursor,storage,Cursor.read_value(~Element,axes,address,result)>:
  match observed:
    case Witness.Read{value,equation}:
      %Equal.sym(Array<Element> & Element,result,(storage,value),equation) :
        VisitWitness.Sample<Array<Element>,Element,Cursor.Cursor,storage,Cursor.read_value(~Element,axes,address,_)>
      VisitWitness.Sample{Cursor.advance(axes,address),value,{==}}

def read(~Element: Data,-storage: Array<Element>,cursor: Cursor.Cursor,evidence: Witness.ArrayWitness<Element,storage>) ->
  VisitWitness.Sample<Array<Element>,Element,Cursor.Cursor,storage,Cursor.read(~Element,storage,cursor)>:
  Cursor.Cursor{axes,+address} = cursor
  read_value(~Element,axes,address,storage,Array.get(Element,storage,address),Witness.observed(Element,storage,address,evidence))

def stride_value(~Element: Data,address: U32,+stride: U32,-storage: Array<Element>,-result: Array<Element> & Element,
  observed: Witness.Read<Element,storage,result>) ->
  VisitWitness.Sample<Array<Element>,Element,Cursor.Stride,storage,Cursor.stride_value(~Element,address,stride,result)>:
  match observed:
    case Witness.Read{value,equation}:
      %Equal.sym(Array<Element> & Element,result,(storage,value),equation) :
        VisitWitness.Sample<Array<Element>,Element,Cursor.Stride,storage,Cursor.stride_value(~Element,address,stride,_)>
      VisitWitness.Sample{Cursor.Stride{(address + stride : U32),stride},value,{==}}

def read_stride(~Element: Data,-storage: Array<Element>,cursor: Cursor.Stride,evidence: Witness.ArrayWitness<Element,storage>) ->
  VisitWitness.Sample<Array<Element>,Element,Cursor.Stride,storage,Cursor.read_stride(~Element,storage,cursor)>:
  Cursor.Stride{+address,+stride} = cursor
  stride_value(~Element,address,stride,storage,Array.get(Element,storage,address),Witness.observed(Element,storage,address,evidence))
