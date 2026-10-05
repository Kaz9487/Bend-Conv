import Base
import ./storage_certificate.bend as Certificate
import ./storage_witness.bend as Witness
import ./traversal_witness.bend as TraversalWitness

type Copies<-Element: Data,-pair: Array<Element> & Array<Element>> is Data:
  Copies{left: Certificate.Certificate<Element,TraversalWitness.first(~Array<Element>,~Array<Element>,pair)>,right: Certificate.Certificate<Element,TraversalWitness.second(~Array<Element>,~Array<Element>,pair)>}

def clone(~Element: Data,-input: Array<Element>,evidence: Certificate.Certificate<Element,input>,left: Certificate.Certificate<Element,input>,right: Certificate.Certificate<Element,input>) ->
  Copies<Element,Array.clone(Element,input)>:
  Certificate.Certificate{+depth,equation} = evidence
  %Equal.sym(Array<Element> & Array<Element>,Array.clone(Element,input),(input,input),
    Witness.clone(Element,depth,input,Certificate.witness_at(Element,depth,input,equation))) : Copies<Element,_>
  Copies{left,right}
