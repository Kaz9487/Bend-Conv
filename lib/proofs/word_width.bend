# Width transport changes the type index, never the stored bits.
import Base

def transport(-source: Nat,-target: Nat,value: Word(source),width: {source == target : Nat}) -> Word(target):
  %width : Word(_)
  value

law transport_cons:
  for +source: Nat
  for +target: Nat
  for +bit: Bool
  for +tail: Word(source)
  for width: {source == target : Nat}
  {transport(1n+source,1n+target,WCon{bit,tail},Equal.cong(Nat,Nat,n => 1n+n,source,target,width)) == WCon{bit,transport(source,target,tail,
        width)} : Word(1n+target)}

def transport_cons(source,target,bit,tail,width):
  %h@width : {transport(1n+source,1n+_,WCon{bit,tail},Equal.cong(Nat,Nat,n => 1n+n,source,_,h)) == WCon{bit,transport(source,_,tail,h)} : Word(1n+_)}
  {==}
