# Shared natural-number addition laws; induction is on a symbolic operand.
import Base

law successor_right:
  for +a: Nat
  for +b: Nat
  {Nat.add(a,1n+b) == 1n+Nat.add(a,b) : Nat}

def successor_right(a,b):
  match a:
    case 0n:
      {==}
    case 1n+ +p:
      %successor_right(p,b) : {1n+Nat.add(p,1n+b) == 1n+_ : Nat}
      {==}

law double:
  for +n: Nat
  {Nat.add(n,n) == Nat.double(n) : Nat}

def double(n):
  match n:
    case 0n:
      {==}
    case 1n+ +p:
      %double(p) : {1n+Nat.add(p,1n+p) == 2n+_ : Nat}
      %successor_right(p,p) : {1n+Nat.add(p,1n+p) == 1n+_ : Nat}
      {==}

law zero_right:
  for +n: Nat
  {Nat.add(n,0n) == n : Nat}

def zero_right(n):
  match n:
    case 0n:
      {==}
    case 1n+ +p:
      %zero_right(p) : {1n+Nat.add(p,0n) == 1n+_ : Nat}
      {==}

law successor_width:
  for +width: Nat
  {Nat.add(width,1n) == 1n+width : Nat}

def successor_width(width):
  match width:
    case 0n:
      {==}
    case 1n+ +rest:
      Equal.cong(Nat,Nat,n => 1n+n,Nat.add(rest,1n),1n+rest,successor_width(rest))

law associative:
  for +first: Nat
  for +second: Nat
  for +third: Nat
  {Nat.add(Nat.add(first,second),third) == Nat.add(first,Nat.add(second,third)) : Nat}

def associative(first,second,third):
  match first:
    case 0n:
      {==}
    case 1n+ +rest:
      Equal.cong(Nat,Nat,n => 1n+n,Nat.add(Nat.add(rest,second),third),Nat.add(rest,Nat.add(second,third)),associative(rest,second,third))

