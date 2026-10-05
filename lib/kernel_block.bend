import Base

# A statically shaped product. Pairing readers preserves one buffer owner.
type Twin<-Element: Data> is Data:
  Twin{first: Element,second: Element}

# Static product composition also lifts an operation to any pair of elements.
# Nesting this template gives wider products without another arithmetic kernel.
def zip_map(~Element: Data,~Context: Data,~combine: Element -> Element -> Context -> Element,
  left: Twin<Element>,right: Twin<Element>,context: Context) -> Twin<Element>:
  Twin{left_first,left_second} = left
  Twin{right_first,right_second} = right
  +context = context
  Twin{combine(left_first,right_first,context),combine(left_second,right_second,context)}

def finish_pair(~State: Type,~Element: Data,first: Element,result: State & Element) -> State & Twin<Element>:
  (state,second) = result
  (state,Twin{first,second})

def read_second(
  ~State: Type,~Context: Data,~Element: Data,~Index: Data,
  ~read: State -> Context -> Index -> State & Element,
  result: State & Element,context: Context,index: Index
) -> State & Twin<Element>:
  (state,first) = result
  finish_pair(~State,~Element,first,read(state,context,index))

def read_at(
  ~State: Type,~Context: Data,~Element: Data,~Index: Data,
  ~read: State -> Context -> Index -> State & Element,
  state: State,+context: Context,first: Index,second: Index
) -> State & Twin<Element>:
  read_second(~State,~Context,~Element,~Index,~read,read(state,context,first),context,second)

def read_pair(
  ~State: Type,~Context: Data,~Element: Data,
  ~read: State -> Context -> Nat -> State & Element,~stride: Nat,
  state: State,context: Context,index: Nat
) -> State & Twin<Element>:
  +context = context
  +index = index
  read_at(~State,~Context,~Element,~Nat,~read,state,context,index,Nat.add(index,stride))
