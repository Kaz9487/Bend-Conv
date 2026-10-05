# Guard facts for actual public Buffer calls. Balance is recovered from Buffer;
# the only logical premises are the successful branches tested by the API.
import Base
import ./bounded_u32_arithmetic.bend as Arithmetic
import ../storage_buffer.bend as Storage
import ../tensor_operations.bend as Operations
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ./storage_buffer_refinement.bend as BufferLaws
import ./storage_certificate.bend as Certificate
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_contiguous_bounds.bend as ContiguousBounds
import ./tensor_cursor_model.bend as CursorModel
import ./tensor_cursor_bounds.bend as CursorBounds
import ./traversal_write_bounds.bend as WriteBounds
import ./boolean_logic.bend as Boolean
import ./u32_full_range.bend as FullRange

def valid(~Element: Data,buffer: Storage.Buffer<Element>,view: Views.View) -> Bool:
  Storage.Buffer{storage,length,certificate} = buffer
  Operations.valid_buffer(length,Storage.capacity(~Element,storage,certificate),view)

def ReadSafety(view: Views.View,count: U32,length: U32) -> Data:
  Views.View{axes,offset} = view
  CursorBounds.ReadSchedule(U32.to_nat(count),CursorModel.prepare_axes(axes,[]),U32.to_nat(offset),length)

type SourceSafety<-Element: Data,-buffer: Storage.Buffer<Element>,-view: Views.View,-count: U32> is Data:
  SourceSafety{
    balanced: BufferLaws.Balanced<Element,buffer>,
    length_exact: {U32.to_nat(U32.from_nat(BufferLaws.length(~Element,buffer))) == BufferLaws.length(~Element,buffer) : Nat},
    reads: ReadSafety(view,count,U32.from_nat(BufferLaws.length(~Element,buffer)))
  }

def source(~Element: Data,buffer: Storage.Buffer<Element>,+view: Views.View,+count: U32,
  counted: {Views.numel(view) == Some{count} : Maybe<&2,U32>},+accepted: {valid(~Element,buffer,view) == True{} : Bool}) -> SourceSafety<Element,buffer,view,count>:
  Storage.Buffer{storage,+length,certificate} = buffer
  Certificate.Certificate{+certificate_depth,+certificate_equation} = certificate
  Views.View{+axes,+offset} = view
  +capacity = Storage.capacity(~Element,storage,Certificate.Certificate{certificate_depth,certificate_equation})
  SourceSafety{BufferLaws.balanced(~Element,Storage.Buffer{storage,length,Certificate.Certificate{certificate_depth,certificate_equation}}),
    FullRange.roundtrip(length,capacity,Boolean.left(Nat.is_le(length,U32.to_nat(capacity)),Views.fits(Views.View{axes,offset},U32.from_nat(length)),accepted)),
    CursorBounds.checked_reads(axes,offset,count,U32.from_nat(length),counted,
      Boolean.right(Nat.is_le(length,U32.to_nat(capacity)),Views.fits(Views.View{axes,offset},U32.from_nat(length)),accepted))}

type DestinationSafety<-Element: Data,-buffer: Storage.Buffer<Element>,-view: Views.View,-count: U32> is Data:
  DestinationSafety{
    balanced: BufferLaws.Balanced<Element,buffer>,
    interval: {Nat.is_le(Nat.add(U32.to_nat(ContiguousBounds.offset(view)),U32.to_nat(count)),U32.to_nat(U32.from_nat(BufferLaws.length(~Element,buffer)))) == True{} : Bool},
    writes: WriteBounds.Schedule(U32.to_nat(count),ContiguousBounds.offset(view),U32.from_nat(BufferLaws.length(~Element,buffer)))
  }

def destination(~Element: Data,buffer: Storage.Buffer<Element>,+view: Views.View,+count: U32,
  counted: {Views.numel(view) == Some{count} : Maybe<&2,U32>},contiguous: {Views.is_contiguous(view) == True{} : Bool},
  accepted: {valid(~Element,buffer,view) == True{} : Bool}) -> DestinationSafety<Element,buffer,view,count>:
  Storage.Buffer{storage,+length,certificate} = buffer
  Certificate.Certificate{+certificate_depth,+certificate_equation} = certificate
  +capacity = Storage.capacity(~Element,storage,Certificate.Certificate{certificate_depth,certificate_equation})
  +room = ContiguousBounds.checked_interval(view,count,U32.from_nat(length),contiguous,counted,
    Boolean.right(Nat.is_le(length,U32.to_nat(capacity)),Views.fits(view,U32.from_nat(length)),accepted))
  DestinationSafety{BufferLaws.balanced(~Element,Storage.Buffer{storage,length,Certificate.Certificate{certificate_depth,certificate_equation}}),room,
    WriteBounds.schedule(U32.to_nat(count),ContiguousBounds.offset(view),U32.from_nat(length),room)}

def same_count(+source: Views.View,+target: Views.View,same: {Shapes.same(Views.shape(source),Views.shape(target)) == True{} : Bool}) ->
  {Views.numel(source) == Views.numel(target) : Maybe<&2,U32>}:
  Views.View{+source_axes,source_offset} = source
  Views.View{+target_axes,target_offset} = target
  Equal.cong(List<&2,U32>,Maybe<&2,U32>,extents => Shapes.numel(Shapes.Shape{extents}),Views.axis_extents(source_axes),Views.axis_extents(target_axes),
    ShapeLaws.same_extents(Views.axis_extents(source_axes),Views.axis_extents(target_axes),same))

def copy_guard(~Element: Data,input: Storage.Buffer<Element>,+source: Views.View,output: Storage.Buffer<Element>,+target: Views.View) -> Bool:
  Shapes.same(Views.shape(source),Views.shape(target)) && Views.is_contiguous(target) && valid(~Element,input,source) && valid(~Element,output,target)

type CopySafety<-Element: Data,-input: Storage.Buffer<Element>,-source: Views.View,-output: Storage.Buffer<Element>,-target: Views.View,-count: U32> is Data:
  CopySafety{source: SourceSafety<Element,input,source,count>,destination: DestinationSafety<Element,output,target,count>}

def copy(~Element: Data,input: Storage.Buffer<Element>,+source_view: Views.View,output: Storage.Buffer<Element>,+target: Views.View,+count: U32,
  +counted: {Views.numel(target) == Some{count} : Maybe<&2,U32>},
  +accepted: {copy_guard(~Element,input,source_view,output,target) == True{} : Bool}) -> CopySafety<Element,input,source_view,output,target,count>:
  Storage.Buffer{input,+input_length,input_certificate} = input
  Certificate.Certificate{+input_certificate_depth,+input_certificate_equation} = input_certificate
  Storage.Buffer{output,+output_length,output_certificate} = output
  Certificate.Certificate{+output_certificate_depth,+output_certificate_equation} = output_certificate
  +conditions = {[Operations.valid_buffer(output_length,Storage.capacity(~Element,output,Certificate.Certificate{output_certificate_depth,output_certificate_equation}),target),
    Operations.valid_buffer(input_length,Storage.capacity(~Element,input,Certificate.Certificate{input_certificate_depth,input_certificate_equation}),source_view),Views.is_contiguous(target),Shapes.same(Views.shape(source_view),Views.shape(target))] : List<&2,Bool>}
  CopySafety{source(~Element,Storage.Buffer{input,input_length,Certificate.Certificate{input_certificate_depth,input_certificate_equation}},source_view,count,
    Equal.trans(Maybe<&2,U32>,Views.numel(source_view),Views.numel(target),Some{count},same_count(source_view,target,Boolean.every_condition_holds(conditions,3n,accepted)),counted),
    Boolean.every_condition_holds(conditions,1n,accepted)),
    destination(~Element,Storage.Buffer{output,output_length,Certificate.Certificate{output_certificate_depth,output_certificate_equation}},target,count,counted,Boolean.every_condition_holds(conditions,2n,accepted),Boolean.every_condition_holds(conditions,0n,accepted))}

def combine_guard(~Element: Data,left: Storage.Buffer<Element>,+left_view: Views.View,right: Storage.Buffer<Element>,+right_view: Views.View) -> Bool:
  Shapes.same(Views.shape(left_view),Views.shape(right_view)) && Views.is_contiguous(left_view) && valid(~Element,left,left_view) && valid(~Element,right,right_view)

type CombineSafety<-Element: Data,-left: Storage.Buffer<Element>,-left_view: Views.View,-right: Storage.Buffer<Element>,-right_view: Views.View,-count: U32> is Data:
  CombineSafety{left: DestinationSafety<Element,left,left_view,count>,right: SourceSafety<Element,right,right_view,count>}

def combine(~Element: Data,left: Storage.Buffer<Element>,+left_view: Views.View,right: Storage.Buffer<Element>,+right_view: Views.View,+count: U32,
  +counted: {Views.numel(left_view) == Some{count} : Maybe<&2,U32>},
  +accepted: {combine_guard(~Element,left,left_view,right,right_view) == True{} : Bool}) -> CombineSafety<Element,left,left_view,right,right_view,count>:
  Storage.Buffer{left,+left_length,left_certificate} = left
  Certificate.Certificate{+left_certificate_depth,+left_certificate_equation} = left_certificate
  Storage.Buffer{right,+right_length,right_certificate} = right
  Certificate.Certificate{+right_certificate_depth,+right_certificate_equation} = right_certificate
  +conditions = {[Operations.valid_buffer(right_length,Storage.capacity(~Element,right,Certificate.Certificate{right_certificate_depth,right_certificate_equation}),right_view),
    Operations.valid_buffer(left_length,Storage.capacity(~Element,left,Certificate.Certificate{left_certificate_depth,left_certificate_equation}),left_view),Views.is_contiguous(left_view),Shapes.same(Views.shape(left_view),Views.shape(right_view))] : List<&2,Bool>}
  CombineSafety{destination(~Element,Storage.Buffer{left,left_length,Certificate.Certificate{left_certificate_depth,left_certificate_equation}},left_view,count,counted,Boolean.every_condition_holds(conditions,2n,accepted),Boolean.every_condition_holds(conditions,1n,accepted)),
    source(~Element,Storage.Buffer{right,right_length,Certificate.Certificate{right_certificate_depth,right_certificate_equation}},right_view,count,
      Equal.trans(Maybe<&2,U32>,Views.numel(right_view),Views.numel(left_view),Some{count},
        Equal.sym(Maybe<&2,U32>,Views.numel(left_view),Views.numel(right_view),same_count(left_view,right_view,Boolean.every_condition_holds(conditions,3n,accepted))),counted),
      Boolean.every_condition_holds(conditions,0n,accepted))}

def apply_guard(~Element: Data,buffer: Storage.Buffer<Element>,+view: Views.View) -> Bool:
  Views.is_contiguous(view) && valid(~Element,buffer,view)

def apply(~Element: Data,buffer: Storage.Buffer<Element>,+view: Views.View,+count: U32,
  counted: {Views.numel(view) == Some{count} : Maybe<&2,U32>},+accepted: {apply_guard(~Element,buffer,view) == True{} : Bool}) -> DestinationSafety<Element,buffer,view,count>:
  Storage.Buffer{storage,+length,certificate} = buffer
  Certificate.Certificate{+certificate_depth,+certificate_equation} = certificate
  +buffer_valid = Operations.valid_buffer(length,Storage.capacity(~Element,storage,Certificate.Certificate{certificate_depth,certificate_equation}),view)
  destination(~Element,Storage.Buffer{storage,length,Certificate.Certificate{certificate_depth,certificate_equation}},view,count,counted,
    Boolean.left(Views.is_contiguous(view),buffer_valid,accepted),Boolean.right(Views.is_contiguous(view),buffer_valid,accepted))

def reduce(~Element: Data,buffer: Storage.Buffer<Element>,+view: Views.View,+count: U32,
  counted: {Views.numel(view) == Some{count} : Maybe<&2,U32>},accepted: {valid(~Element,buffer,view) == True{} : Bool}) -> SourceSafety<Element,buffer,view,count>:
  source(~Element,buffer,view,count,counted,accepted)

# Non-overlap is derived from the interval produced by the actual API guard.
# There is no extra balance or disjointness premise for the caller to supply.
def destination_distinct(~Element: Data,buffer: Storage.Buffer<Element>,+view: Views.View,+count: U32,
  safety: DestinationSafety<Element,buffer,view,count>,+left: Nat,+right: Nat,
  ordered: {Nat.is_lt(left,right) == True{} : Bool},inside: {Nat.is_lt(right,U32.to_nat(count)) == True{} : Bool}) ->
  {U32.is_eq(WriteBounds.position_after(left,ContiguousBounds.offset(view)),WriteBounds.position_after(right,ContiguousBounds.offset(view))) == False{} : Bool}:
  DestinationSafety{balanced,interval,writes} = safety
  %Arithmetic.native_roundtrip(ContiguousBounds.offset(view)) :
    {U32.is_eq(WriteBounds.position_after(left,_),WriteBounds.position_after(right,_)) == False{} : Bool}
  WriteBounds.distinct_positions(left,right,U32.to_nat(count),U32.to_nat(ContiguousBounds.offset(view)),U32.from_nat(BufferLaws.length(~Element,buffer)),ordered,inside,interval)
