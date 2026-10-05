# These lemmas cover the dense constructor: a ready call is exactly the
# checked core operation, so the core theorems in tensor_operations_refinement
# describe its storage. Broadcasting names the stretched views that reach the
# core operation; tensor_broadcast gives their extents. A failed operand
# passes its problem on and its storage through unchanged. Results keep the operands' largest worker count. Band
# representation and schedule refinements require separate connecting lemmas.
import Base
import ../storage_buffer.bend as Storage
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ../tensor_operations.bend as Operations
import ../tensor.bend as Tensors
import ../tensor_layout.bend as Layout
import ./storage_allocation_refinement.bend as AllocationLaws

# Elementwise combination.

# Equal extents take the direct path.
def combine_ready(~Element: Data,~operation: Element -> Element -> Element,+operations: F32,+name: String,
  left: Storage.Buffer<Element>,+left_shape: List<&2,U32>,+fill: Element,+left_view: Views.View,+left_workers: U32,
  right: Storage.Buffer<Element>,+right_shape: List<&2,U32>,+right_fill: Element,+right_view: Views.View,+right_workers: U32,
  same: {Shapes.same_extents(left_shape,right_shape) == True{} : Bool},
  left_views: {Tensors.view_of(left_shape) == Some{left_view} : Maybe<&2,Views.View>},
  right_views: {Tensors.view_of(right_shape) == Some{right_view} : Maybe<&2,Views.View>}) ->
  {Tensors.combine(~Element,~operation,name,operations,Tensors.Tensor{left,left_shape,fill,Tensors.Ready{},left_workers},
      Tensors.Tensor{right,right_shape,right_fill,Tensors.Ready{},right_workers}) ==
    Tensors.combined(~Element,name,left_shape,right_shape,fill,U32.max(left_workers,right_workers),
      Operations.combine(~Element,~operation,left,left_view,right,right_view)) : Tensors.Tensor<Element>}:
  %Equal.sym(Bool,Shapes.same_extents(left_shape,right_shape),True{},same) :
    {Tensors.combine_shapes(~Element,~operation,name,_,left,left_shape,fill,right,right_shape,U32.max(left_workers,right_workers)) ==
    Tensors.combined(~Element,name,left_shape,right_shape,fill,U32.max(left_workers,right_workers),
      Operations.combine(~Element,~operation,left,left_view,right,right_view)) : Tensors.Tensor<Element>}
  %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of(left_shape),Some{left_view},left_views) :
    {Tensors.combine_views(~Element,~operation,name,_,Tensors.view_of(right_shape),left,left_shape,fill,right,right_shape,U32.max(left_workers,right_workers)) ==
    Tensors.combined(~Element,name,left_shape,right_shape,fill,U32.max(left_workers,right_workers),
      Operations.combine(~Element,~operation,left,left_view,right,right_view)) : Tensors.Tensor<Element>}
  %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of(right_shape),Some{right_view},right_views) :
    {Tensors.combine_views(~Element,~operation,name,Some{left_view},_,left,left_shape,fill,right,right_shape,U32.max(left_workers,right_workers)) ==
    Tensors.combined(~Element,name,left_shape,right_shape,fill,U32.max(left_workers,right_workers),
      Operations.combine(~Element,~operation,left,left_view,right,right_view)) : Tensors.Tensor<Element>}
  {==}

# Different extents whose common extents are the left extents: the right
# operand is read through its stretched view and the left storage is reused.
def combine_right_stretched(~Element: Data,~operation: Element -> Element -> Element,+operations: F32,+name: String,
  left: Storage.Buffer<Element>,+left_shape: List<&2,U32>,+fill: Element,+left_view: Views.View,+left_workers: U32,
  right: Storage.Buffer<Element>,+right_shape: List<&2,U32>,+right_fill: Element,+right_view: Views.View,+right_workers: U32,
  +shape: List<&2,U32>,different: {Shapes.same_extents(left_shape,right_shape) == False{} : Bool},
  common: {Shapes.broadcast(left_shape,right_shape) == Some{shape} : Maybe<&2,List<&2,U32>>},
  in_place: {Shapes.same_extents(shape,left_shape) == True{} : Bool},
  left_views: {Tensors.view_of(left_shape) == Some{left_view} : Maybe<&2,Views.View>},
  right_views: {Tensors.stretched(Tensors.view_of(right_shape),left_shape) == Some{right_view} : Maybe<&2,Views.View>}) ->
  {Tensors.combine(~Element,~operation,name,operations,Tensors.Tensor{left,left_shape,fill,Tensors.Ready{},left_workers},
      Tensors.Tensor{right,right_shape,right_fill,Tensors.Ready{},right_workers}) ==
    Tensors.combined(~Element,name,left_shape,right_shape,fill,U32.max(left_workers,right_workers),
      Operations.combine(~Element,~operation,left,left_view,right,right_view)) : Tensors.Tensor<Element>}:
  %Equal.sym(Bool,Shapes.same_extents(left_shape,right_shape),False{},different) :
    {Tensors.combine_shapes(~Element,~operation,name,_,left,left_shape,fill,right,right_shape,U32.max(left_workers,right_workers)) ==
    Tensors.combined(~Element,name,left_shape,right_shape,fill,U32.max(left_workers,right_workers),
      Operations.combine(~Element,~operation,left,left_view,right,right_view)) : Tensors.Tensor<Element>}
  %Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.broadcast(left_shape,right_shape),Some{shape},common) :
    {Tensors.combine_broadcast(~Element,~operation,name,_,left,left_shape,fill,right,right_shape,U32.max(left_workers,right_workers)) ==
    Tensors.combined(~Element,name,left_shape,right_shape,fill,U32.max(left_workers,right_workers),
      Operations.combine(~Element,~operation,left,left_view,right,right_view)) : Tensors.Tensor<Element>}
  %Equal.sym(Bool,Shapes.same_extents(shape,left_shape),True{},in_place) :
    {Tensors.combine_stretched(~Element,~operation,name,_,left,left_shape,fill,right,right_shape,shape,U32.max(left_workers,right_workers)) ==
    Tensors.combined(~Element,name,left_shape,right_shape,fill,U32.max(left_workers,right_workers),
      Operations.combine(~Element,~operation,left,left_view,right,right_view)) : Tensors.Tensor<Element>}
  %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of(left_shape),Some{left_view},left_views) :
    {Tensors.combine_views(~Element,~operation,name,_,Tensors.stretched(Tensors.view_of(right_shape),left_shape),left,left_shape,fill,right,right_shape,U32.max(left_workers,right_workers)) ==
    Tensors.combined(~Element,name,left_shape,right_shape,fill,U32.max(left_workers,right_workers),
      Operations.combine(~Element,~operation,left,left_view,right,right_view)) : Tensors.Tensor<Element>}
  %Equal.sym(Maybe<&2,Views.View>,Tensors.stretched(Tensors.view_of(right_shape),left_shape),Some{right_view},right_views) :
    {Tensors.combine_views(~Element,~operation,name,Some{left_view},_,left,left_shape,fill,right,right_shape,U32.max(left_workers,right_workers)) ==
    Tensors.combined(~Element,name,left_shape,right_shape,fill,U32.max(left_workers,right_workers),
      Operations.combine(~Element,~operation,left,left_view,right,right_view)) : Tensors.Tensor<Element>}
  {==}

# Any other compatible extents: the left operand is copied through its
# stretched view into new storage of the common extents, and combined_fresh
# then combines the stretched right operand into that storage.
def combine_both_stretched(~Element: Data,~operation: Element -> Element -> Element,+operations: F32,+name: String,
  left: Storage.Buffer<Element>,+left_shape: List<&2,U32>,+fill: Element,+left_view: Views.View,+left_workers: U32,
  right: Storage.Buffer<Element>,+right_shape: List<&2,U32>,+right_fill: Element,+right_view: Views.View,+right_workers: U32,
  +shape: List<&2,U32>,+target: Views.View,+count: U32,different: {Shapes.same_extents(left_shape,right_shape) == False{} : Bool},
  common: {Shapes.broadcast(left_shape,right_shape) == Some{shape} : Maybe<&2,List<&2,U32>>},
  fresh: {Shapes.same_extents(shape,left_shape) == False{} : Bool},
  left_views: {Tensors.stretched(Tensors.view_of(left_shape),shape) == Some{left_view} : Maybe<&2,Views.View>},
  right_views: {Tensors.stretched(Tensors.view_of(right_shape),shape) == Some{right_view} : Maybe<&2,Views.View>},
  targets: {Tensors.view_of(shape) == Some{target} : Maybe<&2,Views.View>},counts: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>}) ->
  {Tensors.combine(~Element,~operation,name,operations,Tensors.Tensor{left,left_shape,fill,Tensors.Ready{},left_workers},
      Tensors.Tensor{right,right_shape,right_fill,Tensors.Ready{},right_workers}) ==
    Tensors.combined_fresh(~Element,~operation,name,shape,fill,U32.max(left_workers,right_workers),right,right_view,target,
      Operations.copy(~Element,left,left_view,Tensors.buffer(~Element,fill,count),target)) : Tensors.Tensor<Element>}:
  %Equal.sym(Bool,Shapes.same_extents(left_shape,right_shape),False{},different) :
    {Tensors.combine_shapes(~Element,~operation,name,_,left,left_shape,fill,right,right_shape,U32.max(left_workers,right_workers)) ==
    Tensors.combined_fresh(~Element,~operation,name,shape,fill,U32.max(left_workers,right_workers),right,right_view,target,
      Operations.copy(~Element,left,left_view,Tensors.buffer(~Element,fill,count),target)) : Tensors.Tensor<Element>}
  %Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.broadcast(left_shape,right_shape),Some{shape},common) :
    {Tensors.combine_broadcast(~Element,~operation,name,_,left,left_shape,fill,right,right_shape,U32.max(left_workers,right_workers)) ==
    Tensors.combined_fresh(~Element,~operation,name,shape,fill,U32.max(left_workers,right_workers),right,right_view,target,
      Operations.copy(~Element,left,left_view,Tensors.buffer(~Element,fill,count),target)) : Tensors.Tensor<Element>}
  %Equal.sym(Bool,Shapes.same_extents(shape,left_shape),False{},fresh) :
    {Tensors.combine_stretched(~Element,~operation,name,_,left,left_shape,fill,right,right_shape,shape,U32.max(left_workers,right_workers)) ==
    Tensors.combined_fresh(~Element,~operation,name,shape,fill,U32.max(left_workers,right_workers),right,right_view,target,
      Operations.copy(~Element,left,left_view,Tensors.buffer(~Element,fill,count),target)) : Tensors.Tensor<Element>}
  %Equal.sym(Maybe<&2,Views.View>,Tensors.stretched(Tensors.view_of(left_shape),shape),Some{left_view},left_views) :
    {Tensors.combine_fresh_views(~Element,~operation,name,_,Tensors.stretched(Tensors.view_of(right_shape),shape),Tensors.view_of(shape),Tensors.view_of(shape),Tensors.count_of(shape),left,right,shape,fill,U32.max(left_workers,right_workers)) ==
    Tensors.combined_fresh(~Element,~operation,name,shape,fill,U32.max(left_workers,right_workers),right,right_view,target,
      Operations.copy(~Element,left,left_view,Tensors.buffer(~Element,fill,count),target)) : Tensors.Tensor<Element>}
  %Equal.sym(Maybe<&2,Views.View>,Tensors.stretched(Tensors.view_of(right_shape),shape),Some{right_view},right_views) :
    {Tensors.combine_fresh_views(~Element,~operation,name,Some{left_view},_,Tensors.view_of(shape),Tensors.view_of(shape),Tensors.count_of(shape),left,right,shape,fill,U32.max(left_workers,right_workers)) ==
    Tensors.combined_fresh(~Element,~operation,name,shape,fill,U32.max(left_workers,right_workers),right,right_view,target,
      Operations.copy(~Element,left,left_view,Tensors.buffer(~Element,fill,count),target)) : Tensors.Tensor<Element>}
  %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of(shape),Some{target},targets) :
    {Tensors.combine_fresh_views(~Element,~operation,name,Some{left_view},Some{right_view},_,_,Tensors.count_of(shape),left,right,shape,fill,U32.max(left_workers,right_workers)) ==
    Tensors.combined_fresh(~Element,~operation,name,shape,fill,U32.max(left_workers,right_workers),right,right_view,target,
      Operations.copy(~Element,left,left_view,Tensors.buffer(~Element,fill,count),target)) : Tensors.Tensor<Element>}
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(shape),Some{count},counts) :
    {Tensors.combine_fresh_views(~Element,~operation,name,Some{left_view},Some{right_view},Some{target},Some{target},_,left,right,shape,fill,U32.max(left_workers,right_workers)) ==
    Tensors.combined_fresh(~Element,~operation,name,shape,fill,U32.max(left_workers,right_workers),right,right_view,target,
      Operations.copy(~Element,left,left_view,Tensors.buffer(~Element,fill,count),target)) : Tensors.Tensor<Element>}
  {==}

# Incompatible extents fail and leave the left storage as it was.
def combine_incompatible(~Element: Data,~operation: Element -> Element -> Element,+operations: F32,+name: String,
  left: Storage.Buffer<Element>,+left_shape: List<&2,U32>,+fill: Element,+left_workers: U32,
  right: Storage.Buffer<Element>,+right_shape: List<&2,U32>,+right_fill: Element,+right_workers: U32,
  different: {Shapes.same_extents(left_shape,right_shape) == False{} : Bool},
  incompatible: {Shapes.broadcast(left_shape,right_shape) == None{} : Maybe<&2,List<&2,U32>>}) ->
  {Tensors.combine(~Element,~operation,name,operations,Tensors.Tensor{left,left_shape,fill,Tensors.Ready{},left_workers},
      Tensors.Tensor{right,right_shape,right_fill,Tensors.Ready{},right_workers}) ==
    Tensors.Tensor{left,left_shape,fill,Tensors.Failed{Tensors.Problem{name,Tensors.ShapeMismatch{left_shape,right_shape}}},U32.max(left_workers,right_workers)} : Tensors.Tensor<Element>}:
  %Equal.sym(Bool,Shapes.same_extents(left_shape,right_shape),False{},different) :
    {Tensors.combine_shapes(~Element,~operation,name,_,left,left_shape,fill,right,right_shape,U32.max(left_workers,right_workers)) ==
    Tensors.Tensor{left,left_shape,fill,Tensors.Failed{Tensors.Problem{name,Tensors.ShapeMismatch{left_shape,right_shape}}},U32.max(left_workers,right_workers)} : Tensors.Tensor<Element>}
  %Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.broadcast(left_shape,right_shape),None{},incompatible) :
    {Tensors.combine_broadcast(~Element,~operation,name,_,left,left_shape,fill,right,right_shape,U32.max(left_workers,right_workers)) ==
    Tensors.Tensor{left,left_shape,fill,Tensors.Failed{Tensors.Problem{name,Tensors.ShapeMismatch{left_shape,right_shape}}},U32.max(left_workers,right_workers)} : Tensors.Tensor<Element>}
  {==}

def combine_left_failed(~Element: Data,~operation: Element -> Element -> Element,+operations: F32,+name: String,
  left: Storage.Buffer<Element>,+left_shape: List<&2,U32>,+fill: Element,+problem: Tensors.Problem,+left_workers: U32,
  right: Storage.Buffer<Element>,+right_shape: List<&2,U32>,+right_fill: Element,+right_status: Tensors.Status,+right_workers: U32) ->
  {Tensors.combine(~Element,~operation,name,operations,Tensors.Tensor{left,left_shape,fill,Tensors.Failed{problem},left_workers},
      Tensors.Tensor{right,right_shape,right_fill,right_status,right_workers}) ==
    Tensors.Tensor{left,left_shape,fill,Tensors.Failed{problem},U32.max(left_workers,right_workers)} : Tensors.Tensor<Element>}:
  {==}

def combine_right_failed(~Element: Data,~operation: Element -> Element -> Element,+operations: F32,+name: String,
  left: Storage.Buffer<Element>,+left_shape: List<&2,U32>,+fill: Element,+left_workers: U32,
  right: Storage.Buffer<Element>,+right_shape: List<&2,U32>,+right_fill: Element,+problem: Tensors.Problem,+right_workers: U32) ->
  {Tensors.combine(~Element,~operation,name,operations,Tensors.Tensor{left,left_shape,fill,Tensors.Ready{},left_workers},
      Tensors.Tensor{right,right_shape,right_fill,Tensors.Failed{problem},right_workers}) ==
    Tensors.Tensor{left,left_shape,fill,Tensors.Failed{problem},U32.max(left_workers,right_workers)} : Tensors.Tensor<Element>}:
  {==}

# Unary application, indexed application and reduction.

def apply_ready(~Element: Data,~operation: Element -> Element,+name: String,+operations: F32,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+view: Views.View,
  +workers: U32,views: {Tensors.view_of(shape) == Some{view} : Maybe<&2,Views.View>}) ->
  {Tensors.apply(~Element,~operation,name,operations,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers}) ==
    Tensors.applied(~Element,name,shape,fill,workers,Operations.apply(~Element,~operation,storage,view)) : Tensors.Tensor<Element>}:
  %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of(shape),Some{view},views) : {Tensors.apply_view(~Element,~operation,name,_,storage,shape,fill,workers) ==
    Tensors.applied(~Element,name,shape,fill,workers,Operations.apply(~Element,~operation,storage,view)) : Tensors.Tensor<Element>}
  {==}

def apply_failed(~Element: Data,~operation: Element -> Element,+name: String,+operations: F32,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+problem: Tensors.Problem,
  +workers: U32) ->
  {Tensors.apply(~Element,~operation,name,operations,Tensors.Tensor{storage,shape,fill,Tensors.Failed{problem},workers}) ==
    Tensors.Tensor{storage,shape,fill,Tensors.Failed{problem},workers} : Tensors.Tensor<Element>}:
  {==}

def apply_indexed_ready(~Element: Data,~operation: U32 -> Element -> Element,+operations: F32,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,
  +count: U32,+view: Views.View,+workers: U32,
  counts: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},views: {Tensors.view_of(shape) == Some{view} : Maybe<&2,Views.View>}) ->
  {Tensors.apply_indexed(~Element,~operation,operations,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers}) ==
    Tensors.indexed(~Element,shape,fill,workers,Operations.apply_indexed(~Element,~operation,storage,view,Tensors.buffer(~Element,fill,count))) : Tensors.Tensor<Element>}:
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(shape),Some{count},counts) : {Tensors.apply_indexed_count(~Element,~operation,_,Tensors.view_of(shape),storage,shape,fill,workers) ==
    Tensors.indexed(~Element,shape,fill,workers,Operations.apply_indexed(~Element,~operation,storage,view,Tensors.buffer(~Element,fill,count))) : Tensors.Tensor<Element>}
  %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of(shape),Some{view},views) : {Tensors.apply_indexed_count(~Element,~operation,Some{count},_,storage,shape,fill,workers) ==
    Tensors.indexed(~Element,shape,fill,workers,Operations.apply_indexed(~Element,~operation,storage,view,Tensors.buffer(~Element,fill,count))) : Tensors.Tensor<Element>}
  {==}

def reduce_ready(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,+name: String,storage: Storage.Buffer<Element>,
  +shape: List<&2,U32>,+fill: Element,+view: Views.View,initial: Accumulator,+workers: U32,
  views: {Tensors.view_of(shape) == Some{view} : Maybe<&2,Views.View>}) ->
  {Tensors.reduce(~Element,~Accumulator,~operation,name,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers},initial) ==
    Tensors.reduced(~Element,~Accumulator,name,shape,Operations.reduce(~Element,~Accumulator,~operation,storage,view,initial)) :
    Result<&2,&2,Tensors.Problem,Accumulator>}:
  %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of(shape),Some{view},views) : {Tensors.reduce_view(~Element,~Accumulator,~operation,name,_,storage,shape,initial) ==
    Tensors.reduced(~Element,~Accumulator,name,shape,Operations.reduce(~Element,~Accumulator,~operation,storage,view,initial)) :
    Result<&2,&2,Tensors.Problem,Accumulator>}
  {==}

def reduce_failed(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,+name: String,storage: Storage.Buffer<Element>,
  +shape: List<&2,U32>,+fill: Element,+problem: Tensors.Problem,initial: Accumulator,+workers: U32) ->
  {Tensors.reduce(~Element,~Accumulator,~operation,name,Tensors.Tensor{storage,shape,fill,Tensors.Failed{problem},workers},initial) ==
    Fail{problem} : Result<&2,&2,Tensors.Problem,Accumulator>}:
  {==}

# check is the only operation that observes and clears a failure.

def check_ready(~Element: Data,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  {Tensors.check(~Element,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers}) ==
    Done{Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers}} : Result<&1,&1,Tensors.Tensor<Element> & Tensors.Problem,Tensors.Tensor<Element>>}:
  {==}

def check_failed(~Element: Data,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+problem: Tensors.Problem,+workers: U32) ->
  {Tensors.check(~Element,Tensors.Tensor{storage,shape,fill,Tensors.Failed{problem},workers}) ==
    Fail{(Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers},problem)} : Result<&1,&1,Tensors.Tensor<Element> & Tensors.Problem,Tensors.Tensor<Element>>}:
  {==}

# Layout results are one checked copy from a source view into a new dense tensor.

def gathered(~Element: Data,+name: String,+source_view: Views.View,+view_shape: List<&2,U32>,+target_shape: List<&2,U32>,
  input: Storage.Buffer<Element>,+source_shape: List<&2,U32>,+fill: Element,+target_view: Views.View,+count: U32,+workers: U32,
  targets: {Tensors.view_of(view_shape) == Some{target_view} : Maybe<&2,Views.View>},counts: {Tensors.count_of(view_shape) == Some{count} : Maybe<&2,U32>}) ->
  {Layout.gather_status(~Element,name,Tensors.Ready{},Some{source_view},view_shape,target_shape,input,source_shape,fill,workers) ==
    Layout.copied(~Element,name,source_shape,target_shape,fill,workers,Operations.copy(~Element,input,source_view,Tensors.buffer(~Element,fill,count),target_view)) :
    Tensors.Tensor<Element> & Tensors.Tensor<Element>}:
  %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of(view_shape),Some{target_view},targets) :
    {Layout.gather_views(~Element,name,Some{source_view},_,Tensors.count_of(view_shape),input,source_shape,target_shape,fill,workers) ==
    Layout.copied(~Element,name,source_shape,target_shape,fill,workers,Operations.copy(~Element,input,source_view,Tensors.buffer(~Element,fill,count),target_view)) :
    Tensors.Tensor<Element> & Tensors.Tensor<Element>}
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(view_shape),Some{count},counts) :
    {Layout.gather_views(~Element,name,Some{source_view},Some{target_view},_,input,source_shape,target_shape,fill,workers) ==
    Layout.copied(~Element,name,source_shape,target_shape,fill,workers,Operations.copy(~Element,input,source_view,Tensors.buffer(~Element,fill,count),target_view)) :
    Tensors.Tensor<Element> & Tensors.Tensor<Element>}
  {==}

# Broadcasting to given extents is one checked copy from the stretched view.
def broadcast_to_ready(~Element: Data,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,+target: List<&2,U32>,
  +source_view: Views.View,+target_view: Views.View,+count: U32,
  sources: {Tensors.stretched(Tensors.view_of(shape),target) == Some{source_view} : Maybe<&2,Views.View>},
  targets: {Tensors.view_of(target) == Some{target_view} : Maybe<&2,Views.View>},counts: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>}) ->
  {Layout.broadcast_to(~Element,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers},target) ==
    Layout.result_of(~Element,Layout.copied(~Element,"broadcast_to",shape,target,fill,workers,
      Operations.copy(~Element,storage,source_view,Tensors.buffer(~Element,fill,count),target_view))) : Tensors.Tensor<Element>}:
  %Equal.sym(Maybe<&2,Views.View>,Tensors.stretched(Tensors.view_of(shape),target),Some{source_view},sources) :
    {Layout.broadcast_status(~Element,Tensors.Ready{},_,storage,shape,target,fill,workers) ==
    Layout.result_of(~Element,Layout.copied(~Element,"broadcast_to",shape,target,fill,workers,
      Operations.copy(~Element,storage,source_view,Tensors.buffer(~Element,fill,count),target_view))) : Tensors.Tensor<Element>}
  Equal.cong(Tensors.Tensor<Element> & Tensors.Tensor<Element>,Tensors.Tensor<Element>,pair => Layout.result_of(~Element,pair),
    Layout.gather_status(~Element,"broadcast_to",Tensors.Ready{},Some{source_view},target,target,storage,shape,fill,workers),
    Layout.copied(~Element,"broadcast_to",shape,target,fill,workers,Operations.copy(~Element,storage,source_view,Tensors.buffer(~Element,fill,count),target_view)),
    gathered(~Element,"broadcast_to",source_view,target,target,storage,shape,fill,target_view,count,workers,targets,counts))

# An axis reduction starts from storage filled with the initial value and
# folds the slices of the axis into it in increasing index.
def reduce_axis_ready(~Element: Data,~operation: Element -> Element -> Element,+name: String,+initial: Element,storage: Storage.Buffer<Element>,
  +shape: List<&2,U32>,+fill: Element,+workers: U32,+axis: U32,+count: U32,
  inside: {Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)) == True{} : Bool},
  counts: {Tensors.count_of(Layout.with_extent(shape,U32.to_nat(axis),1)) == Some{count} : Maybe<&2,U32>}) ->
  {Layout.reduce_axis(~Element,~operation,name,initial,Tensors.Tensor{storage,shape,fill,Tensors.Ready{},workers},axis) ==
    Layout.folded(~Element,name,shape,Shapes.without_axis(shape,U32.to_nat(axis)),fill,workers,
      Layout.fold_slices(~Element,~operation,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape,Layout.with_extent(shape,U32.to_nat(axis),1),
        ((Tensors.buffer(~Element,initial,count),storage),True{}))) : Tensors.Tensor<Element>}:
  %Equal.sym(Bool,Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)),True{},inside) :
    {Layout.reduce_axis_status(~Element,~operation,name,Tensors.Ready{},_,axis,storage,shape,initial,fill,workers) ==
    Layout.folded(~Element,name,shape,Shapes.without_axis(shape,U32.to_nat(axis)),fill,workers,
      Layout.fold_slices(~Element,~operation,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape,Layout.with_extent(shape,U32.to_nat(axis),1),
        ((Tensors.buffer(~Element,initial,count),storage),True{}))) : Tensors.Tensor<Element>}
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(Layout.with_extent(shape,U32.to_nat(axis),1)),Some{count},counts) :
    {Layout.reduce_axis_count(~Element,~operation,name,_,axis,storage,shape,Layout.with_extent(shape,U32.to_nat(axis),1),initial,fill,workers) ==
    Layout.folded(~Element,name,shape,Shapes.without_axis(shape,U32.to_nat(axis)),fill,workers,
      Layout.fold_slices(~Element,~operation,U32.to_nat(Layout.extent_at(shape,U32.to_nat(axis))),0,axis,shape,Layout.with_extent(shape,U32.to_nat(axis),1),
        ((Tensors.buffer(~Element,initial,count),storage),True{}))) : Tensors.Tensor<Element>}
  {==}

# One step of the fold is the checked in-place combination of the output with
# the slice of the axis at index. unit is the shape with extent 1 at the axis.
def fold_step_ready(~Element: Data,~operation: Element -> Element -> Element,+index: U32,+axis: U32,+shape: List<&2,U32>,+unit: List<&2,U32>,
  +slice: Views.View,+target: Views.View,output: Storage.Buffer<Element>,input: Storage.Buffer<Element>,+valid: Bool,
  slices: {Layout.sliced(Tensors.view_of(shape),axis,index,1,1) == Some{slice} : Maybe<&2,Views.View>},
  targets: {Tensors.view_of(unit) == Some{target} : Maybe<&2,Views.View>}) ->
  {Layout.fold_step(~Element,~operation,index,axis,shape,unit,((output,input),valid)) ==
    Layout.settle(~Element,valid,Operations.combine(~Element,~operation,output,target,input,slice)) : (Storage.Buffer<Element> & Storage.Buffer<Element>) & Bool}:
  %Equal.sym(Maybe<&2,Views.View>,Layout.sliced(Tensors.view_of(shape),axis,index,1,1),Some{slice},slices) :
    {Layout.fold_slice(~Element,~operation,valid,_,Tensors.view_of(unit),output,input) ==
    Layout.settle(~Element,valid,Operations.combine(~Element,~operation,output,target,input,slice)) : (Storage.Buffer<Element> & Storage.Buffer<Element>) & Bool}
  %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of(unit),Some{target},targets) :
    {Layout.fold_slice(~Element,~operation,valid,Some{slice},_,output,input) ==
    Layout.settle(~Element,valid,Operations.combine(~Element,~operation,output,target,input,slice)) : (Storage.Buffer<Element> & Storage.Buffer<Element>) & Bool}
  {==}

# A constant padding with padded extents is one write of the tensor through
# the interior view, unless nothing is added.
def pad_constant_selected(~Element: Data,+value: Element,tensor: Tensors.Tensor<Element>,+before: List<&2,U32>,+after: List<&2,U32>,+padded: List<&2,U32>) ->
  {Layout.pad_border(~Element,Layout.Constant{value},tensor,before,after,Some{padded}) ==
    Layout.pad_constant(~Element,Layout.all_zero(before) && Layout.all_zero(after),value,tensor,before,padded) : Tensors.Tensor<Element>}:
  {==}

def combined_fresh_accepted(~Element: Data,~operation: Element -> Element -> Element,+name: String,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  left: Storage.Buffer<Element>,output: Storage.Buffer<Element>,right: Storage.Buffer<Element>,+right_view: Views.View,+target: Views.View) ->
  {Tensors.combined_fresh(~Element,~operation,name,shape,fill,workers,right,right_view,target,Operations.Accepted{(left,output)}) ==
    Tensors.combined(~Element,name,shape,shape,fill,workers,Operations.combine(~Element,~operation,output,target,right,right_view)) : Tensors.Tensor<Element>}:
  {==}
