# Rank-independent address validation. This theorem guarantees bounds of an
# accepted address; it does not substitute for non-wrapping layout arithmetic.
import Base
import ../tensor_view.bend as Views
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws

def accepted_address_is_inside(result: Maybe<&2,U32>,length: U32) -> Bool:
  match result:
    case None{}: True{}
    case Some{address}: U32.is_lt(address,length)

def finish_is_inside(+valid: Bool,+address: U32,+length: U32,
  checked: {U32.is_lt(address,length) == valid : Bool}) ->
  {accepted_address_is_inside(Views.finish_address(valid,address),length) == True{} : Bool}:
  match valid:
    case False{}: {==}
    case True{}: checked

def locate_is_inside(+axes: List<&2,Views.Axis>,+coordinates: List<&2,U32>,+address: Maybe<&2,U32>,+length: U32) ->
  {accepted_address_is_inside(Views.locate(axes,coordinates,address,length),length) == True{} : Bool}:
  match axes coordinates address:
    case Nil{} Nil{} Some{+offset}:
      finish_is_inside(U32.is_lt(offset,length),offset,length,{==})
    case Con{Views.Axis{+extent,+stride},+tail} Con{+coordinate,+rest} Some{+offset}:
      locate_is_inside(tail,rest,Views.advance_inside(U32.is_lt(coordinate,extent) && U32.is_lt(offset,length),offset,coordinate,stride,length),length)
    case Nil{} Nil{} None{}: {==}
    case Nil{} Con{coordinate,rest} None{}: {==}
    case Nil{} Con{coordinate,rest} Some{offset}: {==}
    case Con{Views.Axis{extent,stride},tail} Nil{} None{}: {==}
    case Con{Views.Axis{extent,stride},tail} Nil{} Some{offset}: {==}
    case Con{Views.Axis{extent,stride},tail} Con{coordinate,rest} None{}: {==}

def view_address_is_inside(+view: Views.View,+coordinates: List<&2,U32>,+length: U32) ->
  {accepted_address_is_inside(Views.address(view,coordinates,length),length) == True{} : Bool}:
  Views.View{axes,offset} = view
  locate_is_inside(axes,coordinates,Some{offset},length)
