# Every actual replan choice preserves the allocation and coverage contract.
# Cost comparisons choose between valid plans; their numerical value is free.
import Base
import ../tensor_band.bend as Bands
import ../tensor_band_window.bend as BandWindow
import ../kernel_band_planning.bend as BandPlanning
import ../kernel_planning.bend as KernelPlanning
import ../kernel_cost.bend as Cost
import ../machine_limits.bend as Machine
import ./tensor_band_dense.bend as BandDense
import ./tensor_band_plan.bend as BandPlan
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_inspection.bend as Inspection
import ./storage_shape.bend as Shape
import ./nat_order.bend as Order
import ./row_major_index.bend as RowMajor
import ./machine_limit_bounds.bend as Limits

def layout(selection: BandPlanning.LayoutPlan) -> Bands.Plan:
  BandPlanning.LayoutPlan{layout,parallel,time,copies} = selection
  layout

def LayoutFacts(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+plan: Bands.Plan) -> Data:
  Shape.Both<BandDense.Fits(geometry,plan,capacity,Machine.half_word(limits)),BandPlan.Coverage(plan,0n,Inspection.total_rows(geometry))>

def SelectedFacts(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,selection: BandPlanning.LayoutPlan) -> Data:
  LayoutFacts(geometry,limits,capacity,layout(selection))

def transported(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+plan: Bands.Plan,+selection: BandPlanning.LayoutPlan,
  exact: {layout(selection) == plan : Bands.Plan},facts: LayoutFacts(geometry,limits,capacity,plan)) -> SelectedFacts(geometry,limits,capacity,selection):
  %Equal.sym(Bands.Plan,layout(selection),plan,exact) : LayoutFacts(geometry,limits,capacity,_)
  facts

def image_plan(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+depth: Nat,+domain: BandGather.Domain<geometry,limits,capacity>) ->
  LayoutFacts(geometry,limits,capacity,BandDense.split_plan(geometry,depth)):
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandGather.Domain{capacity_ok,cp,+hp,wp,room,allocation_room} = domain
  +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  +row_room = BandGather.rows_room(geometry,limits,capacity,domain)
  +images_room = Order.transitive(U32.to_nat(images),Nat.mul(U32.to_nat(images),U32.to_nat(height)),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(U32.to_nat(images),U32.to_nat(height),hp),row_room)
  Shape.Both{BandDense.image_plan(geometry,limits,depth,capacity,domain),BandPlan.image_coverage(images,height,depth,Machine.capacity(limits),Limits.capacity_fits(limits),images_room,row_room)}

def cost_layout(+plan: Bands.Plan,+parallel: Bool,+workers: U32,+copies: F32,+wake: F32,+reorder: F32,cost: BandPlanning.Movement) ->
  {layout(BandPlanning.layout_cost(plan,parallel,workers,copies,wake,reorder,cost)) == plan : Bands.Plan}:
  BandPlanning.Movement{work,span,open,settled,preparation,minimum,leaves} = cost
  {==}

def evaluated_layout(+plan: Bands.Plan,+previous: Bands.Plan,+work: KernelPlanning.Work,+window: BandWindow.Geometry,+workers: U32,config: Cost.Configuration,+cost: BandPlanning.Movement) ->
  {layout(BandPlanning.evaluated(plan,previous,work,window,workers,config,cost)) == plan : Bands.Plan}:
  Cost.Configuration{+mac,+byte_ns,+operation_ns,+grain,+wake} = config
  +config = {Cost.Configuration{mac,byte_ns,operation_ns,grain,wake} : Cost.Configuration}
  cost_layout(plan,BandPlanning.retained_parallel(window,workers,config,cost),workers,
    Bool.pick(F32,Bands.same_plan(plan,previous),0.0,BandPlanning.transfer_layout(plan,BandWindow.input_geometry(window),byte_ns)),BandPlanning.wake_cost(window,config),BandPlanning.reorder_cost(work,config),cost)

def evaluated(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+plan: Bands.Plan,+previous: Bands.Plan,+work: KernelPlanning.Work,+window: BandWindow.Geometry,+workers: U32,+config: Cost.Configuration,+cost: BandPlanning.Movement,
  facts: LayoutFacts(geometry,limits,capacity,plan)) -> SelectedFacts(geometry,limits,capacity,BandPlanning.evaluated(plan,previous,work,window,workers,config,cost)):
  transported(geometry,limits,capacity,plan,BandPlanning.evaluated(plan,previous,work,window,workers,config,cost),evaluated_layout(plan,previous,work,window,workers,config,cost),facts)

def picked(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,choose: Bool,+candidate: BandPlanning.LayoutPlan,+best: BandPlanning.LayoutPlan,
  candidate_facts: SelectedFacts(geometry,limits,capacity,candidate),best_facts: SelectedFacts(geometry,limits,capacity,best)) ->
  SelectedFacts(geometry,limits,capacity,Bool.pick(BandPlanning.LayoutPlan,choose,candidate,best)):
  match choose:
    case True{}: candidate_facts
    case False{}: best_facts

def prefer(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+candidate: BandPlanning.LayoutPlan,+best: BandPlanning.LayoutPlan,
  candidate_facts: SelectedFacts(geometry,limits,capacity,candidate),best_facts: SelectedFacts(geometry,limits,capacity,best)) ->
  SelectedFacts(geometry,limits,capacity,BandPlanning.prefer(candidate,best)):
  BandPlanning.LayoutPlan{+plan,+parallel,+time,+copies} = candidate
  BandPlanning.LayoutPlan{+old,+old_parallel,+old_time,+old_copies} = best
  picked(geometry,limits,capacity,F32.is_lt(time,old_time),BandPlanning.LayoutPlan{plan,parallel,time,copies},BandPlanning.LayoutPlan{old,old_parallel,old_time,old_copies},candidate_facts,best_facts)

def candidate(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+plan: Bands.Plan,+previous: Bands.Plan,+work: KernelPlanning.Work,+window: BandWindow.Geometry,+workers: U32,+config: Cost.Configuration,+best: BandPlanning.LayoutPlan,
  plan_facts: LayoutFacts(geometry,limits,capacity,plan),best_facts: SelectedFacts(geometry,limits,capacity,best)) ->
  SelectedFacts(geometry,limits,capacity,BandPlanning.layout_candidate(plan,previous,work,window,workers,config,best)):
  prefer(geometry,limits,capacity,BandPlanning.evaluated(plan,previous,work,window,workers,config,BandPlanning.movement(plan,workers,work,window,config)),best,
    evaluated(geometry,limits,capacity,plan,previous,work,window,workers,config,BandPlanning.movement(plan,workers,work,window,config),plan_facts),best_facts)

def layouts(+images: U32,+channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,+capacity: U32,remaining: Nat,allowed: Bool,+depth: U32,+tasks: U32,+ceiling: U32,+previous: Bands.Plan,
  +work: KernelPlanning.Work,+window: BandWindow.Geometry,+workers: U32,+config: Cost.Configuration,+best: BandPlanning.LayoutPlan,
  +domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>,best_facts: SelectedFacts(Bands.Geometry{images,channels,height,width},limits,capacity,best)) ->
  SelectedFacts(Bands.Geometry{images,channels,height,width},limits,capacity,BandPlanning.layouts(remaining,allowed,depth,tasks,ceiling,height,images,previous,work,window,workers,config,best)):
  match remaining allowed:
    case 0n allowed: best_facts
    case 1n+rest False{}: best_facts
    case 1n+rest True{}:
      +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
      +next_tasks = (tasks * 2 : U32)
      layouts(images,channels,height,width,limits,capacity,rest,U32.is_le(next_tasks,ceiling) && U32.is_le(next_tasks,height),(depth + 1 : U32),next_tasks,ceiling,previous,work,window,workers,config,
        BandPlanning.layout_candidate(Bands.image_plan(U32.to_nat(images),0,U32.to_nat(depth),height),previous,work,window,workers,config,best),domain,
        candidate(geometry,limits,capacity,Bands.image_plan(U32.to_nat(images),0,U32.to_nat(depth),height),previous,work,window,workers,config,best,
          image_plan(geometry,limits,capacity,U32.to_nat(depth),domain),best_facts))

def replan(+previous: Bands.Plan,+work: KernelPlanning.Work,+window: BandWindow.Geometry,+workers: U32,+config: Cost.Configuration,+limits: Machine.Limits,+capacity: U32,
  +domain: BandGather.Domain<BandWindow.input_geometry(window),limits,capacity>,previous_facts: LayoutFacts(BandWindow.input_geometry(window),limits,capacity,previous)) ->
  SelectedFacts(BandWindow.input_geometry(window),limits,capacity,BandPlanning.replan(previous,work,window,workers,config)):
  BandWindow.Geometry{Bands.Geometry{+images,+channels,+height,+width},+kernel,+stride,+padding,+oh,+ow} = window
  +window = {BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,oh,ow} : BandWindow.Geometry}
  +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  +bounded_workers = U32.max(1,U32.min(workers,858993459))
  layouts(images,channels,height,width,limits,capacity,32n,True{},0,1,Bool.pick(U32,U32.is_eq(bounded_workers,1),1,(bounded_workers * 5 : U32)),previous,work,window,bounded_workers,config,
    BandPlanning.evaluated(previous,previous,work,window,bounded_workers,config,BandPlanning.movement(previous,bounded_workers,work,window,config)),domain,
    evaluated(geometry,limits,capacity,previous,previous,work,window,bounded_workers,config,BandPlanning.movement(previous,bounded_workers,work,window,config),previous_facts))
