# Observe the actual planner choice together with its allocation and coverage
# evidence. Consumers can reason about the selected layout before transporting
# the result to the executable replan expression.
import Base
import ../tensor_band.bend as Bands
import ../tensor_band_window.bend as BandWindow
import ../kernel_band_planning.bend as BandPlanning
import ../kernel_planning.bend as KernelPlanning
import ../kernel_cost.bend as Cost
import ../machine_limits.bend as Machine
import ./kernel_band_planning_refinement.bend as KernelBandPlanningRefinement
import ./tensor_band_gather.bend as BandGather

type ReplanObservation<-previous: Bands.Plan,-work: KernelPlanning.Work,-window: BandWindow.Geometry,-workers: U32,-configuration: Cost.Configuration,-limits: Machine.Limits,-capacity: U32> is Data:
  ReplanObservation{selection: BandPlanning.LayoutPlan,
    exact: {selection == BandPlanning.replan(previous,work,window,workers,configuration) : BandPlanning.LayoutPlan},
    facts: KernelBandPlanningRefinement.SelectedFacts(BandWindow.input_geometry(window),limits,capacity,selection)}

def observe(+previous: Bands.Plan,+work: KernelPlanning.Work,+window: BandWindow.Geometry,+workers: U32,+configuration: Cost.Configuration,+limits: Machine.Limits,+capacity: U32,
  domain: BandGather.Domain<BandWindow.input_geometry(window),limits,capacity>,
  previous_facts: KernelBandPlanningRefinement.LayoutFacts(BandWindow.input_geometry(window),limits,capacity,previous)) ->
  ReplanObservation<previous,work,window,workers,configuration,limits,capacity>:
  +selection = BandPlanning.replan(previous,work,window,workers,configuration)
  ReplanObservation{selection,{==},KernelBandPlanningRefinement.replan(previous,work,window,workers,configuration,limits,capacity,domain,previous_facts)}
