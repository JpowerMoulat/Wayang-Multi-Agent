from typing import TypedDict, Optional, List, Any
from ai_wayang_multi.llm.models import WayangPlan, WayangPlanHighLevel, DataSources


class WayangAgentState(TypedDict):
    # Input variables
    query: str
    model: Optional[str]
    reasoning: Optional[str]
    use_debugger: bool

    # Interpretation layer outputs
    refined_query: Optional[str]        # Output of Specifier
    data_selected: Optional[DataSources]  # Output of Selector

    # Decomposition layer output
    highlevel_plan: Optional[WayangPlanHighLevel]  # Output of Decomposer

    # Working plan state (passed through refiner, mapper, validator, executor, debugger)
    raw_plan: Optional[WayangPlan]      # Output of Builder merge, Refiner, or Debugger
    wayang_plan: Optional[dict]         # Output of PlanMapper (executable JSON)
    val_success: bool
    val_errors: List[str]
    status_code: Optional[int]
    result: Optional[str]               # Raw response from Wayang server
    iteration: int                      # Debugger iteration counter

    # Agent instances stored to preserve chat history across debug loop iterations
    refiner: Any
    debugger: Any

    # Logger instance (created in main, stored to write to a single session log file)
    logger: Any
