from langgraph.graph import StateGraph, END

from ai_wayang_multi.agent.state import WayangAgentState
from ai_wayang_multi.llm.agent_specifier import Specifier
from ai_wayang_multi.llm.agent_selector import Selector
from ai_wayang_multi.llm.agent_decomposer import Decomposer
from ai_wayang_multi.llm.agent_builder import Builder
from ai_wayang_multi.llm.agent_refiner import Refiner
from ai_wayang_multi.llm.agent_debugger import Debugger
from ai_wayang_multi.wayang.step_handler import StepHandler
from ai_wayang_multi.wayang.plan_mapper import PlanMapper
from ai_wayang_multi.wayang.plan_validator import PlanValidator
from ai_wayang_multi.wayang.wayang_executor import WayangExecutor
from ai_wayang_multi.config.settings import INPUT_CONFIG, OUTPUT_CONFIG, DEBUGGER_AGENT_CONFIG

# Module-level objects with no mutable per-request state
_config = {"input_config": INPUT_CONFIG, "output_config": OUTPUT_CONFIG}
_step_handler = StepHandler()
_plan_mapper = PlanMapper(config=_config)
_plan_validator = PlanValidator()
_wayang_executor = WayangExecutor()

MAX_ITERATIONS = int(DEBUGGER_AGENT_CONFIG.get("max_itr", 5))

# --- Logging helper-functions ---


def _get_raw_meta(raw) -> dict:
    """Extract model name and token usage from a LangChain AIMessage."""
    meta = raw.response_metadata if hasattr(raw, "response_metadata") else {}
    return {
        "model": meta.get("model_name", ""),
        "usage": meta.get("token_usage", {}),
    }


def _log(state: WayangAgentState, title: str, msg) -> None:
    """Write a log entry if a logger is present in state."""
    logger = state.get("logger")
    if logger:
        logger.add_message(title, msg)


# --- Node functions ---

def specifier_node(state: WayangAgentState) -> dict:
    """Clarifies and refines the natural-language query."""
    print("[INFO] SpecifierAgent: Refining user query")
    specifier = Specifier(model=state["model"], reasoning=state["reasoning"])
    specifier.start()
    response = specifier.generate(state["query"])
    print("[INFO] SpecifierAgent: Query refined")

    _log(state, "Agent Usage: SpecifierAgent Information", _get_raw_meta(response["raw"]))
    _log(state, "Agent: SpecifierAgent Output", response["refined_query"])

    return {"refined_query": response["refined_query"]}


def selector_node(state: WayangAgentState) -> dict:
    """Selects the relevant data sources for the query."""
    print("[INFO] SelectorAgent: Selecting relevant data sources")
    selector = Selector(model=state["model"], reasoning=state["reasoning"])
    selector.start()
    response = selector.generate(state["query"])
    print("[INFO] SelectorAgent: Data sources selected")

    _log(state, "Agent Usage: SelectorAgent Information", _get_raw_meta(response["raw"]))
    _log(state, "Agent: SelectorAgent Output", response["selected_data"].model_dump())

    return {"data_selected": response["selected_data"]}


def decomposer_node(state: WayangAgentState) -> dict:
    """Decomposes the query into a high-level step plan."""
    print("[INFO] DecomposerAgent: Decomposing query into steps")
    decomposer = Decomposer(model=state["model"], reasoning=state["reasoning"])
    decomposer.start()
    response = decomposer.generate(state["refined_query"], state["data_selected"])
    print("[INFO] DecomposerAgent: High-level plan created")

    _log(state, "Agent Usage: DecomposerAgent Information", _get_raw_meta(response["raw"]))
    _log(state, "Agent: DecomposerAgent Output", response["response"].model_dump())

    return {"highlevel_plan": response["response"]}


def build_steps_node(state: WayangAgentState) -> dict:
    """Translates each step of the high-level plan into Wayang operators."""
    print("[INFO] BuilderAgent(s): Building step subplans")
    builder = Builder(model=state["model"], reasoning=state["reasoning"])
    builder.start(state["refined_query"], state["data_selected"])

    steps = state["highlevel_plan"].steps # Create the steps of the full plan
    step_dependencies = _step_handler.build_step_dependency_map(steps) # Map the sequential dependencies
    step_queue = _step_handler.build_step_queue(step_dependencies) # Create the stepque

    _log(state, "Class: StepHandler Step dependencies and queue", f"Step queue {step_queue}")

    subplans = {}
    for step_id in step_queue: # For each step, generate a subplan
        current_step = next(s for s in steps if s.step_id == step_id)
        previous_steps = _step_handler.get_steps(step_dependencies.get(step_id, []), subplans, step_queue) # Feed the previous subplans to the current one
        response = builder.generate(current_step, previous_steps) # Generate the current subplan
        subplans = _step_handler.update_subplan(step_id, response["wayang_subplan"], subplans)
        
        print(f"[INFO] BuilderAgent: Subplan built for step {step_id}")
        _log(state, f"Agent Usage: BuilderAgent Information step {step_id}", _get_raw_meta(response["raw"]))
        _log(state, f"Agent: BuilderAgent Subplan for step {step_id}", response["wayang_subplan"].model_dump())

    raw_plan = _step_handler.step_merger(step_queue, subplans) # Concat the full subplan
    print("[INFO] StepHandler: Subplans merged into full plan")
    _log(state, "Class: StepHandler Merged subplans to full plan", raw_plan.model_dump())

    return {"raw_plan": raw_plan}


def refine_plan_node(state: WayangAgentState) -> dict:
    """Refines the full plan for executability. Reuses the Refiner instance in the debug loop."""
    refiner = state.get("refiner")
    if refiner is None:
        print("[INFO] RefinerAgent: Refining full plan")
        refiner = Refiner(model=state["model"], reasoning=state["reasoning"])
        refiner.start(state["data_selected"])
    else:
        print("[INFO] RefinerAgent: Refining debugged plan")
        refiner.set_model_and_reasoning(state["model"], state["reasoning"])

    response = refiner.generate(state["refined_query"], state["raw_plan"])
    print("[INFO] RefinerAgent: Plan refined")

    version = state.get("iteration", 0) + 1
    _log(state, f"Agent Usage: RefinerAgent version {version} information", _get_raw_meta(response["raw"]))
    _log(state, f"Agent: RefinerAgent plan version {version}", response["wayang_plan"].model_dump())

    return {"raw_plan": response["wayang_plan"], "refiner": refiner}


def map_plan_node(state: WayangAgentState) -> dict:
    """Maps the abstract WayangPlan to an executable JSON plan."""
    print("[INFO] PlanMapper: Mapping plan to executable JSON")
    wayang_plan = _plan_mapper.plan_to_json(state["raw_plan"])
    print("[INFO] PlanMapper: Plan mapped")

    _log(state, "Class: PlanMapper Mapped plan", {"version": state.get("iteration", 0) + 1, "plan": wayang_plan})

    return {"wayang_plan": wayang_plan}


def validate_plan_node(state: WayangAgentState) -> dict:
    """Validates the executable JSON plan."""
    print("[INFO] PlanValidator: Validating plan")
    val_success, val_errors = _plan_validator.validate_plan(state["wayang_plan"])

    if val_success:
        print("[INFO] PlanValidator: Plan validated successfully")
        _log(state, "Class: PlanValidator Plan validated successfully", "")
    else:
        print(f"[INFO] PlanValidator: Plan failed validation: {val_errors}")
        _log(state, "Err: PlanValidator Failed validation", {"version": state.get("iteration", 0) + 1, "errors": val_errors})

    return {"val_success": val_success, "val_errors": val_errors}


def execute_plan_node(state: WayangAgentState) -> dict:
    """Sends the validated JSON plan to the Wayang server for execution."""
    print("[INFO] WayangExecutor: Sending plan to Wayang for execution")
    status_code, result = _wayang_executor.execute_plan(state["wayang_plan"])

    if status_code == 200:
        print("[INFO] WayangExecutor: Plan executed successfully")
        _log(state, "Wayang: Plan executed successfully", "")
    else:
        print(f"[ERROR] WayangExecutor: Execution failed, status {status_code}")
        _log(state, "Err: Wayang error. Execution failed", {"status_code": status_code, "output": result})

    return {"status_code": status_code, "result": result}


def debug_plan_node(state: WayangAgentState) -> dict:
    """Uses the Debugger agent to fix a failed plan, preserving chat history across iterations."""
    debugger = state.get("debugger")
    if debugger is None:
        print("[INFO] DebuggerAgent: Starting debug session")
        debugger = Debugger(model=state["model"], reasoning=state["reasoning"])
        debugger.set_vesion(1)
        debugger.start()
    else:
        debugger.set_model_and_reasoning(state["model"], state["reasoning"])

    failed_plan = _plan_mapper.plan_from_json(state["wayang_plan"])
    response = debugger.debug_plan(
        query=state["refined_query"],
        plan=failed_plan,
        wayang_errors=state["result"],
        val_errors=state["val_errors"],
    )
    version = debugger.get_version()
    print(f"[INFO] DebuggerAgent: Produced plan version {version}")

    _log(state, f"Agent Usage: DebuggerAgent version {version} information", _get_raw_meta(response["raw"]))
    _log(state, f"Agent: DebuggerAgent thoughts plan {version}", {"version": version, "thoughts": response["wayang_plan"].thoughts})
    _log(state, f"Agent: DebuggerAgent plan {version}", {"version": version, "plan": response["wayang_plan"].model_dump()})

    return {
        "raw_plan": response["wayang_plan"],
        "iteration": version,
        "debugger": debugger,
    }


def finish_node(state: WayangAgentState) -> dict:
    """Terminal node — logs final outcome."""
    if state.get("status_code") == 200:
        _log(state, "Final: Successful. Plan executed", "Success")
    else:
        _log(state, "Final: Unsuccessful. Plan executed unsuccessful", {"status_code": state.get("status_code"), "output": state.get("result")})
    return {}


# --- Routing functions ---

def route_after_validate(state: WayangAgentState) -> str:
    if state["val_success"]:
        return "execute_plan"
    if state["use_debugger"] and state["iteration"] < MAX_ITERATIONS:
        return "debug_plan"
    return "finish"


def route_after_execute(state: WayangAgentState) -> str:
    if state["status_code"] == 200:
        return "finish"
    if state["use_debugger"] and state["iteration"] < MAX_ITERATIONS:
        return "debug_plan"
    return "finish"


# --- Build the graph ---

workflow = StateGraph(WayangAgentState)

workflow.add_node("specifier", specifier_node)
workflow.add_node("selector", selector_node)
workflow.add_node("decomposer", decomposer_node)
workflow.add_node("build_steps", build_steps_node)
workflow.add_node("refine_plan", refine_plan_node)
workflow.add_node("map_plan", map_plan_node)
workflow.add_node("validate_plan", validate_plan_node)
workflow.add_node("execute_plan", execute_plan_node)
workflow.add_node("debug_plan", debug_plan_node)
workflow.add_node("finish", finish_node)

workflow.set_entry_point("specifier")
workflow.add_edge("specifier", "selector")
workflow.add_edge("selector", "decomposer")
workflow.add_edge("decomposer", "build_steps")
workflow.add_edge("build_steps", "refine_plan")
workflow.add_edge("refine_plan", "map_plan")
workflow.add_edge("map_plan", "validate_plan")
workflow.add_conditional_edges("validate_plan", route_after_validate)
workflow.add_conditional_edges("execute_plan", route_after_execute)
workflow.add_edge("debug_plan", "refine_plan")
workflow.add_edge("finish", END)

graph = workflow.compile()
