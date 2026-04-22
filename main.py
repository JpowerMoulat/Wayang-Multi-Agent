"""
Entrypoint — CLI interface for AI-Wayang-Multi
"""
import sys
from pathlib import Path

# Add src folder so modules can be found
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from ai_wayang_multi.agent.graph import graph
from ai_wayang_multi.config.settings import OUTPUT_CONFIG, DEBUGGER_AGENT_CONFIG
from ai_wayang_multi.utils.logger import Logger


def main():
    output_folder = OUTPUT_CONFIG.get("output_folder") or "the configured output folder"

    print("AI Wayang - Natural Language to Wayang Plan")
    print("--------------------------------------------")

    query = input("Enter your query: ").strip()
    if not query:
        print("No query provided. Exiting.")
        return

    print("\nProcessing query...")

    use_debugger = DEBUGGER_AGENT_CONFIG.get("use_debugger") == "True"

    logger = Logger()
    logger.add_message("User query", query)
    logger.add_message("Architecture", {"architecture": "Multi", "debugger": use_debugger})

    initial_state = {
        "query": query,
        "model": None,          # uses agent defaults from .env
        "reasoning": None,      # uses agent defaults from .env
        "use_debugger": use_debugger,
        "refined_query": None,
        "data_selected": None,
        "highlevel_plan": None,
        "raw_plan": None,
        "wayang_plan": None,
        "val_success": False,
        "val_errors": [],
        "status_code": None,
        "result": None,
        "iteration": 0,
        "refiner": None,
        "debugger": None,
        "logger": logger,
    }

    final_state = graph.invoke(initial_state)

    print()
    if final_state["status_code"] == 200:
        output_path = None
        wayang_plan = final_state.get("wayang_plan") or {}
        for op in wayang_plan.get("operators", []):
            if op.get("cat") == "output":
                output_path = op.get("data", {}).get("filename")
                break

        if output_path:
            display_path = output_path.replace("file:///", "/")
            print(f"Success! Output written to: {display_path}")
        else:
            print(f"Success! Output written to: {output_folder}")
    else:
        print(f"Failed to execute query (status {final_state['status_code']}).")
        if final_state.get("result"):
            print(f"Error: {final_state['result']}")


if __name__ == "__main__":
    main()
