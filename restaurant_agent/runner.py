"""Interactive entry point: run the agent with real user input.

Usage:
    python -m restaurant_agent.runner          # uses LLM if a key is set, else offline parser
"""
from __future__ import annotations

from .graph import build_graph
from .nodes import default_services


def main() -> None:
    app = build_graph()  # default services: real LLM when configured, offline otherwise
    state: dict = {}
    while True:
        state = app.invoke(state, config={"recursion_limit": 50})
        print(f"\n--- Run finished: {state.get('final_result') or state.get('status')} ---")
        again = input("Start another order? (y/n): ").strip().lower()
        if again not in {"y", "yes"}:
            break


if __name__ == "__main__":
    main()
