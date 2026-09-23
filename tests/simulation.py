"""TC1/TC2/TC3 simulation harness.

Runs the compiled LangGraph with scripted user input and an injected
randomness sequence, then prints the full message log, node trace, counters
and final result, and checks them against the expectations in test_cases.py.

Usage:
    python -m tests.simulation      (from the project root)
"""
from __future__ import annotations

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from restaurant_agent.extraction import offline_extract
from restaurant_agent.graph import build_graph
from restaurant_agent.nodes import make_services


def run_case(spec: dict, quiet: bool = False):
    """Run one test case end-to-end.

    Returns (final_state, ok, failures) where `ok` says whether every
    expectation (final result, failure reason, node trace counts) held.
    """
    inputs = list(spec["inputs"])
    outcomes = list(spec["outcomes"])
    random.seed(1234)  # fixed seed; the injected sequence is deterministic anyway

    services = make_services(
        extractor=offline_extract,
        cook_fn=lambda: outcomes.pop(0),
        serve_fn=lambda: outcomes.pop(0),
        input_fn=lambda _prompt="": inputs.pop(0),
    )
    app = build_graph(services=services)
    state = app.invoke({}, config={"recursion_limit": 100})

    trace = state.get("trace", [])
    failures: list[str] = []

    if state.get("final_result") != spec["expect"]:
        failures.append(f"final_result: expected {spec['expect']!r}, got {state.get('final_result')!r}")
    reason = state.get("fail_reason")
    if spec.get("expect_reason") and (not reason or spec["expect_reason"] not in reason):
        failures.append(f"fail_reason: expected ~{spec['expect_reason']!r}, got {reason!r}")
    for node, count in spec.get("expect_node_counts", {}).items():
        if trace.count(node) != count:
            failures.append(f"node '{node}' visited {trace.count(node)}x, expected {count}x")
    ok = not failures

    if not quiet:
        _print_report(spec, state, trace, ok, failures)
    return state, ok, failures


def _print_report(spec, state, trace, ok, failures):
    line = "=" * 76
    print(line)
    print(spec["name"])
    print(line)
    for m in state["messages"]:
        speaker = "USER " if m.type == "human" else "AGENT"
        print(f"  {speaker}: {m.content}")
    print(f"\n  node trace     : {' -> '.join(trace)}")
    print(f"  dish/req/avail : {state.get('dish_name')!r} / "
          f"{state.get('required_quantity')} / {state.get('available_quantity')}")
    print(f"  retries left   : order={state.get('order_retries')} "
          f"cook={state.get('cook_retries')} serve={state.get('serve_retries')}")
    print(f"  fail_reason    : {state.get('fail_reason')}")
    print(f"  FINAL RESULT   : {state.get('final_result')}   (expected: {spec['expect']})")
    if ok:
        print("  RESULT         : PASS [OK]")
    else:
        print("  RESULT         : FAIL [X]")
        for f in failures:
            print(f"                   - {f}")


def main() -> None:
    from tests.test_cases import TC1, TC2, TC3

    for spec in (TC1, TC2, TC3):
        run_case(spec)


if __name__ == "__main__":
    main()
