"""Pytest wrappers around the TC1/TC2/TC3 simulations."""

from tests.simulation import run_case
from tests.test_cases import TC1, TC2, TC3


def test_tc1_order_retries_exhausted():
    state, ok, _ = run_case(TC1, quiet=True)
    assert ok
    assert state["final_result"] == "FAIL"


def test_tc2_cook_and_serve_recover():
    state, ok, _ = run_case(TC2, quiet=True)
    assert ok
    assert state["final_result"] == "SUCCESS"


def test_tc3_kitchen_retries_exhausted():
    state, ok, _ = run_case(TC3, quiet=True)
    assert ok
    assert state["final_result"] == "FAIL"
