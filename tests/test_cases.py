"""The three test scenarios: scripted user inputs + injected cook/serve outcomes.

`outcomes` is consumed in order: every cook attempt pops the first value for a
cook roll, every serve attempt pops the first value for a serve roll.
True = success, False = failure.

`expect_node_counts` pins the exact node path taken (the "workflow route").
Menu reference: pizza 5, burger 3, pasta 2, salad 4, biryani 6, dosa 8.
"""
from __future__ import annotations

TC1 = {
    "name": "TC1 - unrelated question, rejected partial, unavailable dish -> END on order retries",
    "inputs": [
        "What's the weather like today?",  # not an order -> refusal, one order retry used (3->2)
        "8 pizzas",                        # PARTIAL: only 5 available
        "no",                              # reject partial -> order again (2->1)
        "1 chocolate cake",                # UNAVAILABLE: not on the menu (available_quantity = 0)
        "no",                              # reject -> last attempt used (1->0) -> END
    ],
    "outcomes": [],                        # kitchen never reached
    "expect": "FAIL",
    "expect_reason": "order retries exhausted",
    "expect_node_counts": {"cook": 0, "serve": 0, "order_confirm": 2, "user_decision": 2},
}

TC2 = {
    "name": "TC2 - available order; cook fails once, serve fails once -> SUCCESS",
    "inputs": [
        "I want 3 burgers",                # menu has 3 -> CONFIRMED
    ],
    "outcomes": [False, True, False, True, True],
    # cook fail -> cook ok -> serve fail -> cook remake ok -> serve ok
    "expect": "SUCCESS",
    "expect_reason": None,
    "expect_node_counts": {"cook": 3, "serve": 2, "order_confirm": 1, "user_decision": 0},
}

TC3 = {
    "name": "TC3 - rejected partial, then serve fails twice; cook retries exhausted -> FAIL",
    "inputs": [
        "4 pastas",                        # PARTIAL: menu has 2
        "no",                              # reject -> order again
        "2 dosas",                         # menu has 8 -> CONFIRMED
    ],
    "outcomes": [False, True, False, True, False],
    # cook fail -> cook ok -> serve fail -> cook remake ok -> serve fail -> END
    # (the remake used the last cook attempt, so no further cook retry is possible)
    "expect": "FAIL",
    "expect_reason": "cook retries exhausted",
    "expect_node_counts": {"cook": 3, "serve": 2, "order_confirm": 2, "user_decision": 1},
}
