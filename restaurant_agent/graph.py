"""Graph assembly: State, Nodes, Edges.

    START -> user_input -> llm
    llm --order parsed-------------------> order_confirm --CONFIRMED--> cook
       --not an order, retries left------> user_input     (ask again)
       --not an order, retries exhausted-> END            (apology)
    order_confirm --PARTIAL/UNAVAILABLE---> user_decision
    user_decision --accept partial--------> cook
                 --reject, retries left---> user_input    (new order)
                 --reject, retries used up-> END           (apology)
    cook --READY----> serve
         --COOKING-> cook    (retry while cook_retries last)
         --FAILED---> END    (apology once cook_retries hits 0)
    serve --COMPLETED-> END  (success, final_result = SUCCESS)
          --COOKING--> cook  (remake while attempts remain)
          --FAILED---> END   (apology when nothing can be retried)
"""

from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from .nodes import (
    Services,
    cook,
    default_services,
    llm_node,
    order_confirm,
    route_after_cook,
    route_after_confirm,
    route_after_decision,
    route_after_llm,
    route_after_serve,
    serve,
    user_decision,
    user_input,
)
from .state import OrderState


def build_graph(services: Services = None):
    """Compile the order-management workflow.

    `services` lets callers (tests, simulations) inject the extractor, the
    cook/serve randomness functions and a scripted input function.
    """
    services = services or default_services()

    g = StateGraph(OrderState)
    g.add_node("user_input", partial(user_input, services=services))
    g.add_node("llm", partial(llm_node, services=services))
    g.add_node("order_confirm", order_confirm)
    g.add_node("user_decision", partial(user_decision, services=services))
    g.add_node("cook", partial(cook, services=services))
    g.add_node("serve", partial(serve, services=services))

    # ---- Edges ----
    g.add_edge(START, "user_input")
    g.add_edge("user_input", "llm")

    g.add_conditional_edges(
        "llm", route_after_llm,
        {"order_confirm": "order_confirm", "user_input": "user_input", "__end__": END},
    )
    g.add_conditional_edges(
        "order_confirm", route_after_confirm,
        {"cook": "cook", "user_decision": "user_decision"},
    )
    g.add_conditional_edges(
        "user_decision", route_after_decision,
        {"cook": "cook", "user_input": "user_input", "__end__": END},
    )
    g.add_conditional_edges(
        "cook", route_after_cook,
        {"serve": "serve", "cook": "cook", "__end__": END},
    )
    g.add_conditional_edges(
        "serve", route_after_serve,
        {"cook": "cook", "__end__": END},
    )

    return g.compile()
