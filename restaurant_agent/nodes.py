"""LangGraph nodes and conditional-edge routers for the order workflow.

Flow (edges are defined in graph.py):

    START -> user_input -> llm
    llm --order parsed-------------------> order_confirm --CONFIRMED--> cook
       --not an order, retries left------> user_input     (ask the user again)
       --not an order, retries exhausted-> END            (apology)
    order_confirm --PARTIAL/UNAVAILABLE---> user_decision
    user_decision --accept partial--------> cook
                 --reject, retries left---> user_input    (place a new order)
                 --reject, retries used up-> END           (apology)
    cook --READY----> serve
         --COOKING-> cook    (one more attempt while cook_retries last)
         --FAILED---> END    (apology once cook_retries hits 0)
    serve --COMPLETED-> END  (success message, final_result = SUCCESS)
          --COOKING--> cook  (remake the dish while attempts remain)
          --FAILED---> END   (apology when nothing can be retried)

Retry-counter semantics (decremented on each FAILURE, 0 = exhausted):
- order_retries (3): consumed by a non-order turn or by rejecting a
  PARTIAL/UNAVAILABLE verdict. At 0 the agent apologizes and ends.
- cook_retries (2): consumed by each failed cooking attempt. The dish is
  retried while cook_retries > 0.
- serve_retries (2): consumed by each failed serving attempt. A failed serve
  sends the dish back to cook (a remake) while attempts remain; the remake
  consumes one cook attempt. When a serve failure is blocked by an empty cook
  budget, the final reason is "cook retries exhausted".
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage

from . import kitchen
from .extraction import OrderExtraction, make_extractor, offline_extract
from .menu import lookup, menu_snapshot
from .state import OrderState

INITIAL_ORDER_RETRIES = 3  # non-order turns + rejections consume one each
INITIAL_COOK_RETRIES = 2   # cooking attempts
INITIAL_SERVE_RETRIES = 2  # serving attempts

# --------------------------------------------------------------------------- #
# Runtime services passed into the nodes (lets tests inject randomness and
# canned user answers without touching the graph itself).
# --------------------------------------------------------------------------- #

Services = dict  # simple dict-based service container


def make_services(
    extractor=None,
    cook_fn=kitchen.cook_once,
    serve_fn=kitchen.serve_once,
    input_fn=input,
) -> Services:
    return {
        "extractor": extractor,
        "cook": cook_fn,
        "serve": serve_fn,
        "input": input_fn,
    }


def default_services() -> Services:
    from .extraction import load_llm

    llm, provider = load_llm()
    print(f"[info] extraction backend: {provider}")
    return make_services(extractor=make_extractor(llm))


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

_ACCEPT_WORDS = {"y", "yes", "ok", "okay", "sure", "partial", "continue", "go ahead", "accept"}
_REJECT_WORDS = {"n", "no", "reject", "again", "new", "new order", "order again", "change", "another"}


def _ai(text: str) -> AIMessage:
    return AIMessage(content=text)


def _msg_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):  # provider-specific content blocks
        content = " ".join(str(part) for part in content)
    return str(content)


def _menu_line() -> str:
    items = ", ".join(f"{dish} ({qty})" for dish, qty in menu_snapshot().items())
    return f"Today's menu: {items}."


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #

def user_input(state: OrderState, services: Services = None) -> dict:
    """Entry node: collect one utterance from the user.

    Also (re)initializes the retry budgets whenever a brand-new session starts
    (first entry, or the runner re-invoking the graph after a finished run).
    Mid-session re-entries (after a rejection or a non-order turn) keep the
    counters so retries are actually consumed.
    """
    services = services or default_services()
    raw = services["input"]("\nYou: ").strip()
    update: dict = {
        "messages": [HumanMessage(content=raw)],
        "status": "PENDING",
        "trace": ["user_input"],
    }
    if "order_retries" not in state or state.get("final_result") is not None:
        update.update({
            "order_retries": INITIAL_ORDER_RETRIES,
            "cook_retries": INITIAL_COOK_RETRIES,
            "serve_retries": INITIAL_SERVE_RETRIES,
            "final_result": None,
            "fail_reason": None,
            "dish_name": "",
            "required_quantity": 0,
            "available_quantity": 0,
        })
    return update


def llm_node(state: OrderState, services: Services = None) -> dict:
    """LLM node: parse the latest user message into dish + quantity.

    - Non-order input -> the agent says it is a food-ordering agent only,
      consumes one order retry and asks the user to try again (END + apology
      when the retries are used up).
    - Order input     -> writes dish/quantity and routes to order_confirm.
    """
    services = services or default_services()
    extractor = services["extractor"] or offline_extract

    user_text = _msg_text(state["messages"][-1]).strip()
    extraction: OrderExtraction = extractor(user_text) if user_text else OrderExtraction(is_order=False)

    if not extraction.is_order or not extraction.dish_name:
        refusal = ("I am an AI food-ordering agent, not a general-purpose assistant. "
                   "I can only take your food order. " + _menu_line())
        remaining = state.get("order_retries", INITIAL_ORDER_RETRIES) - 1
        if remaining <= 0:
            return {
                "messages": [_ai(refusal),
                             _ai("You have used all your order attempts. We are very sorry - "
                                 "your order could not be completed.")],
                "order_retries": 0,
                "status": "FAILED",
                "final_result": "FAIL",
                "fail_reason": "order retries exhausted",
                "trace": ["llm"],
            }
        return {
            "messages": [_ai(refusal + " Please type your order.")],
            "order_retries": remaining,
            "status": "PENDING",
            "trace": ["llm"],
        }

    dish = extraction.dish_name.strip().lower()
    qty = max(1, int(extraction.required_quantity or 1))
    return {
        "messages": [_ai(f"Got it - you want {qty} x {dish}. Let me check availability.")],
        "dish_name": dish,
        "required_quantity": qty,
        "status": "PENDING",
        "trace": ["llm"],
    }


def order_confirm(state: OrderState) -> dict:
    """Compare the order against the menu and classify it.

    Writes available_quantity (0 when the dish is not on the menu or sold out)
    and the status: CONFIRMED / PARTIAL / UNAVAILABLE.
    """
    dish = state["dish_name"]
    required = state["required_quantity"]
    available = lookup(dish)

    if available == 0:
        status = "UNAVAILABLE"
        reply = (f"Sorry, {dish} is not on the menu today (or is sold out). {_menu_line()} "
                 "Would you like to order something else?")
    elif available < required:
        status = "PARTIAL"
        reply = (f"We only have {available} x {dish} left - not the {required} you asked for. "
                 "Type 'yes' to take the partial order, or 'no' to order something else.")
    else:
        status = "CONFIRMED"
        reply = f"{required} x {dish} confirmed - sending it to the kitchen!"

    return {
        "available_quantity": available,
        "status": status,
        "messages": [_ai(reply)],
        "trace": ["order_confirm"],
    }


def user_decision(state: OrderState, services: Services = None) -> dict:
    """Ask the user what to do after a PARTIAL/UNAVAILABLE verdict.

    'yes'  -> go ahead with what is available (partial) and cook it.
    'no'   -> reject; consumes one order retry. With retries left the user is
              asked for a new order; at zero the agent apologizes and ends.
    Unrecognized answers are re-asked (max 3 times) before counting as a
    rejection.
    """
    services = services or default_services()
    available = state["available_quantity"]
    dish = state["dish_name"]
    new_messages: list[Any] = []

    decision = None
    for _ in range(3):
        raw = services["input"]("You (yes = take what's available / no = order again): ").strip().lower()
        if raw in _ACCEPT_WORDS and available > 0:
            decision = "accept"
            new_messages.append(HumanMessage(content=raw))
            break
        if raw in _REJECT_WORDS:
            decision = "reject"
            new_messages.append(HumanMessage(content=raw))
            break
        # Unrecognized answer (or a pointless 'yes' on an unavailable dish).
        new_messages.append(HumanMessage(content=raw))
        if available > 0:
            new_messages.append(_ai("Sorry, I didn't catch that. "
                                    "Type 'yes' to take the partial order or 'no' to order again."))
        else:
            new_messages.append(_ai("There is nothing available to prepare for this dish. "
                                    "Type 'no' and name something else from the menu."))
    if decision is None:  # three unrecognized answers -> treat as a rejection
        decision = "reject"

    if decision == "accept":
        new_messages.append(_ai(f"Great - sending the {available} x {dish} you accepted to the kitchen."))
        return {
            "messages": new_messages,
            "required_quantity": available,
            "status": "PENDING",
            "trace": ["user_decision"],
        }

    remaining = state.get("order_retries", INITIAL_ORDER_RETRIES) - 1
    if remaining <= 0:
        new_messages.append(_ai("That was your last allowed order attempt. We are very sorry - "
                                "your order could not be completed."))
        return {
            "messages": new_messages,
            "order_retries": 0,
            "status": "FAILED",
            "final_result": "FAIL",
            "fail_reason": "order retries exhausted",
            "dish_name": "",
            "required_quantity": 0,
            "available_quantity": 0,
            "trace": ["user_decision"],
        }
    new_messages.append(_ai("No problem - please place a new order. " + _menu_line()))
    return {
        "messages": new_messages,
        "order_retries": remaining,
        "status": "PENDING",
        "dish_name": "",
        "required_quantity": 0,
        "available_quantity": 0,
        "trace": ["user_decision"],
    }


def cook(state: OrderState, services: Services = None) -> dict:
    """One cooking attempt: 60% success, up to 2 attempts (cook_retries)."""
    services = services or default_services()
    dish = state["dish_name"]
    qty = state["required_quantity"]

    ok = services["cook"]()
    if ok:
        return {
            "messages": [_ai(f"Cooking {qty} x {dish}... done! The dish is ready.")],
            "status": "READY",
            "trace": ["cook"],
        }
    remaining = state["cook_retries"] - 1
    if remaining <= 0:
        return {
            "messages": [_ai(f"Cooking {qty} x {dish} failed again and the kitchen has no attempts "
                             "left. We are very sorry - your order could not be completed.")],
            "status": "FAILED",
            "cook_retries": 0,
            "final_result": "FAIL",
            "fail_reason": "cooking failed, retries exhausted",
            "trace": ["cook"],
        }
    return {
        "messages": [_ai(f"Cooking {qty} x {dish} failed. The kitchen is trying again... "
                         f"({remaining} attempt{'s' if remaining != 1 else ''} left)")],
        "status": "COOKING",
        "cook_retries": remaining,
        "trace": ["cook"],
    }


def serve(state: OrderState, services: Services = None) -> dict:
    """One serving attempt (simulated pass/fail), up to 2 attempts.

    A failed serve is sent back to cook for a remake only while serve AND cook
    attempts remain; the remake consumes one cook attempt. When the blocked
    transition is the cook retry (no cook attempts left for a remake), the
    final reason is "cook retries exhausted" instead of a serving failure.
    """
    services = services or default_services()
    dish = state["dish_name"]
    qty = state["required_quantity"]

    ok = services["serve"]()
    if ok:
        return {
            "messages": [_ai(f"Serving {qty} x {dish}... done. Your order is complete - enjoy your meal!")],
            "status": "COMPLETED",
            "final_result": "SUCCESS",
            "fail_reason": None,
            "trace": ["serve"],
        }
    remaining = state["serve_retries"] - 1
    if remaining > 0 and state["cook_retries"] > 0:
        return {
            "messages": [_ai(f"Serving {qty} x {dish} failed. Sending it back to the kitchen "
                             "for a remake...")],
            "status": "COOKING",
            "serve_retries": remaining,
            "cook_retries": state["cook_retries"] - 1,  # the remake consumes a cook attempt
            "trace": ["serve"],
        }
    if state["cook_retries"] <= 0:
        # The blocked transition is the cook retry: remaking the dish would
        # need a cook attempt, but none remain.
        reason = "cook retries exhausted - cannot remake the dish"
        apology = (f"Serving {qty} x {dish} failed and the kitchen has no cook attempts left "
                   "to remake the dish. We are very sorry - your order could not be completed.")
    else:
        reason = "serving failed, retries exhausted"
        apology = (f"Serving {qty} x {dish} failed and no serving attempts remain. "
                   "We are very sorry - your order could not be completed.")
    return {
        "messages": [_ai(apology)],
        "status": "FAILED",
        "serve_retries": max(0, remaining),
        "final_result": "FAIL",
        "fail_reason": reason,
        "trace": ["serve"],
    }


# --------------------------------------------------------------------------- #
# Conditional-edge routers
# --------------------------------------------------------------------------- #

def route_after_llm(state: OrderState) -> Literal["order_confirm", "user_input", "__end__"]:
    if state.get("final_result") == "FAIL":
        return "__end__"
    return "order_confirm" if state.get("dish_name") else "user_input"


def route_after_confirm(state: OrderState) -> Literal["cook", "user_decision"]:
    return "cook" if state["status"] == "CONFIRMED" else "user_decision"


def route_after_decision(state: OrderState) -> Literal["cook", "user_input", "__end__"]:
    if state.get("final_result") == "FAIL":
        return "__end__"
    return "cook" if state.get("dish_name") else "user_input"


def route_after_cook(state: OrderState) -> Literal["serve", "cook", "__end__"]:
    if state["status"] == "READY":
        return "serve"
    if state["status"] == "COOKING":  # failed but attempts remain -> retry
        return "cook"
    return "__end__"


def route_after_serve(state: OrderState) -> Literal["cook", "__end__"]:
    if state["status"] == "COOKING":  # failed serve -> remake while attempts remain
        return "cook"
    return "__end__"  # COMPLETED or FAILED both finish the run
