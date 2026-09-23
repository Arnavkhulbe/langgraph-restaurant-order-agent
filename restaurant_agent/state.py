"""LangGraph state definition for the restaurant order agent.

The state is a TypedDict annotated with reducer functions:
- `messages`: add_messages reducer -> LLM / Human messages are appended, not overwritten.
- `trace`: operator.add reducer -> every node appends its own name, giving the
  exact execution path (used to verify the TC1/TC2/TC3 workflow routes).
- plain fields (order details, status, counters, result): last-value-wins,
  i.e. every node write simply overwrites the previous value.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages

Status = Literal["PENDING", "CONFIRMED", "PARTIAL", "UNAVAILABLE", "COOKING", "READY", "COMPLETED", "FAILED"]


class OrderState(TypedDict):
    # Chat history (LLM + user), appended via add_messages reducer.
    messages: Annotated[list[Any], add_messages]

    # Node execution trace, e.g. ["user_input", "llm", "order_confirm", ...]
    trace: Annotated[list[str], operator.add]

    # ---- Order details ----
    dish_name: str
    required_quantity: int
    available_quantity: int  # written by order_confirm by reading the menu; 0 when the dish is absent

    # ---- Workflow status ----
    status: Status

    # ---- Retry counters (decremented on each failure; 0 means exhausted) ----
    order_retries: int  # 3 user order attempts (non-order turns and rejections consume one)
    cook_retries: int   # 2 cooking attempts
    serve_retries: int  # 2 serving attempts

    # ---- Final result ----
    final_result: Optional[str]  # "SUCCESS" | "FAIL"
    fail_reason: Optional[str]
