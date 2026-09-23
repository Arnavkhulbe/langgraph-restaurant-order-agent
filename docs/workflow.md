# LangGraph Workflow Diagram

Topology below is generated from the compiled graph
(`restaurant_agent/graph.py`); the edge labels mirror the conditional-edge
routers in `restaurant_agent/nodes.py`.

- **Solid edges** are unconditional.
- **Dotted edges** are conditional (chosen at runtime by a router function).
- Dotted **self-loops** on `cook` are the cook-retry path (`COOKING` status).

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
    __start__([__start__]):::first
    user_input(user_input)
    llm(llm)
    order_confirm(order_confirm)
    user_decision(user_decision)
    cook(cook)
    serve(serve)
    __end__([__end__]):::last

    __start__ --> user_input;
    user_input --> llm;

    llm -.->|order parsed| order_confirm;
    llm -.->|not an order, retries left| user_input;
    llm -.->|not an order, retries exhausted| __end__;

    order_confirm -.->|CONFIRMED| cook;
    order_confirm -.->|PARTIAL / UNAVAILABLE| user_decision;

    user_decision -.->|accept partial| cook;
    user_decision -.->|reject, retries left| user_input;
    user_decision -.->|reject, retries exhausted| __end__;

    cook -.->|READY| serve;
    cook -.->|COOKING, retries left| cook;
    cook -.->|FAILED, retries exhausted| __end__;

    serve -.->|COMPLETED| __end__;
    serve -.->|failed, serve AND cook attempts left| cook;
    serve -.->|failed, remake not possible| __end__;

    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0
    classDef last fill:#bfb6fc
```

## Edge conditions (source of truth: `route_*` functions in `nodes.py`)

| From | Condition | To |
|---|---|---|
| `llm` | dish extracted | `order_confirm` |
| `llm` | not an order, `order_retries` left | `user_input` |
| `llm` | not an order, `order_retries == 0` | END (apology) |
| `order_confirm` | `status == CONFIRMED` | `cook` |
| `order_confirm` | `PARTIAL` / `UNAVAILABLE` | `user_decision` |
| `user_decision` | accepted partial (`dish_name` kept) | `cook` |
| `user_decision` | rejected, `order_retries` left | `user_input` |
| `user_decision` | rejected, `order_retries == 0` | END (apology) |
| `cook` | `status == READY` | `serve` |
| `cook` | `status == COOKING` (failed, attempts left) | `cook` |
| `cook` | `status == FAILED` (retries exhausted) | END (apology) |
| `serve` | `status == COMPLETED` | END (success) |
| `serve` | failed, **both** conditions hold: a serve attempt remains (`serve_retries > 1` after this failure) **and** a cook attempt is available (`cook_retries > 0`) | `cook` (remake; consumes one cook attempt) |
| `serve` | failed, remake not possible: no serve attempt remains (`serve_retries == 0`) **or** no cook attempt is available (`cook_retries == 0`) | END (apology — "cook retries exhausted - cannot remake the dish" when `cook_retries == 0`, otherwise "serving failed, retries exhausted") |

> **Note:** the remake edge requires **both** counters — `serve_retries` alone
> never permits a transition to `cook`. A failed serve with serve attempts
> left but an empty cook budget still ends the run, because the blocked
> transition is the cook retry. This gate lives inside the `serve` node
> (`nodes.py`); the router simply follows the `COOKING` status that `serve`
> sets only when the remake is permitted.
