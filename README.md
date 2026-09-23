# Restaurant Order Management AI Agent (LangGraph)

A conversational food-ordering agent built as a **LangGraph state machine**.
The user types an order, an LLM extracts `dish + quantity`, an `order_confirm`
node checks the menu, and the `cook` / `serve` nodes simulate the kitchen with
probabilistic outcomes and retry budgets.

## Architecture

```
                START
                  |
             user_input  <-----------------------------+
                  |                                    |
                 llm ---- not an order (retries) ------+
                  |            \--- not an order (0 retries) --> END (apology)
           order_confirm -- CONFIRMED ------------------------------> cook
                  |                                                   |
       PARTIAL / UNAVAILABLE                                     READY | COOKING (retry)
                  |                                              |        |
           user_decision -- accept partial --> cook          serve <------+
                  |                                            |
            reject (retries) --> user_input          COMPLETED | COOKING (remake)
            reject (0 retries) --> END (apology)         |         |
                                                  END (success)   cook
                                                                  |
                                                            0 retries -> END (apology)
```

Nodes: `user_input`, `llm`, `order_confirm`, `user_decision`, `cook`, `serve`
(declared in `restaurant_agent/nodes.py`, wired in `restaurant_agent/graph.py`).

## State (`restaurant_agent/state.py`)

| Field | Meaning |
|---|---|
| `messages` | Annotated with `add_messages` — full chat history (user + agent) |
| `trace` | Annotated with `operator.add` — ordered list of visited nodes |
| `dish_name` / `required_quantity` | Written by the `llm` node |
| `available_quantity` | Written by `order_confirm` from the menu; **0** when the dish is missing |
| `status` | `PENDING → CONFIRMED / PARTIAL / UNAVAILABLE → COOKING → READY → COMPLETED` or `FAILED` |
| `order_retries` (3) | Consumed by a non-order turn **or** by rejecting a partial/unavailable verdict |
| `cook_retries` (2) | Consumed by each failed cooking attempt |
| `serve_retries` (2) | Consumed by each failed serving attempt |
| `final_result` | `"SUCCESS"` / `"FAIL"` (+ `fail_reason`) |

Counters decrement on each failure and are read by the routers/nodes to decide
retry vs. apology + END. A failed serve goes back to `cook` for a remake while
attempts remain (the remake consumes one cook attempt); a failed cook retries
while `cook_retries > 0`. A serve failure blocked by an empty cook budget ends
the run with the reason "cook retries exhausted".

## Kitchen simulation (`restaurant_agent/kitchen.py`)

- `cook`: **60% success / 40% failure** per attempt.
- `serve`: simulated pass/fail (**60% success / 40% failure**, matching cook).
- Tests inject deterministic outcomes; a fixed `random.seed` makes runs reproducible.

## Menu (`restaurant_agent/menu.py`)

pizza 5, burger 3, pasta 2, salad 4, biryani 6, dosa 8 — quantities are
placeholder; edit `MENU` to change stock. Availability is currently static
(stock is not decremented after orders).

## LLM extraction (`restaurant_agent/extraction.py`)

Set `OPENAI_API_KEY` or `GOOGLE_API_KEY` (see `.env.example`) for real LLM
extraction with structured output. With no key configured the agent falls back
to a deterministic offline parser so everything runs offline.

## Run

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python

# interactive chat
.venv/Scripts/python -m restaurant_agent.runner

# TC1 / TC2 / TC3 simulations with verdicts
.venv/Scripts/python -m tests.simulation

# pytest wrappers for the same three cases
.venv/Scripts/python -m pytest -q
```

## Test cases

| TC | Script | Expected | Result |
|---|---|---|---|
| TC1 | unrelated question → partial order (8 pizzas) → reject → unavailable dish (chocolate cake) → reject | END, FAIL, "order retries exhausted" | PASS |
| TC2 | available order (3 burgers); cook fails once, serve fails once, remakes succeed | SUCCESS | PASS |
| TC3 | partial (4 pastas) → reject → available (2 dosas); cook fail→ok, serve fail→remake ok→serve fail (no cook retries left) | FAIL, "cook retries exhausted" | PASS |
