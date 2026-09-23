"""Kitchen simulation: cooking and serving with probabilistic outcomes.

- cook: 60% success / 40% failure per attempt, max 2 attempts.
- serve: 60% success / 40% failure per attempt, max 2 attempts.

Both functions are deterministic under `random.seed(...)`, which lets the
TC1/TC2/TC3 simulations inject exact pass/fail sequences (see tests/).
"""

from __future__ import annotations

import random

COOK_SUCCESS_RATE = 0.60
SERVE_SUCCESS_RATE = 0.60


def cook_once() -> bool:
    """One cooking attempt: 60% success."""
    return random.random() < COOK_SUCCESS_RATE


def serve_once() -> bool:
    """One serving attempt: simulated pass/fail (60% success)."""
    return random.random() < SERVE_SUCCESS_RATE
