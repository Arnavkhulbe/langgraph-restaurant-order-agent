"""The restaurant menu."""

from __future__ import annotations

MENU: dict[str, int] = {
    "pizza": 5,
    "burger": 3,
    "pasta": 2,
    "salad": 4,
    "biryani": 6,
    "dosa": 8,
}


def menu_snapshot() -> dict[str, int]:
    """Return a copy of the menu (dish -> available quantity)."""
    return dict(MENU)


def lookup(dish: str) -> int:
    """Available quantity for a dish; 0 when the dish is not on the menu."""
    return MENU.get(dish.strip().lower(), 0)
