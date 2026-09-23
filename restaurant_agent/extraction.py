"""Order extraction: dish + quantity from free-form user text.

Primary path : a real LLM with structured output (OpenAI or Gemini, picked
               automatically from environment variables).
Fallback path : a deterministic regex/keyword parser so the agent (and the
                TC1/TC2/TC3 simulations) also run with no API key configured.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from pydantic import BaseModel, Field

from . import menu as menu_mod


class OrderExtraction(BaseModel):
    """Structured result of parsing one user utterance."""

    is_order: bool = Field(description="True when the user is trying to order food")
    dish_name: str = Field(default="", description="Canonical dish name, empty when not an order")
    required_quantity: int = Field(default=0, description="Requested quantity (>=1)")


# --------------------------------------------------------------------------- #
# LLM provider factory
# --------------------------------------------------------------------------- #

def load_llm():
    """Return (llm, provider_name) or (None, 'offline') based on env vars."""
    try:  # pick up a local .env if the user created one
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:  # pragma: no cover
        pass
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI

            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            return ChatOpenAI(model=model, temperature=0), f"openai:{model}"
        except Exception as exc:  # pragma: no cover
            print(f"[warn] OpenAI unavailable ({exc}); falling back to offline parser")
    if os.getenv("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            return ChatGoogleGenerativeAI(model=model, temperature=0), f"gemini:{model}"
        except Exception as exc:  # pragma: no cover
            print(f"[warn] Gemini unavailable ({exc}); falling back to offline parser")
    return None, "offline"


def make_extractor(llm):
    """Wrap an LLM into a callable text -> OrderExtraction (or None for offline)."""
    if llm is None:
        return None
    structured = llm.with_structured_output(OrderExtraction)
    dishes = ", ".join(sorted(menu_mod.MENU))

    def extract(text: str) -> OrderExtraction:
        try:
            return structured.invoke(
                [
                    (
                        "system",
                        "You are the order-parsing module of a restaurant ordering agent. "
                        f"The menu contains: {dishes}. Extract the single dish and quantity the "
                        "user wants. Set is_order=False if the message is not a food order. "
                        "Default quantity to 1 when the user omits it.",
                    ),
                    ("human", text),
                ]
            )
        except Exception as exc:  # pragma: no cover - network failures
            print(f"[warn] LLM extraction failed ({exc}); using offline parser")
            return offline_extract(text)

    return extract


# --------------------------------------------------------------------------- #
# Offline fallback parser (deterministic, no network)
# --------------------------------------------------------------------------- #

WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}

FILLER_WORDS = {
    "i", "id", "i'd", "ill", "i'll", "we", "want", "wants", "would", "like",
    "can", "could", "get", "give", "have", "has", "need", "order", "ordering",
    "please", "me", "us", "let", "some", "of", "for", "to", "the", "and",
    "take", "grab", "buy", "hey", "hi", "hello",
}

QUESTION_STARTERS = {"what", "how", "why", "who", "when", "where", "which", "whats"}

# Foods a user may order that are NOT on today's menu (they must still be
# treated as orders so order_confirm can answer UNAVAILABLE). Anything that
# matches no known food is treated as chit-chat, not an order.
FOOD_WORDS = {
    "cake", "chocolate cake", "ice cream", "coffee", "tea", "sandwich",
    "rice", "fried rice", "chicken", "fish", "soup", "noodles", "fries",
    "naan", "chapati", "roti", "paratha", "idli", "vada", "samosa",
    "momo", "momos", "curry", "paneer", "pancakes", "waffles", "steak",
    "sushi", "burrito", "taco", "wrap", "lasagna", "risotto", "poha",
}


def _is_known_food(dish: str) -> bool:
    """True when the extracted dish is on the menu or a known food phrase."""
    candidates = {dish}
    if dish:
        candidates.add(dish.split()[-1])  # last word, e.g. 'chocolate cake' -> 'cake'
        candidates |= {c.rstrip("s") for c in candidates}  # naive de-pluralize
    return any(c and (c in menu_mod.MENU or c in FOOD_WORDS) for c in candidates)


def _canonical_dish(raw: str) -> str:
    """Map a raw dish phrase to a menu key when possible ('2 pizzas' -> pizza)."""
    raw = raw.strip().strip(".,!?").lower()
    if raw in menu_mod.MENU:
        return raw
    if raw.endswith("s") and raw[:-1] in menu_mod.MENU:
        return raw[:-1]
    # longest menu key contained in the phrase ('chicken pizza' -> 'pizza')
    best = ""
    for key in menu_mod.MENU:
        if key in raw and len(key) > len(best):
            best = key
    return best or raw


def offline_extract(text: str) -> OrderExtraction:
    """Deterministic order parser used when no LLM provider is configured."""
    t = text.lower().strip()
    tokens = re.findall(r"[a-z']+|\d+", t)

    # Reject obvious non-order chatter ("what's the weather?")
    first = tokens[0] if tokens else ""
    looks_like_question = first in QUESTION_STARTERS or "weather" in t or "joke" in t

    quantity: Optional[int] = None
    dish_tokens: list[str] = []

    # "2 pizzas", "2 x pizza", "pizza x 2"
    m = re.search(r"(\d+)\s*x\s*([a-z].*)$", t) or re.search(r"^(\d+)\s+([a-z].*)$", t)
    if m:
        quantity = int(m.group(1))
        dish_tokens = m.group(2).split()
    else:
        m = re.search(r"([a-z ]+?)\s*x\s*(\d+)$", t)
        if m:
            dish_tokens = m.group(1).split()
            quantity = int(m.group(2))

    if quantity is None:
        for tok in tokens:
            if tok in WORD_NUMBERS:
                quantity = WORD_NUMBERS[tok]
                break
            if tok.isdigit():  # "I want 3 burgers" - digits outside "N dish" patterns
                quantity = int(tok)
                break

    if not dish_tokens:
        dish_tokens = [
            tok for tok in tokens
            if tok not in WORD_NUMBERS
            and tok not in FILLER_WORDS
            and not tok.isdigit()
            and tok != "x"
        ]
    elif quantity is not None:
        # drop a leading word-number that was the quantity ("two pizzas")
        dish_tokens = [tok for tok in dish_tokens if tok not in WORD_NUMBERS or WORD_NUMBERS[tok] != quantity]

    dish = _canonical_dish(" ".join(dish_tokens))

    if quantity is None:
        quantity = 1
    quantity = max(1, quantity)

    if looks_like_question and dish not in menu_mod.MENU:
        return OrderExtraction(is_order=False)
    if not dish or not _is_known_food(dish):
        # Not a recognizable food -> chit-chat, not an order (the LLM path
        # makes this call with real language understanding instead).
        return OrderExtraction(is_order=False)
    return OrderExtraction(is_order=True, dish_name=dish, required_quantity=quantity)
