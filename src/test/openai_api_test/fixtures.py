"""Tools, schemas and prompts shared by the OpenAI API tests."""

from __future__ import annotations

from typing import Any


def user(text: str) -> list[dict[str, Any]]:
    """A one-message conversation."""
    return [{"role": "user", "content": text}]


def continued(text: str, assistant: Any) -> list[dict[str, Any]]:
    """A question and the start of its answer, for continue_final_message."""
    return user(text) + [{"role": "assistant", "content": assistant}]


def function_choice(name: str) -> dict[str, Any]:
    """tool_choice forcing one function."""
    return {"type": "function", "function": {"name": name}}


GET_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Paris"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

GET_TIME = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get the current local time in a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

WEATHER_TOOLS = [GET_WEATHER, GET_TIME]

# Every argument type the grammars handle: integer, number, boolean, string
# enum, array with typed items, and an optional field.
BOOK_FLIGHT = {
    "type": "function",
    "function": {
        "name": "book_flight",
        "description": "Book a flight",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "passengers": {"type": "integer"},
                "max_price": {"type": "number"},
                "refundable": {"type": "boolean"},
                "cabin": {"type": "string", "enum": ["economy", "business", "first"]},
                "meals": {"type": "array", "items": {"type": "string", "enum": ["veg", "vegan", "standard"]}},
            },
            "required": ["origin", "destination", "passengers", "refundable", "cabin"],
        },
    },
}

PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"},
    },
    "required": ["name", "age", "city"],
    "additionalProperties": False,
}

SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {"sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]}},
    "required": ["sentiment"],
    "additionalProperties": False,
}

# What Pydantic emits for a model with a nested list of models.
ORDER_SCHEMA = {
    "$defs": {
        "Item": {
            "title": "Item",
            "type": "object",
            "properties": {
                "name": {"title": "Name", "type": "string"},
                "qty": {"title": "Qty", "type": "integer"},
            },
            "required": ["name", "qty"],
            "additionalProperties": False,
        },
    },
    "title": "Order",
    "type": "object",
    "properties": {
        "customer": {"title": "Customer", "type": "string"},
        "items": {"title": "Items", "type": "array", "items": {"$ref": "#/$defs/Item"}},
    },
    "required": ["customer", "items"],
    "additionalProperties": False,
}

REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"const": "review"},
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "vegetarian": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
        "date": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
    },
    "required": ["kind", "score", "vegetarian", "date"],
    "additionalProperties": False,
}


def json_schema_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """response_format for a JSON schema."""
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
