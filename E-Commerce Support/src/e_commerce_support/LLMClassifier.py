from crewai import LLM
import os

from pydantic import BaseModel, Field
from typing import Literal

class IntentResult(BaseModel):
    intent: Literal[
        "Product Search",
        "Order Status",
        "Returns & Refunds",
        "Complaint / Escalation",
        "FAQ / General"
    ] = Field(..., description="The single best-matching customer intent category for the query")


def classify_intent(query: str) -> str:
    llm = LLM(
        model="openrouter/openai/gpt-4o-mini",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        response_format=IntentResult
    )
    response = llm.call(
        messages=[
            {"role": "system", "content": (
                "Classify the customer's message into exactly one of these categories:\n"
                "- Product Search (asking about product info, price, availability)\n"
                "- Order Status (tracking, delivery, shipment questions)\n"
                "- Returns & Refunds (wants to return/exchange, refund status)\n"
                "- Complaint / Escalation (angry, frustrated, wants a human/manager)\n"
                "- FAQ / General (policy questions, general store info)\n\n"
                "Respond ONLY with the category name, exactly as written above."
            )},
            {"role": "user", "content": query}
        ]
    )
    if isinstance(response, IntentResult):
        result = response
    else:
        result = IntentResult.model_validate_json(response)

    return result.intent