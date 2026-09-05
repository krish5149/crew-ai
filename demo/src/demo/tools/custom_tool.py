from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class WordCountInput(BaseModel):
    """Input schema for WordCountTool."""

    text: str = Field(..., description="The text to measure.")


class WordCountTool(BaseTool):
    name: str = "word_count"
    description: str = (
        "Count the words in a piece of text. Call this when you need to check "
        "a draft against a length requirement before returning it, rather than "
        "estimating the length yourself."
    )
    args_schema: Type[BaseModel] = WordCountInput

    def _run(self, text: str) -> str:
        return f"{len(text.split())} words"