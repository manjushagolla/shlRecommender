"""
app/models.py
=============
Pydantic models matching the EXACT API schema from the assignment.
The evaluator is strict — do not change field names or types.
"""

from typing import Optional
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str        # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(
        ...,
        description="Full conversation history including the latest user message.",
        min_length=1,
    )


class Recommendation(BaseModel):
    name:      str   # exact name from catalog
    url:       str   # exact URL from catalog — never hallucinated
    test_type: str   # single letter: A/B/C/D/E/K/P/S


class ChatResponse(BaseModel):
    reply:               str                    # agent's natural language reply
    recommendations:     list[Recommendation]   # [] when clarifying/refusing, 1-10 when recommending
    end_of_conversation: bool                   # True only when agent considers task complete


class HealthResponse(BaseModel):
    status: str = "ok"