from __future__ import annotations

from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(UTC)


class Debate(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    topic: str
    mode: str = "serious"
    rounds_requested: int = 3
    created_at: datetime = Field(default_factory=utc_now)
    final_verdict: str | None = None
    final_leaning: str | None = None
    final_confidence: int | None = None


class DebateRound(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    debate_id: str = Field(foreign_key="debate.id", index=True)
    round_number: int
    round_type: str
    created_at: datetime = Field(default_factory=utc_now)


class AgentMessage(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    debate_id: str = Field(foreign_key="debate.id", index=True)
    round_id: str = Field(foreign_key="debateround.id", index=True)
    agent_name: str = Field(index=True)
    role: str
    position: str
    confidence_before: Optional[int] = None
    confidence_after: int
    content: str
    best_counterargument_heard: str | None = None
    changed_mind: bool = False
    changed_mind_reason: str | None = None
    questions_json: str = "[]"
    evidence_json: str = "[]"
    flags_json: str = "[]"
    created_at: datetime = Field(default_factory=utc_now)


class MindChange(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    debate_id: str = Field(foreign_key="debate.id", index=True)
    message_id: str = Field(foreign_key="agentmessage.id", index=True)
    agent_name: str = Field(index=True)
    old_confidence: int | None = None
    new_confidence: int
    reason: str
    created_at: datetime = Field(default_factory=utc_now)


class ArgumentScore(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    message_id: str = Field(foreign_key="agentmessage.id", index=True)
    logic_score: int
    evidence_score: int
    empathy_score: int
    clarity_score: int
    notes: str
