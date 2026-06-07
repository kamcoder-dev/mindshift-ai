from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Position = Literal[
    "strongly_for",
    "mostly_for",
    "mixed",
    "mostly_against",
    "strongly_against",
    "unclear",
]

Leaning = Literal[
    "strongly_for",
    "mostly_for",
    "mixed",
    "mostly_against",
    "strongly_against",
]


class ModeratorFrame(BaseModel):
    debate_question: str = Field(description="A clean neutral version of the user's topic.")
    neutral_framing: str = Field(description="A short framing that does not assume a side.")
    key_stakes: list[str] = Field(default_factory=list, description="The biggest trade-offs.")
    hidden_assumptions: list[str] = Field(default_factory=list)


class AgentOutput(BaseModel):
    position: Position
    confidence_before: int | None = Field(default=None, ge=0, le=100)
    confidence_after: int = Field(ge=0, le=100)
    main_argument: str
    best_counterargument_heard: str | None = None
    changed_mind: bool = False
    changed_mind_reason: str | None = None
    questions_for_other_agents: list[str] = Field(default_factory=list)
    evidence_used: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)


class ArgumentScoreOut(BaseModel):
    logic_score: int = Field(ge=0, le=100)
    evidence_score: int = Field(ge=0, le=100)
    empathy_score: int = Field(ge=0, le=100)
    clarity_score: int = Field(ge=0, le=100)
    notes: str


class CouncilVerdict(BaseModel):
    final_answer: str
    leaning: Leaning
    confidence: int = Field(ge=0, le=100)
    strongest_argument: str
    biggest_mind_shift: str
    remaining_uncertainties: list[str] = Field(default_factory=list)
    next_questions: list[str] = Field(default_factory=list)


class DebateTranscriptItem(BaseModel):
    round_number: int
    round_type: str
    agent_name: str
    role: str
    output: AgentOutput
