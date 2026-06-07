from __future__ import annotations

from dataclasses import dataclass

from mindshift.models.db import AgentMessage, ArgumentScore, Debate, DebateRound, MindChange


@dataclass
class DebateView:
    debate: Debate
    rounds: list[DebateRound]
    messages: list[AgentMessage]
    scores: dict[str, ArgumentScore]
    mind_changes: list[MindChange]
