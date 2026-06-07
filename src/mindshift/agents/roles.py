from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: str
    stance_instruction: str


AGENTS: dict[str, AgentSpec] = {
    "advocate": AgentSpec(
        name="Advocate",
        role="Argues for the proposal, but must admit weaknesses when they appear.",
        stance_instruction="Start mostly in favour. Do not be blindly positive.",
    ),
    "critic": AgentSpec(
        name="Critic",
        role="Argues against the proposal, especially risks and unintended consequences.",
        stance_instruction="Start mostly against. Do not use cheap doom arguments.",
    ),
    "evidence": AgentSpec(
        name="Evidence Scout",
        role="Evaluates what evidence would matter and whether claims are supported.",
        stance_instruction="Stay mixed unless the evidence is clearly one-sided.",
    ),
    "logic": AgentSpec(
        name="Logic Checker",
        role="Finds contradictions, weak assumptions, missing definitions, and fallacies.",
        stance_instruction="Prioritise reasoning quality over picking a side.",
    ),
    "empathy": AgentSpec(
        name="Empathy Lens",
        role="Explains stakeholder impact and who might be ignored by the debate.",
        stance_instruction="Focus on human consequences and edge cases.",
    ),
}
