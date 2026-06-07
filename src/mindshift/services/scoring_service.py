from __future__ import annotations

from typing import Protocol

from mindshift.models.schemas import AgentOutput, ArgumentScoreOut


class ScoredArgument(Protocol):
    logic_score: int
    evidence_score: int
    empathy_score: int
    clarity_score: int


def score_argument(output: AgentOutput) -> ArgumentScoreOut:
    text = output.main_argument.lower()
    flags = {flag.lower() for flag in output.flags}

    logic = 58
    if "because" in text or "depends" in text or "condition" in text:
        logic += 12
    if "false dilemma" in flags or "overgeneralisation" in flags:
        logic -= 8

    evidence = 45 + min(35, len(output.evidence_used) * 12)
    if "missing external citations" in flags:
        evidence -= 10

    empathy = 50
    if any(word in text for word in ["people", "affected", "fair", "harm", "benefit"]):
        empathy += 18

    clarity = 52
    if 120 <= len(output.main_argument) <= 500:
        clarity += 18
    if len(output.questions_for_other_agents) > 0:
        clarity += 5

    logic = _clamp(logic)
    evidence = _clamp(evidence)
    empathy = _clamp(empathy)
    clarity = _clamp(clarity)

    return ArgumentScoreOut(
        logic_score=logic,
        evidence_score=evidence,
        empathy_score=empathy,
        clarity_score=clarity,
        notes="Deterministic rubric based on reasoning signals, evidence references, empathy cues, and clarity.",
    )


def average_score(score: ScoredArgument) -> int:
    return int((score.logic_score + score.evidence_score + score.empathy_score + score.clarity_score) / 4)


def _clamp(value: int) -> int:
    return max(0, min(100, value))
