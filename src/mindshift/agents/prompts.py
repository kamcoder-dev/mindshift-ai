from __future__ import annotations

from typing import Literal

from mindshift.agents.roles import AgentSpec

Phase = Literal["opening", "challenge", "update"]

_BASE_AGENT_RULES = """
You are inside a multi-agent debate system called MindShift Arena.
Return only the requested structured object.
Be concise, specific, and willing to change your mind.
Never claim certainty you do not have.
Every confidence score must be an integer from 0 to 100.
""".strip()

_PHASE_RULES: dict[Phase, str] = {
    "opening": "No prior transcript. Set confidence_before to null.",
    "challenge": "Challenge prior arguments. Name the weakest claim you see.",
    "update": "Set confidence_before from your last turn. Explain any mind shift in changed_mind_reason.",
}

MODERATOR_FRAME_PROMPT = """
You are the Moderator.
Frame the user's topic as a fair debate question.
Do not answer the topic yet.
Identify stakes and hidden assumptions.

Return fields:
- debate_question, neutral_framing
- key_stakes, hidden_assumptions
""".strip()

VERDICT_PROMPT = """
You are the final Moderator.
Synthesize the whole debate into a balanced council verdict.
Do not pretend both sides are equal if one side was stronger.

Return fields:
- final_answer, leaning, confidence
- strongest_argument, biggest_mind_shift
- remaining_uncertainties, next_questions
""".strip()


def agent_prompt(spec: AgentSpec, *, phase: Phase) -> str:
    return f"""{_BASE_AGENT_RULES}

Agent name: {spec.name}
Agent role: {spec.role}
Stance rule: {spec.stance_instruction}
Phase: {phase}
Phase rules: {_PHASE_RULES[phase]}

Return fields:
- position, main_argument, confidence_after
- confidence_before (null unless update phase)
- best_counterargument_heard (null if none yet)
- changed_mind, changed_mind_reason (only if changed_mind is true)
- questions_for_other_agents, evidence_used
- flags (e.g. weak_logic, missing_evidence, missing_definition, risk)
""".strip()
