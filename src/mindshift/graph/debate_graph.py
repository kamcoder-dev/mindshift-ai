from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from mindshift.agents.prompts import MODERATOR_FRAME_PROMPT, VERDICT_PROMPT, Phase, agent_prompt
from mindshift.agents.roles import AGENTS, AgentSpec
from mindshift.llm import LLMProvider
from mindshift.models.schemas import AgentOutput, CouncilVerdict, ModeratorFrame


class DebateState(TypedDict, total=False):
    debate_id: str
    topic: str
    mode: str
    max_rounds: int
    frame: dict[str, Any]
    transcript: list[dict[str, Any]]
    verdict: dict[str, Any]


def create_debate_graph(llm: LLMProvider):

    graph = StateGraph(DebateState)

    graph.add_node("frame_topic", _frame_topic(llm))
    graph.add_node("opening_round", _opening_round(llm))
    graph.add_node("challenge_round", _challenge_round(llm))
    graph.add_node("update_round", _update_round(llm))
    graph.add_node("final_verdict", _final_verdict(llm))

    graph.add_edge(START, "frame_topic")
    graph.add_edge("frame_topic", "opening_round")
    graph.add_edge("opening_round", "challenge_round")
    graph.add_edge("challenge_round", "update_round")
    graph.add_edge("update_round", "final_verdict")
    graph.add_edge("final_verdict", END)

    return graph.compile()


def _frame_topic(llm: LLMProvider):
    def node(state: DebateState) -> dict[str, Any]:
        frame = llm.complete_structured(
            schema=ModeratorFrame,
            instructions=MODERATOR_FRAME_PROMPT,
            input_text=f"Topic: {state['topic']}\nMode: {state.get('mode', 'serious')}",
        )
        return {"frame": frame.model_dump(), "transcript": state.get("transcript", [])}

    return node


def _opening_round(llm: LLMProvider):
    def node(state: DebateState) -> dict[str, Any]:
        transcript = list(state.get("transcript", []))
        for key in ["advocate", "critic"]:
            spec = AGENTS[key]
            output = _run_agent(llm, spec, state, phase="opening", transcript=transcript)
            transcript.append(_transcript_item(1, "opening", spec, output))
        return {"transcript": transcript}

    return node


def _challenge_round(llm: LLMProvider):
    def node(state: DebateState) -> dict[str, Any]:
        transcript = list(state.get("transcript", []))
        for key in ["evidence", "logic"]:
            spec = AGENTS[key]
            output = _run_agent(llm, spec, state, phase="challenge", transcript=transcript)
            transcript.append(_transcript_item(2, "challenge", spec, output))
        return {"transcript": transcript}

    return node


def _update_round(llm: LLMProvider):
    def node(state: DebateState) -> dict[str, Any]:
        transcript = list(state.get("transcript", []))
        for key in ["advocate", "critic", "empathy"]:
            spec = AGENTS[key]
            output = _run_agent(llm, spec, state, phase="update", transcript=transcript)
            transcript.append(_transcript_item(3, "mind_update", spec, output))
        return {"transcript": transcript}

    return node


def _final_verdict(llm: LLMProvider):
    def node(state: DebateState) -> dict[str, Any]:
        verdict = llm.complete_structured(
            schema=CouncilVerdict,
            instructions=VERDICT_PROMPT,
            input_text=_state_context(state, state.get("transcript", [])),
        )
        return {"verdict": verdict.model_dump()}

    return node


def _run_agent(
    llm: LLMProvider,
    spec: AgentSpec,
    state: DebateState,
    *,
    phase: Phase,
    transcript: list[dict[str, Any]],
) -> AgentOutput:
    return llm.complete_structured(
        schema=AgentOutput,
        instructions=agent_prompt(spec, phase=phase),
        input_text=_state_context(state, transcript),
    )


def _transcript_item(round_number: int, round_type: str, spec: AgentSpec, output: AgentOutput) -> dict[str, Any]:
    return {
        "round_number": round_number,
        "round_type": round_type,
        "agent_name": spec.name,
        "role": spec.role,
        "output": output.model_dump(),
    }


def _state_context(state: DebateState, transcript: list[dict[str, Any]]) -> str:
    frame = state.get("frame") or {}
    lines = [
        f"Topic: {state['topic']}",
        f"Mode: {state.get('mode', 'serious')}",
        f"Framed question: {frame.get('debate_question', state['topic'])}",
        f"Neutral framing: {frame.get('neutral_framing', '')}",
        "",
        "Transcript so far:",
    ]
    if not transcript:
        lines.append("No previous transcript yet.")
    for item in transcript:
        out = item["output"]
        lines.append(
            f"R{item['round_number']} {item['agent_name']} [{item['round_type']}]: "
            f"position={out.get('position')}; confidence={out.get('confidence_after')}; "
            f"argument={out.get('main_argument')}"
        )
        if out.get("best_counterargument_heard"):
            lines.append(f"  best counterargument heard: {out['best_counterargument_heard']}")
        if out.get("flags"):
            lines.append(f"  flags: {', '.join(out['flags'])}")
    return "\n".join(lines)
