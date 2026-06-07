from __future__ import annotations

import hashlib
import json
import os
from typing import Protocol, TypeVar

from openai import OpenAI
from pydantic import BaseModel

from mindshift.models.schemas import AgentOutput, CouncilVerdict, ModeratorFrame

T = TypeVar("T", bound=BaseModel)

LOCAL_API_KEY = "lm-studio"
PLACEHOLDER_KEYS = {"sk-your-key-here", "your-api-key", "changeme"}


class LLMProvider(Protocol):
    def complete_structured(self, *, schema: type[T], instructions: str, input_text: str) -> T:
        """Return a Pydantic object from an LLM or mock provider."""


class OpenAILLM:
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.local = base_url is not None
        client_kwargs: dict[str, str] = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key
        self.client = OpenAI(**client_kwargs)

    def complete_structured(self, *, schema: type[T], instructions: str, input_text: str) -> T:
        if self.local:
            return self._complete_structured_local(
                schema=schema,
                instructions=instructions,
                input_text=input_text,
            )
        response = self.client.responses.parse(
            model=self.model,
            instructions=instructions,
            input=input_text,
            text_format=schema,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("OpenAI returned no parsed structured output.")
        return parsed

    def _complete_structured_local(
        self,
        *,
        schema: type[T],
        instructions: str,
        input_text: str,
    ) -> T:
        schema_def = schema.model_json_schema()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{instructions}\n\n"
                        "Respond with valid JSON only. No markdown fences."
                    ),
                },
                {"role": "user", "content": input_text},
            ],
            temperature=0.7,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema_def,
                },
            },
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Local LLM returned an empty response.")
        return schema.model_validate_json(content)


class MockLLM:
    """Deterministic fake LLM for demos, tests, and no-key development."""

    def _stable_confidence(self, text: str, base: int = 62) -> int:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        bump = int(digest[:2], 16) % 23
        return max(5, min(95, base + bump))

    def complete_structured(self, *, schema: type[T], instructions: str, input_text: str) -> T:
        topic = _extract_topic(input_text)
        role = _extract_role(instructions)

        if schema is ModeratorFrame:
            return schema(
                debate_question=f"What is the strongest balanced view on: {topic}?",
                neutral_framing=(
                    "The council should compare benefits, harms, evidence quality, edge cases, "
                    "and what would change its mind."
                ),
                key_stakes=["practical impact", "fairness", "long-term consequences"],
                hidden_assumptions=[
                    "That the topic has one clean answer",
                    "That all affected groups experience the policy the same way",
                ],
            )

        if schema is CouncilVerdict:
            return schema(
                final_answer=(
                    f"The council lands on a mixed but usable answer for '{topic}': the best choice "
                    "depends on safeguards, context, and measurable outcomes rather than vibes."
                ),
                leaning="mixed",
                confidence=68,
                strongest_argument=(
                    "The strongest argument was the one that named a concrete trade-off instead of "
                    "pretending the issue has zero downsides."
                ),
                biggest_mind_shift=(
                    "The Critic softened after the Evidence Scout separated realistic risks from "
                    "dramatic worst-case claims."
                ),
                remaining_uncertainties=[
                    "The quality of real-world evidence is uneven.",
                    "Different groups may be affected differently.",
                ],
                next_questions=[
                    "What measurable result would prove this worked?",
                    "Who pays the cost if the council is wrong?",
                ],
            )

        if schema is AgentOutput:
            return schema(**_mock_agent_output(role, topic, input_text))

        raise TypeError(f"MockLLM does not know how to return {schema.__name__}")


def build_llm(*, model: str, mock: bool = False) -> LLMProvider:
    if mock:
        return MockLLM()

    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    model, base_url = _resolve_model_and_base_url(model, base_url)

    if base_url:
        resolved_model = model or _detect_local_model(base_url, api_key or LOCAL_API_KEY)
        return OpenAILLM(
            model=resolved_model,
            base_url=base_url,
            api_key=api_key or LOCAL_API_KEY,
        )

    if not api_key or _is_placeholder_key(api_key):
        return MockLLM()

    return OpenAILLM(model=model, api_key=api_key)


def _resolve_model_and_base_url(model: str, base_url: str | None) -> tuple[str, str | None]:
    if model.startswith(("http://", "https://")):
        normalized = model.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized = f"{normalized}/v1"
        return "", normalized
    return model, base_url


def _is_placeholder_key(key: str) -> bool:
    lowered = key.strip().lower()
    return lowered in PLACEHOLDER_KEYS or lowered.startswith("sk-your-")


def _detect_local_model(base_url: str, api_key: str) -> str:
    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        models = client.models.list()
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach local LLM at {base_url}. "
            "Start LM Studio (or your local server) and load a model first."
        ) from exc

    if not models.data:
        raise RuntimeError(
            f"No models found at {base_url}. Load a chat/instruct model in LM Studio first."
        )

    return _pick_chat_model([entry.id for entry in models.data])


def _pick_chat_model(model_ids: list[str]) -> str:
    skip = ("embed", "embedding", "vision", "-vl", "_vl", "/vl")
    usable = [model_id for model_id in model_ids if not any(token in model_id.lower() for token in skip)]
    if not usable:
        return model_ids[0]
    for hint in ("instruct", "mythomax", "samantha", "chat", "llama", "mistral", "qwen"):
        for model_id in usable:
            if hint in model_id.lower():
                return model_id
    return usable[0]


def _extract_topic(input_text: str) -> str:
    marker = "Topic:"
    if marker in input_text:
        return input_text.split(marker, 1)[1].splitlines()[0].strip()
    return input_text.strip().splitlines()[0][:120]


def _extract_role(instructions: str) -> str:
    lowered = instructions.lower()
    if "advocate" in lowered:
        return "Advocate"
    if "critic" in lowered:
        return "Critic"
    if "evidence" in lowered:
        return "Evidence Scout"
    if "logic" in lowered:
        return "Logic Checker"
    if "empathy" in lowered:
        return "Empathy Lens"
    return "Agent"


def _mock_agent_output(role: str, topic: str, input_text: str) -> dict:
    previous = 82 if "update" in input_text.lower() and role == "Advocate" else None
    if role == "Advocate":
        return {
            "position": "mostly_for",
            "confidence_before": previous,
            "confidence_after": 66 if previous else 78,
            "main_argument": (
                f"For '{topic}', the strongest pro case is that a clear decision can reduce confusion "
                "and create predictable standards, as long as exceptions are handled openly."
            ),
            "best_counterargument_heard": "A rule can become unfair if it ignores edge cases.",
            "changed_mind": bool(previous),
            "changed_mind_reason": "The council exposed that the pro side needs safeguards, not just enthusiasm.",
            "questions_for_other_agents": ["What exception would break the policy?"],
            "evidence_used": ["mock: practical consistency argument"],
            "flags": [],
        }
    if role == "Critic":
        return {
            "position": "mostly_against",
            "confidence_before": 84 if "update" in input_text.lower() else None,
            "confidence_after": 58 if "update" in input_text.lower() else 81,
            "main_argument": (
                f"Against '{topic}', the strongest concern is that neat solutions often hide costs, "
                "especially for people with less flexibility or power."
            ),
            "best_counterargument_heard": "Consistency can matter when inconsistency creates stress or unfairness.",
            "changed_mind": "update" in input_text.lower(),
            "changed_mind_reason": "Evidence separated realistic harms from exaggerated slippery-slope claims.",
            "questions_for_other_agents": ["Who is most likely to be harmed by this?"],
            "evidence_used": ["mock: unintended-consequences argument"],
            "flags": ["watch for overgeneralisation"],
        }
    if role == "Evidence Scout":
        return {
            "position": "mixed",
            "confidence_before": None,
            "confidence_after": 64,
            "main_argument": (
                "The evidence should be treated as directional, not absolute: look for repeated patterns, "
                "baseline comparisons, and whether outcomes are measured fairly."
            ),
            "best_counterargument_heard": "Anecdotes can reveal harms that broad averages miss.",
            "changed_mind": False,
            "changed_mind_reason": None,
            "questions_for_other_agents": ["What data would actually settle this?"],
            "evidence_used": ["mock: evidence-quality checklist"],
            "flags": ["missing external citations in mock mode"],
        }
    if role == "Logic Checker":
        return {
            "position": "unclear",
            "confidence_before": None,
            "confidence_after": 72,
            "main_argument": (
                "Both sides need to avoid false binaries. The useful question is not simply yes/no, "
                "but under what conditions the answer changes."
            ),
            "best_counterargument_heard": "Some decisions still require a practical yes/no rule.",
            "changed_mind": False,
            "changed_mind_reason": None,
            "questions_for_other_agents": ["Are you arguing against the idea or a bad implementation?"],
            "evidence_used": [],
            "flags": ["possible false dilemma", "define success metric"],
        }
    return {
        "position": "mixed",
        "confidence_before": 50 if "update" in input_text.lower() else None,
        "confidence_after": 70 if "update" in input_text.lower() else 55,
        "main_argument": (
            "The people affected will not experience this equally, so the final verdict should include "
            "who benefits, who absorbs the inconvenience, and who needs an exception."
        ),
        "best_counterargument_heard": "Too many exceptions can make a rule impossible to enforce.",
        "changed_mind": "update" in input_text.lower(),
        "changed_mind_reason": "The debate showed empathy must be paired with workable enforcement.",
        "questions_for_other_agents": ["Whose experience is missing from this debate?"],
        "evidence_used": ["mock: stakeholder impact reasoning"],
        "flags": [],
    }
