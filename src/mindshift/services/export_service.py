from __future__ import annotations

import json
from pathlib import Path

from mindshift.models.views import DebateView


def debate_to_markdown(view: DebateView) -> str:
    lines = [
        f"# MindShift Debate: {view.debate.topic}",
        "",
        f"- Debate ID: `{view.debate.id}`",
        f"- Mode: `{view.debate.mode}`",
        f"- Final leaning: **{view.debate.final_leaning or 'unknown'}**",
        f"- Final confidence: **{view.debate.final_confidence or 0}%**",
        "",
        "## Final verdict",
        "",
        view.debate.final_verdict or "No verdict stored.",
        "",
        "## Transcript",
        "",
    ]

    for msg in view.messages:
        score = view.scores.get(msg.id)
        lines.extend(
            [
                f"### {msg.agent_name}",
                "",
                f"**Position:** `{msg.position}`  ",
                f"**Confidence:** {msg.confidence_before if msg.confidence_before is not None else 'n/a'} → {msg.confidence_after}",
                "",
                msg.content,
                "",
            ]
        )
        if msg.best_counterargument_heard:
            lines.extend([f"**Best counterargument heard:** {msg.best_counterargument_heard}", ""])
        questions = json.loads(msg.questions_json or "[]")
        if questions:
            lines.extend(["**Questions:**", *[f"- {q}" for q in questions], ""])
        if score:
            lines.extend(
                [
                    f"**Scores:** logic {score.logic_score}, evidence {score.evidence_score}, "
                    f"empathy {score.empathy_score}, clarity {score.clarity_score}",
                    "",
                ]
            )

    if view.mind_changes:
        lines.extend(["## Mind changes", ""])
        for change in view.mind_changes:
            old = change.old_confidence if change.old_confidence is not None else "n/a"
            lines.append(f"- **{change.agent_name}:** {old} → {change.new_confidence}. {change.reason}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def export_markdown(view: DebateView, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(debate_to_markdown(view), encoding="utf-8")
    return output_path
