from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from mindshift.models.db import AgentMessage, Debate
from mindshift.models.views import DebateView
from mindshift.services.scoring_service import average_score

console = Console()


def render_debate(view: DebateView) -> None:
    console.print(
        Panel.fit(
            f"[bold]{view.debate.topic}[/bold]\n"
            f"ID: {view.debate.id}\n"
            f"Leaning: [bold]{view.debate.final_leaning}[/bold] | "
            f"Confidence: [bold]{view.debate.final_confidence}%[/bold]",
            title="MindShift Arena",
        )
    )

    table = Table(title="Agent transcript", show_lines=True)
    table.add_column("Agent", style="bold")
    table.add_column("Position")
    table.add_column("Confidence")
    table.add_column("Argument")
    table.add_column("Scores")

    for msg in view.messages:
        score = view.scores.get(msg.id)
        conf_before = str(msg.confidence_before) if msg.confidence_before is not None else "—"
        conf = f"{conf_before} → {msg.confidence_after}"
        score_text = "—"
        if score:
            score_text = (
                f"L:{score.logic_score} E:{score.evidence_score} "
                f"Em:{score.empathy_score} C:{score.clarity_score}"
            )
        table.add_row(msg.agent_name, msg.position, conf, msg.content, score_text)

    console.print(table)

    if view.mind_changes:
        changes_table = Table(title="Mind changes")
        changes_table.add_column("Agent", style="bold")
        changes_table.add_column("Shift")
        changes_table.add_column("Reason")
        for change in view.mind_changes:
            before = str(change.old_confidence) if change.old_confidence is not None else "—"
            changes_table.add_row(change.agent_name, f"{before} → {change.new_confidence}", change.reason)
        console.print(changes_table)

    console.print(
        Panel(Markdown(view.debate.final_verdict or "No final verdict stored."), title="Final verdict")
    )


def render_debate_list(debates: list[Debate]) -> None:
    table = Table(title="Saved debates")
    table.add_column("ID")
    table.add_column("Created")
    table.add_column("Topic")
    table.add_column("Leaning")
    table.add_column("Confidence")
    for debate in debates:
        table.add_row(
            debate.id,
            debate.created_at.strftime("%Y-%m-%d %H:%M"),
            debate.topic,
            debate.final_leaning or "—",
            str(debate.final_confidence or "—"),
        )
    console.print(table)


def render_autopsy(view: DebateView) -> None:
    strongest: tuple[AgentMessage, int] | None = None
    strongest_score = -1
    weakest: tuple[AgentMessage, int] | None = None
    weakest_score = 101

    for msg in view.messages:
        score = view.scores.get(msg.id)
        if not score:
            continue
        avg = average_score(score)
        if avg > strongest_score:
            strongest = (msg, avg)
            strongest_score = avg
        if avg < weakest_score:
            weakest = (msg, avg)
            weakest_score = avg

    console.print(Panel.fit(f"[bold]{view.debate.topic}[/bold]", title="Debate autopsy"))

    if strongest:
        msg, avg = strongest
        console.print(Panel(f"[bold]{msg.agent_name}[/bold] avg score {avg}\n{msg.content}", title="Strongest argument"))
    if weakest:
        msg, avg = weakest
        console.print(Panel(f"[bold]{msg.agent_name}[/bold] avg score {avg}\n{msg.content}", title="Weakest argument"))

    if view.mind_changes:
        biggest = max(
            view.mind_changes,
            key=lambda c: abs((c.new_confidence or 0) - (c.old_confidence or 0)),
        )
        console.print(
            Panel(
                f"[bold]{biggest.agent_name}[/bold]: "
                f"{biggest.old_confidence if biggest.old_confidence is not None else '—'} → {biggest.new_confidence}\n"
                f"{biggest.reason}",
                title="Biggest mind shift",
            )
        )

    console.print(Panel(view.debate.final_verdict or "No verdict stored.", title="Final verdict"))
