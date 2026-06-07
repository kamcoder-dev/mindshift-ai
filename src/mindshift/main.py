from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from mindshift.config import Settings
from mindshift.db import create_db_and_tables, make_engine
from mindshift.llm import build_llm
from mindshift.services.debate_service import DebateService
from mindshift.services.export_service import export_markdown
from mindshift.services.render_service import render_autopsy, render_debate, render_debate_list

app = typer.Typer(
    help="MindShift Arena: a CLI multi-agent debate room where agents can change their minds.",
    no_args_is_help=True,
)
console = Console()
settings = Settings()


@app.command()
def init(
    db_path: Path = typer.Option(settings.db_path, help="SQLite database path."),
) -> None:
    engine = make_engine(db_path)
    create_db_and_tables(engine)
    console.print(f"[green]Database ready:[/green] {db_path}")


@app.command()
def debate(
    topic: str = typer.Argument(..., help="Debate topic or question."),
    rounds: int = typer.Option(3, min=3, max=3, help="MVP currently uses exactly 3 rounds."),
    mode: str = typer.Option("serious", help="Debate tone, e.g. serious, academic, chaotic, comic-book."),
    db_path: Path = typer.Option(settings.db_path, help="SQLite database path."),
    model: str = typer.Option(settings.openai_model, help="OpenAI model name."),
    mock: bool = typer.Option(False, help="Use deterministic mock agents instead of calling an LLM."),
) -> None:
    llm = build_llm(model=model, mock=mock)
    if mock:
        console.print("[yellow]Running in mock mode.[/yellow]")
    elif getattr(llm, "local", False):
        console.print(f"[cyan]Using local LLM:[/cyan] {llm.model}")
    service = DebateService(db_path=db_path, llm=llm)
    view = service.run_debate(topic=topic, rounds=rounds, mode=mode)
    render_debate(view)


@app.command("list")
def list_debates(
    limit: int = typer.Option(20, min=1, max=100),
    db_path: Path = typer.Option(settings.db_path, help="SQLite database path."),
) -> None:
    service = DebateService(db_path=db_path)
    render_debate_list(service.list_debates(limit=limit))


@app.command()
def show(
    debate_id: str = typer.Argument(..., help="Saved debate ID."),
    db_path: Path = typer.Option(settings.db_path, help="SQLite database path."),
) -> None:
    service = DebateService(db_path=db_path)
    render_debate(service.get_debate(debate_id))


@app.command()
def replay(
    debate_id: str = typer.Argument(..., help="Saved debate ID."),
    db_path: Path = typer.Option(settings.db_path, help="SQLite database path."),
) -> None:
    service = DebateService(db_path=db_path)
    render_debate(service.get_debate(debate_id))


@app.command()
def export(
    debate_id: str = typer.Argument(..., help="Saved debate ID."),
    output: Path = typer.Option(Path("debate.md"), "--output", "-o", help="Markdown output path."),
    db_path: Path = typer.Option(settings.db_path, help="SQLite database path."),
) -> None:
    service = DebateService(db_path=db_path)
    path = export_markdown(service.get_debate(debate_id), output)
    console.print(f"[green]Exported:[/green] {path}")


@app.command()
def autopsy(
    debate_id: str = typer.Argument(..., help="Saved debate ID."),
    db_path: Path = typer.Option(settings.db_path, help="SQLite database path."),
) -> None:
    service = DebateService(db_path=db_path)
    render_autopsy(service.get_debate(debate_id))


if __name__ == "__main__":
    app()
