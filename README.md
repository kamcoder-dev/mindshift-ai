# MindShift Arena

A CLI-only multi-agent debate system where agents argue, challenge each other, update confidence, and save the whole debate to SQLite.

No UI required. It runs from the terminal.

## What it does

Given a topic like:

```bash
mindshift debate "Will AI take over jobs" --mock
```

It runs a council:

- **Advocate** argues mostly for the idea.
- **Critic** argues mostly against it.
- **Evidence Scout** checks what evidence would matter.
- **Logic Checker** spots weak assumptions and false binaries.
- **Empathy Lens** checks stakeholder impact.
- **Moderator** writes the final verdict.

Then it stores:

- debate topic
- rounds
- agent messages
- confidence shifts
- mind-change reasons
- argument scores
- final verdict

## Install

Requires Python 3.12+.

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# Linux / macOS / Git Bash
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the project (editable install puts `mindshift` on your PATH):

```bash
pip install -e .
```

For development dependencies (pytest, ruff):

```bash
pip install -e ".[dev]"
```

## Configure the LLM

Copy the example env file:

```bash
cp .env.example .env
```

### Local LLM (LM Studio, Ollama, etc.)

Point at your OpenAI-compatible local server:

```bash
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio
OPENAI_MODEL=
```

Start LM Studio, load a chat/instruct model, and enable the local server. Leave
`OPENAI_MODEL` blank to auto-detect the loaded model.

### Cloud OpenAI

```bash
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4.1-mini
```

If you do not configure a real key or local server, use `--mock` to run deterministic fake agents.

## Commands

After activating your venv, run `mindshift` directly.

Create the database:

```bash
mindshift init
```

Run a debate in mock mode:

```bash
mindshift debate "Will AI take over jobs?" --mock
```

Run with an OpenAI model:

```bash
mindshift debate "Is AI art real art?" --model gpt-4.1-mini
```

List saved debates:

```bash
mindshift list
```

Show a saved debate:

```bash
mindshift show DEBATE_ID
```

Replay a saved debate:

```bash
mindshift replay DEBATE_ID
```

Export a debate to Markdown:

```bash
mindshift export DEBATE_ID -o exports/debate.md
```

Analyze a debate:

```bash
mindshift autopsy DEBATE_ID
```

## Project structure

```text
src/mindshift/
  main.py                  # Typer CLI
  config.py                # settings and env
  db.py                    # SQLite engine/session helpers
  llm.py                   # OpenAI + mock LLM providers
  agents/
    roles.py               # agent role definitions
    prompts.py             # prompt builders
  graph/
    debate_graph.py        # LangGraph StateGraph workflow
  models/
    db.py                  # SQLModel database tables
    schemas.py             # Pydantic structured outputs
    views.py               # DebateView dataclass for service layer
  services/
    debate_service.py      # run + persist debates
    export_service.py      # markdown export
    render_service.py      # Rich terminal rendering
    scoring_service.py     # deterministic argument scoring
```

## Run tests

```bash
python -m pytest -vv

```
