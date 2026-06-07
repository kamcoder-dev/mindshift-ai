from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlmodel import Session, select

from mindshift.db import create_db_and_tables, make_engine
from mindshift.graph.debate_graph import create_debate_graph
from mindshift.llm import LLMProvider
from mindshift.models.db import AgentMessage, ArgumentScore, Debate, DebateRound, MindChange
from mindshift.models.schemas import AgentOutput
from mindshift.models.views import DebateView
from mindshift.services.scoring_service import score_argument


class DebateService:
    def __init__(self, *, db_path: Path, llm: LLMProvider | None = None) -> None:
        self.engine = make_engine(db_path)
        create_db_and_tables(self.engine)
        self.llm = llm

    def run_debate(self, *, topic: str, rounds: int = 3, mode: str = "serious") -> DebateView:
        if self.llm is None:
            raise RuntimeError("An LLM provider is required to run a debate.")
        debate_id = str(uuid4())
        graph = create_debate_graph(self.llm)
        state = graph.invoke(
            {
                "debate_id": debate_id,
                "topic": topic,
                "mode": mode,
                "max_rounds": rounds,
                "transcript": [],
            }
        )
        self._persist_state(debate_id=debate_id, topic=topic, rounds=rounds, mode=mode, state=state)
        return self.get_debate(debate_id)

    def get_debate(self, debate_id: str) -> DebateView:
        with Session(self.engine) as session:
            debate = session.get(Debate, debate_id)
            if debate is None:
                raise LookupError(f"No debate found with id {debate_id}")

            rounds = session.exec(
                select(DebateRound)
                .where(DebateRound.debate_id == debate_id)
                .order_by(DebateRound.round_number, DebateRound.created_at)
            ).all()

            messages = session.exec(
                select(AgentMessage)
                .where(AgentMessage.debate_id == debate_id)
                .order_by(AgentMessage.created_at)
            ).all()

            score_by_message: dict[str, ArgumentScore] = {}
            if messages:
                message_ids = [message.id for message in messages]
                scores = session.exec(
                    select(ArgumentScore).where(ArgumentScore.message_id.in_(message_ids))
                ).all()
                score_by_message = {score.message_id: score for score in scores}

            mind_changes = session.exec(
                select(MindChange)
                .where(MindChange.debate_id == debate_id)
                .order_by(MindChange.created_at)
            ).all()

            return DebateView(
                debate=debate,
                rounds=rounds,
                messages=messages,
                scores=score_by_message,
                mind_changes=mind_changes,
            )

    def list_debates(self, *, limit: int = 20) -> list[Debate]:
        with Session(self.engine) as session:
            return session.exec(select(Debate).order_by(Debate.created_at.desc()).limit(limit)).all()

    def _persist_state(
        self,
        *,
        debate_id: str,
        topic: str,
        rounds: int,
        mode: str,
        state: dict[str, Any],
    ) -> None:
        verdict = state.get("verdict", {})
        transcript = state.get("transcript", [])
        with Session(self.engine) as session:
            debate = Debate(
                id=debate_id,
                topic=topic,
                mode=mode,
                rounds_requested=rounds,
                final_verdict=verdict.get("final_answer"),
                final_leaning=verdict.get("leaning"),
                final_confidence=verdict.get("confidence"),
            )
            session.add(debate)
            session.flush()

            round_cache: dict[tuple[int, str], DebateRound] = {}
            for item in transcript:
                key = (int(item["round_number"]), str(item["round_type"]))
                if key not in round_cache:
                    debate_round = DebateRound(
                        debate_id=debate_id,
                        round_number=key[0],
                        round_type=key[1],
                    )
                    session.add(debate_round)
                    session.flush()
                    round_cache[key] = debate_round

                output = AgentOutput.model_validate(item["output"])
                message = AgentMessage(
                    debate_id=debate_id,
                    round_id=round_cache[key].id,
                    agent_name=item["agent_name"],
                    role=item["role"],
                    position=output.position,
                    confidence_before=output.confidence_before,
                    confidence_after=output.confidence_after,
                    content=output.main_argument,
                    best_counterargument_heard=output.best_counterargument_heard,
                    changed_mind=output.changed_mind,
                    changed_mind_reason=output.changed_mind_reason,
                    questions_json=json.dumps(output.questions_for_other_agents),
                    evidence_json=json.dumps(output.evidence_used),
                    flags_json=json.dumps(output.flags),
                )
                session.add(message)
                session.flush()

                score = score_argument(output)
                session.add(
                    ArgumentScore(
                        message_id=message.id,
                        logic_score=score.logic_score,
                        evidence_score=score.evidence_score,
                        empathy_score=score.empathy_score,
                        clarity_score=score.clarity_score,
                        notes=score.notes,
                    )
                )

                if output.changed_mind:
                    session.add(
                        MindChange(
                            debate_id=debate_id,
                            message_id=message.id,
                            agent_name=item["agent_name"],
                            old_confidence=output.confidence_before,
                            new_confidence=output.confidence_after,
                            reason=output.changed_mind_reason or "No reason supplied.",
                        )
                    )

            session.commit()
