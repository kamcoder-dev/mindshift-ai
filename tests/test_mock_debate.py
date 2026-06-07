from pathlib import Path

from mindshift.llm import MockLLM
from mindshift.services.debate_service import DebateService


def test_mock_debate_persists(tmp_path: Path) -> None:
    db_path = tmp_path / "mindshift-test.db"
    service = DebateService(db_path=db_path, llm=MockLLM())

    view = service.run_debate(topic="Will AI take over jobs", rounds=3, mode="serious")

    assert view.debate.topic == "Will AI take over jobs"
    assert view.debate.final_confidence == 68
    assert len(view.messages) == 7
    assert len(view.mind_changes) >= 1
