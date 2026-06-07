from mindshift.models.schemas import AgentOutput
from mindshift.services.scoring_service import score_argument


def test_score_argument_returns_bounded_scores() -> None:
    output = AgentOutput(
        position="mixed",
        confidence_before=None,
        confidence_after=64,
        main_argument="This depends on context because people are affected differently by the trade-off.",
        evidence_used=["example source"],
        questions_for_other_agents=["What would change your mind?"],
    )

    score = score_argument(output)

    assert 0 <= score.logic_score <= 100
    assert 0 <= score.evidence_score <= 100
    assert 0 <= score.empathy_score <= 100
    assert 0 <= score.clarity_score <= 100
    assert score.logic_score >= 60
