from __future__ import annotations

from clinical_trial_env.graders import grade_easy, grade_hard, grade_medium
from clinical_trial_env.models import ClinicalTrialAction, Severity, ViolationFlag
from clinical_trial_env.protocol_generator import generate_protocol
from clinical_trial_env.server.environment import ClinicalTrialEnvironment


def make_env():
    return ClinicalTrialEnvironment()


def test_reset_easy_returns_valid_observation():
    env = make_env()
    obs = env.reset(task="easy", seed=42)
    assert obs.task == "easy"
    assert obs.step == 0
    assert obs.episode_done is False
    assert len(obs.protocol_text) > 100
    assert "easy" in obs.available_actions or "flag_violation" in obs.available_actions


def test_reset_medium():
    env = make_env()
    obs = env.reset(task="medium", seed=42)
    assert obs.task == "medium"
    assert obs.max_steps == 8


def test_reset_hard():
    env = make_env()
    obs = env.reset(task="hard", seed=42)
    assert obs.task == "hard"
    assert obs.max_steps == 14


def test_step_flag_violation_correct():
    env = make_env()
    env.reset(task="easy", seed=42)
    action = ClinicalTrialAction(
        action_type="flag_violation",
        violation_flags=[
            ViolationFlag(
                section_id="section_03_criteria",
                rule_id="RULE_004",
                severity=Severity.MAJOR,
                explanation="Eligibility criteria not defined",
                suggested_correction="Add explicit inclusion and exclusion criteria listing age, diagnosis, and contraindications",
            )
        ],
    )
    obs = env.step(action)
    assert obs.step == 1
    assert isinstance(obs.cumulative_reward, float)


def test_step_false_positive_penalized():
    env = make_env()
    env.reset(task="easy", seed=42)
    before_reward = env._cumulative_reward
    action = ClinicalTrialAction(
        action_type="flag_violation",
        violation_flags=[
            ViolationFlag(
                section_id="section_01_objectives",
                rule_id="RULE_003",
                severity=Severity.MINOR,
                explanation="Wrong flag",
                suggested_correction="N/A",
            )
        ],
    )
    env.step(action)
    assert env._cumulative_reward <= before_reward + 0.01


def test_easy_episode_completes():
    env = make_env()
    env.reset(task="easy", seed=42)
    for _ in range(10):
        if env._done:
            break
        action = ClinicalTrialAction(action_type="accept_protocol", violation_flags=[])
        env.step(action)
    assert env._done is True
    assert 0.0 <= env._cumulative_reward <= 1.0


def test_seed_reproducibility():
    env1 = make_env()
    obs1 = env1.reset(task="medium", seed=99)
    env2 = make_env()
    obs2 = env2.reset(task="medium", seed=99)
    assert obs1.protocol_text == obs2.protocol_text
    assert obs1.protocol_id == obs2.protocol_id


def test_different_seeds_produce_different_protocols():
    env = make_env()
    obs1 = env.reset(task="easy", seed=1)
    obs2 = env.reset(task="easy", seed=2)
    assert obs1.protocol_text != obs2.protocol_text


def test_grader_variance():
    scores = []
    for seed in [42, 43, 44]:
        env = make_env()
        env.reset(task="easy", seed=seed)
        if seed % 2 == 0:
            action = ClinicalTrialAction(
                action_type="flag_violation",
                violation_flags=[
                    ViolationFlag(
                        section_id="section_03_criteria",
                        rule_id="RULE_004",
                        severity=Severity.MAJOR,
                        explanation="test",
                        suggested_correction="fix",
                    )
                ],
            )
        else:
            action = ClinicalTrialAction(
                action_type="flag_violation",
                violation_flags=[],
            )
        env.step(action)
        env.step(ClinicalTrialAction(action_type="accept_protocol", violation_flags=[]))
        scores.append(round(env._cumulative_reward, 3))
    assert not all(s == scores[0] for s in scores), f"All scores identical: {scores}"


def test_hard_cascade_section_exists():
    proto = generate_protocol("hard", 42)
    cascade_sections = [s for s in proto.sections if s.cascade_from is not None]
    assert len(cascade_sections) >= 1


def test_bad_accept_caps_score():
    env = make_env()
    env.reset(task="hard", seed=42)
    env.step(ClinicalTrialAction(action_type="accept_protocol", violation_flags=[]))
    assert env._cumulative_reward <= 0.30


def test_grade_easy_perfect_score():
    proto = generate_protocol("easy", 42)
    flags = []
    for section in proto.sections:
        if section.has_violation:
            for rule_id in section.violated_rule_ids:
                from clinical_trial_env.graders import RULES

                sev = section.severity or "minor"
                flags.append(
                    ViolationFlag(
                        section_id=section.section_id,
                        rule_id=rule_id,
                        severity=Severity(sev),
                        explanation="Correct detection",
                        suggested_correction=" ".join(RULES[rule_id]["fix_keywords"]),
                    )
                )
    score = grade_easy(flags, proto)
    assert score >= 0.8, f"Perfect flags should score high, got {score}"


def test_grader_bounds_medium_and_hard():
    proto_medium = generate_protocol("medium", 42)
    medium_score = grade_medium([], proto_medium, steps_used=8)
    assert 0.0 <= medium_score <= 1.0

    proto_hard = generate_protocol("hard", 42)
    hard_score = grade_hard(
        round1_flags=[],
        final_flags=[],
        report_text="",
        protocol=proto_hard,
        adapted_to_feedback=False,
        accepted_with_criticals=True,
        steps_used=14,
    )
    assert 0.0 <= hard_score <= 1.0


def test_episode_summary_info_populated_on_done():
    env = make_env()
    env.reset(task="easy", seed=42)
    env.step(ClinicalTrialAction(action_type="accept_protocol", violation_flags=[]))
    assert env._done is True
    info = env.last_info
    assert "episode_summary" in info
    summary = info["episode_summary"]
    assert "total_violations_in_protocol" in summary
    assert "correctly_flagged" in summary
    assert "final_grader_score" in summary
    assert 0.0 <= float(summary["final_grader_score"]) <= 1.0


def test_parse_llm_response_sanitizes_flags():
    import clinical_trial_env.inference as inference

    parsed = inference.parse_llm_response(
        """```json
        {"action_type":"flag_violation","violation_flags":[{"section_id":"section_03_criteria","rule_id":"RULE_004"}]}
        ```"""
    )
    assert parsed["action_type"] == "flag_violation"
    assert len(parsed["violation_flags"]) == 1
    flag = parsed["violation_flags"][0]
    assert flag["section_id"] == "section_03_criteria"
    assert flag["rule_id"] == "RULE_004"
    assert flag["severity"] == "major"
    assert flag["explanation"] == ""
    assert flag["suggested_correction"] == ""


def test_headroom_reward_formula_reaches_one_for_perfect_easy_agent():
    env = make_env()
    env.reset(task="easy", seed=42)
    proto = env._protocol
    flags = []
    for section in proto.sections:
        if section.has_violation:
            for rule_id in section.violated_rule_ids:
                from clinical_trial_env.regulatory_rules import RULES

                flags.append(
                    ViolationFlag(
                        section_id=section.section_id,
                        rule_id=rule_id,
                        severity=Severity(section.severity or "major"),
                        explanation="Perfect detection",
                        suggested_correction=" ".join(RULES[rule_id]["fix_keywords"]),
                    )
                )
    env.step(ClinicalTrialAction(action_type="flag_violation", violation_flags=flags))
    obs = env.step(ClinicalTrialAction(action_type="accept_protocol", violation_flags=[]))
    assert obs.episode_done is True
    assert abs(obs.cumulative_reward - 1.0) < 1e-6


def test_perfect_medium_agent_scores_at_least_ninety_five():
    env = make_env()
    env.reset(task="medium", seed=42)
    proto = env._protocol
    flags = []
    for section in proto.sections:
        if section.has_violation:
            for rule_id in section.violated_rule_ids:
                from clinical_trial_env.regulatory_rules import RULES

                flags.append(
                    ViolationFlag(
                        section_id=section.section_id,
                        rule_id=rule_id,
                        severity=Severity(section.severity or "major"),
                        explanation="Perfect medium detection",
                        suggested_correction=" ".join(RULES[rule_id]["fix_keywords"]),
                    )
                )
    env.step(ClinicalTrialAction(action_type="flag_violation", violation_flags=flags))
    obs = env.step(
        ClinicalTrialAction(
            action_type="submit_report",
            violation_flags=[],
            report_text=(
                "Comprehensive report covering all violations, severity classifications, "
                "suggested corrections, and overall protocol risk."
            ),
        )
    )
    assert obs.episode_done is True
    assert obs.cumulative_reward >= 0.95
