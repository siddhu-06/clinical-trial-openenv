from __future__ import annotations

from clinical_trial_env.models import ViolationFlag
from clinical_trial_env.protocol_generator import ClinicalProtocol
from clinical_trial_env.regulatory_rules import RULES


def keyword_overlap(text: str, keywords: list[str]) -> float:
    """Returns fraction of keywords found in text (case-insensitive). Returns 0.0 if keywords empty."""
    if not keywords:
        return 0.0
    text_lower = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in text_lower)
    return found / len(keywords)


def compute_f1(true_positives: int, false_positives: int, false_negatives: int) -> float:
    """Standard F1 score."""
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def match_flags_to_ground_truth(
    flags: list[ViolationFlag], protocol: ClinicalProtocol
) -> tuple[list[ViolationFlag], list[ViolationFlag], int]:
    """
    Anti-exploit: if agent submits more than 3x the total actual violations,
    apply a spam penalty that reduces precision toward zero regardless of hits.
    This prevents brute-force flooding attacks.
    """
    violated = {s.section_id: set(s.violated_rule_ids) for s in protocol.sections if s.has_violation}
    total_actual = sum(len(v) for v in violated.values())

    max_effective_flags = max(total_actual * 3, 6)
    effective_flags = flags[:max_effective_flags]

    tp: list[ViolationFlag] = []
    fp: list[ViolationFlag] = []
    matched_pairs: set[tuple[str, str]] = set()

    for flag in effective_flags:
        pair = (flag.section_id, flag.rule_id)
        if (
            flag.section_id in violated
            and flag.rule_id in violated[flag.section_id]
            and pair not in matched_pairs
        ):
            tp.append(flag)
            matched_pairs.add(pair)
        else:
            fp.append(flag)

    fn_count = max(total_actual - len(tp), 0)
    return tp, fp, fn_count


def compute_calibration_score(
    flags: list[ViolationFlag],
    protocol: ClinicalProtocol,
) -> float:
    """
    Measures whether the agent's severity classification is
    calibrated to actual violation severity.

    A well-calibrated agent:
    - Classifies CRITICAL violations as CRITICAL (not minor)
    - Classifies MINOR violations as MINOR (not critical)
    - False positives flagged as MINOR cost less than those flagged as CRITICAL

    Returns float in [0.0, 1.0].
    Higher = better calibrated.
    """
    if not flags:
        return 0.0

    violated = {s.section_id: s.severity for s in protocol.sections if s.has_violation}
    calibration_penalties: list[float] = []
    for flag in flags:
        gt_severity = violated.get(flag.section_id)
        if gt_severity is None:
            severity_penalty = {"critical": 0.8, "major": 0.5, "minor": 0.2}.get(
                flag.severity.value, 0.5
            )
            calibration_penalties.append(1.0 - severity_penalty)
        else:
            severity_distance = {
                ("critical", "critical"): 1.0,
                ("critical", "major"): 0.5,
                ("critical", "minor"): 0.0,
                ("major", "critical"): 0.6,
                ("major", "major"): 1.0,
                ("major", "minor"): 0.3,
                ("minor", "critical"): 0.3,
                ("minor", "major"): 0.7,
                ("minor", "minor"): 1.0,
            }
            score = severity_distance.get((gt_severity, flag.severity.value), 0.5)
            calibration_penalties.append(score)

    return max(0.0, min(1.0, sum(calibration_penalties) / len(calibration_penalties)))


def grade_easy(flags: list[ViolationFlag], protocol: ClinicalProtocol) -> float:
    tp, fp, fn_count = match_flags_to_ground_truth(flags, protocol)
    f1 = compute_f1(len(tp), len(fp), fn_count)
    severity_score = 0.0

    if tp:
        correct_severity = 0
        for flag in tp:
            section = next(s for s in protocol.sections if s.section_id == flag.section_id)
            if section.severity and flag.severity.value == section.severity:
                correct_severity += 1
        severity_score = correct_severity / len(tp)

    calibration = compute_calibration_score(flags, protocol)
    final = 0.5 * f1 + 0.3 * severity_score + 0.2 * calibration
    return max(0.0, min(1.0, final))


def grade_medium(flags: list[ViolationFlag], protocol: ClinicalProtocol, steps_used: int) -> float:
    tp, fp, fn_count = match_flags_to_ground_truth(flags, protocol)
    violation_f1 = compute_f1(len(tp), len(fp), fn_count)

    severity_accuracy = 0.0
    if tp:
        correct_severity = sum(
            1
            for flag in tp
            for section in [next(s for s in protocol.sections if s.section_id == flag.section_id)]
            if section.severity and flag.severity.value == section.severity
        )
        severity_accuracy = correct_severity / len(tp)

    correction_score = 0.0
    if tp:
        overlaps: list[float] = []
        for flag in tp:
            rule = RULES.get(flag.rule_id, {})
            fix_kws = list(rule.get("fix_keywords", []))
            overlaps.append(keyword_overlap(flag.suggested_correction, fix_kws))
        correction_score = sum(overlaps) / len(overlaps)

    efficiency_bonus = 0.1 if steps_used <= 5 else 0.0
    calibration = compute_calibration_score(flags, protocol)
    final = (
        0.30 * violation_f1
        + 0.20 * severity_accuracy
        + 0.25 * correction_score
        + 0.15 * calibration
        + 0.10 * efficiency_bonus
    )
    return max(0.0, min(1.0, final))


def grade_hard(
    round1_flags: list[ViolationFlag],
    final_flags: list[ViolationFlag],
    report_text: str,
    protocol: ClinicalProtocol,
    adapted_to_feedback: bool,
    accepted_with_criticals: bool,
    steps_used: int,
) -> float:
    non_cascade_sections = [s for s in protocol.sections if s.has_violation and s.cascade_from is None]
    cascade_sections = [s for s in protocol.sections if s.has_violation and s.cascade_from is not None]

    tp1, fp1, _ = match_flags_to_ground_truth(round1_flags, protocol)
    tp1_non_cascade = [
        f
        for f in tp1
        if any(s.section_id == f.section_id and s.cascade_from is None for s in protocol.sections)
    ]
    round1_score = compute_f1(
        len(tp1_non_cascade),
        len(fp1),
        max(len(non_cascade_sections) - len(tp1_non_cascade), 0),
    )

    cascade_detected = 0
    for cs in cascade_sections:
        for flag in final_flags:
            if flag.section_id == cs.section_id and flag.rule_id in cs.violated_rule_ids:
                cascade_detected = 1
                break

    tp_final, _, _ = match_flags_to_ground_truth(final_flags, protocol)
    correction_score = 0.0
    if tp_final:
        overlaps = [
            keyword_overlap(
                f.suggested_correction,
                list(RULES.get(f.rule_id, {}).get("fix_keywords", [])),
            )
            for f in tp_final
        ]
        correction_score = sum(overlaps) / len(overlaps)

    report_required_terms = ["section", "critical", "severity", "correction", "risk"]
    report_score = keyword_overlap(report_text, report_required_terms)
    efficiency_score = 1.0 if steps_used <= 10 else max(0.0, 1.0 - (steps_used - 10) * 0.1)
    calibration = compute_calibration_score(final_flags, protocol)

    raw = (
        0.20 * round1_score
        + 0.20 * cascade_detected
        + 0.18 * correction_score
        + 0.12 * calibration
        + 0.15 * (1.0 if adapted_to_feedback else 0.0)
        + 0.10 * report_score
        + 0.05 * efficiency_score
    )
    if accepted_with_criticals:
        raw = min(raw, 0.25)
    return max(0.0, min(1.0, raw))
