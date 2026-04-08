from __future__ import annotations

import random

from clinical_trial_env.models import ViolationFlag
from clinical_trial_env.protocol_generator import ClinicalProtocol
from clinical_trial_env.regulatory_rules import RULES


def generate_reviewer_feedback(
    protocol: ClinicalProtocol,
    agent_flags: list[ViolationFlag],
    seed: int,
) -> str:
    rng = random.Random(seed + 1000)

    violated = {s.section_id: set(s.violated_rule_ids) for s in protocol.sections if s.has_violation}
    correct_flags = [
        f for f in agent_flags if f.section_id in violated and f.rule_id in violated[f.section_id]
    ]

    flagged_pairs = {(f.section_id, f.rule_id) for f in agent_flags}
    unflagged: list[tuple[str, str]] = []
    for section in protocol.sections:
        if not section.has_violation:
            continue
        for rule_id in section.violated_rule_ids:
            if (section.section_id, rule_id) not in flagged_pairs:
                unflagged.append((section.section_id, rule_id))

    disputed_text = ""
    if correct_flags:
        disputed = rng.choice(correct_flags)
        disputed_text = (
            "Disputed Finding:\n"
            f"Your flag on section '{disputed.section_id}' citing {disputed.rule_id} "
            "requires further justification. The language used may satisfy the regulatory "
            "requirement implicitly. Please provide additional evidence for this finding "
            "or revise your assessment.\n"
        )

    if unflagged:
        new_sid, new_rid = rng.choice(unflagged)
        rule_desc = str(RULES.get(new_rid, {}).get("description", "regulatory requirement"))
        new_finding_text = (
            "Additional Finding Identified:\n"
            f"Upon independent review, section '{new_sid}' appears to violate {new_rid}: "
            f"{rule_desc}. This finding was not included in your submission. "
            "Please incorporate this into your revised report.\n"
        )
    else:
        new_finding_text = (
            "Additional Finding Identified:\n"
            "No additional violations identified by independent reviewer at this time.\n"
        )

    return (
        "REGULATORY REVIEWER FEEDBACK\n"
        "════════════════════════════\n\n"
        + disputed_text
        + "\n"
        + new_finding_text
        + "\n"
        + "Please revise your report accordingly and resubmit.\n"
    )
