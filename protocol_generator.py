from __future__ import annotations

from dataclasses import dataclass
import random

from clinical_trial_env.regulatory_rules import RULES


@dataclass
class ProtocolSection:
    section_id: str
    section_title: str
    content: str
    has_violation: bool
    violated_rule_ids: list[str]
    severity: str | None
    cascade_from: str | None


@dataclass
class ClinicalProtocol:
    protocol_id: str
    title: str
    sponsor: str
    phase: str
    sections: list[ProtocolSection]
    full_text: str
    seed: int
    task: str


SPONSORS = [
    "Novagen Therapeutics",
    "BioPath Sciences",
    "Meridian Clinical",
    "Apex Biopharma",
    "Celrogen Inc",
]
INDICATIONS = [
    "moderate-to-severe rheumatoid arthritis",
    "non-small cell lung cancer stage IIIB",
    "treatment-resistant major depressive disorder",
    "Type 2 diabetes with cardiovascular risk",
    "relapsing-remitting multiple sclerosis",
]
COMPOUNDS = ["Compound XR-42", "BPC-771", "MRD-220", "APX-115", "CLG-904"]
PHASES = ["Phase I", "Phase II", "Phase III"]
SECTION_TITLES = [
    ("section_01_objectives", "Study Objectives and Primary Endpoints"),
    ("section_02_design", "Study Design and Methodology"),
    ("section_03_criteria", "Subject Selection and Eligibility Criteria"),
    ("section_04_consent", "Informed Consent Procedures"),
    ("section_05_investigator", "Investigator Qualifications and Responsibilities"),
    ("section_06_safety", "Safety Monitoring and Adverse Event Reporting"),
    ("section_07_statistics", "Statistical Analysis Plan"),
    ("section_08_randomization", "Randomization and Blinding Procedures"),
    ("section_09_dmc", "Data Monitoring Committee"),
    ("section_10_stopping", "Stopping Rules and Early Termination"),
    ("section_11_version", "Protocol Version Control and Document History"),
    ("section_12_sample", "Sample Size Justification"),
]


def _clean_content_for_section(
    section_id: str,
    sponsor: str,
    compound: str,
    indication: str,
    phase: str,
) -> str:
    templates: dict[str, str] = {
        "section_01_objectives": (
            f"The primary objective of this {phase} trial sponsored by {sponsor} is to evaluate "
            f"the efficacy of {compound} in participants with {indication}. "
            "The primary endpoint is prospectively defined as change from baseline at Week 24 "
            "on a validated disease activity scale with centralized endpoint adjudication. "
            "Secondary objectives assess safety, tolerability, and quality-of-life outcomes "
            "using pre-specified clinical assessment schedules."
        ),
        "section_02_design": (
            f"This multicenter, controlled protocol evaluates {compound} across screened participants "
            f"with {indication}. "
            "Study visits are scheduled every three weeks with protocol-defined assessments and source documentation requirements. "
            "Site monitoring plans include routine data verification, protocol deviation oversight, and audit-readiness checks. "
            "Operational conduct follows sponsor SOPs and ICH-aligned quality management practices."
        ),
        "section_03_criteria": (
            "Eligibility is determined using written inclusion and exclusion criteria approved before site activation. "
            "Inclusion criteria include documented diagnosis, age at least 18 years, and protocol-defined baseline disease burden. "
            "Exclusion criteria include clinically significant organ dysfunction, unstable comorbidity, and contraindicated concomitant medications. "
            "All criteria are verified during screening and recorded before enrollment authorization."
        ),
        "section_04_consent": (
            "Informed consent will be obtained from all participants in accordance with ICH E6(R2) Good Clinical Practice guidelines. "
            "Written informed consent must be signed and dated by the participant before any study-specific procedures are performed. "
            "The consent process is documented in source records and a copy of the signed form is provided to each participant. "
            "All consent forms are approved by the Institutional Review Board prior to site use."
        ),
        "section_05_investigator": (
            f"Investigators participating in the {sponsor} program must demonstrate therapeutic-area expertise for {indication}. "
            "Current investigator CVs, medical licenses, and delegated responsibility logs are maintained in the site trial master file. "
            "Protocol-specific training records are completed before first participant enrollment and reviewed during monitoring visits. "
            "Principal investigators attest to participant safety oversight and protocol compliance responsibilities."
        ),
        "section_06_safety": (
            "Safety monitoring procedures define adverse event collection, seriousness assessment, and sponsor escalation requirements. "
            "Serious adverse events must be reported within 24 hours of site awareness using standardized pharmacovigilance workflows. "
            "Follow-up of unresolved safety events continues until clinical resolution or stabilization criteria are met. "
            "Regular safety review meetings are conducted with sponsor medical monitors and investigators."
        ),
        "section_07_statistics": (
            "The statistical analysis plan is finalized and approved before database lock and unblinding. "
            "Primary efficacy analyses are pre-specified for the intent-to-treat population with protocol-defined sensitivity analyses. "
            "Missing data handling, multiplicity control, and subgroup analyses are documented in the SAP before first patient first visit. "
            "All analyses are executed with validated software and quality-controlled outputs."
        ),
        "section_08_randomization": (
            "Participants are randomized centrally using an interactive web response system with allocation concealment controls. "
            "Randomization sequences are generated by independent statisticians and stratified by key baseline factors. "
            "Blinding procedures define access controls, emergency unblinding pathways, and blinded endpoint review safeguards. "
            "Site staff receive role-specific blinding maintenance training before activation."
        ),
        "section_09_dmc": (
            "An independent Data Monitoring Committee periodically reviews aggregate efficacy and safety data for participant protection. "
            "Committee membership includes clinical, statistical, and ethics experts independent of sponsor operations. "
            "Meeting cadence, review materials, and recommendation pathways are documented in a signed DMC charter. "
            "Actionable recommendations are tracked through sponsor governance procedures."
        ),
        "section_10_stopping": (
            "Predefined stopping boundaries cover futility, overwhelming efficacy, and unacceptable toxicity thresholds. "
            "Interim review timing and alpha-spending assumptions are documented before enrollment starts. "
            "Early termination procedures define participant transition plans and regulatory communication timelines. "
            "Stopping decisions are overseen by independent review in alignment with the protocol charter."
        ),
        "section_11_version": (
            "This protocol is controlled under formal document governance with explicit version and effective date metadata. "
            "Every amendment includes revision history, rationale, and approval signatures before implementation. "
            "Superseded versions are archived, access-controlled, and retained in the trial master file. "
            "All participating sites receive version-controlled distributions with acknowledgement tracking."
        ),
        "section_12_sample": (
            "Sample size assumptions are justified by expected treatment effect, endpoint variance, and attrition projections. "
            "A two-sided alpha of 0.05 and target 80 percent power are used for primary endpoint planning. "
            "Sensitivity analyses test robustness to dropout and protocol deviation scenarios. "
            "Enrollment assumptions are monitored prospectively and revisited through predefined governance checkpoints."
        ),
    }
    return templates[section_id]


def _violation_content(
    sponsor: str,
    compound: str,
    indication: str,
    rule_id: str,
    include_cascade_note: bool = False,
) -> str:
    trigger = str(RULES[rule_id]["violation_trigger"])
    base = (
        f"The {sponsor} protocol for {compound} in {indication} describes operational procedures for this section. "
        f"However, {trigger}. "
        "This language introduces a material regulatory compliance concern and requires protocol revision before further trial conduct. "
        "The sponsor acknowledges this deviation and will need a corrective amendment."
    )
    if include_cascade_note:
        base = (
            base
            + " Note: The analysis population cannot be defined because eligibility criteria have not been specified."
        )
    return base


def _build_sections_for_task(
    task: str,
    sponsor: str,
    compound: str,
    indication: str,
    phase: str,
) -> list[ProtocolSection]:
    all_sections: dict[str, ProtocolSection] = {}
    for section_id, section_title in SECTION_TITLES:
        all_sections[section_id] = ProtocolSection(
            section_id=section_id,
            section_title=section_title,
            content=_clean_content_for_section(section_id, sponsor, compound, indication, phase),
            has_violation=False,
            violated_rule_ids=[],
            severity=None,
            cascade_from=None,
        )

    task_lower = task.lower()
    if task_lower == "easy":
        selected_ids = [
            "section_01_objectives",
            "section_02_design",
            "section_03_criteria",
            "section_04_consent",
            "section_05_investigator",
        ]
        violations = {
            "section_03_criteria": "RULE_004",
            "section_05_investigator": "RULE_005",
        }
    elif task_lower == "medium":
        selected_ids = [
            "section_01_objectives",
            "section_02_design",
            "section_03_criteria",
            "section_04_consent",
            "section_05_investigator",
            "section_06_safety",
            "section_08_randomization",
            "section_11_version",
        ]
        violations = {
            "section_04_consent": "RULE_001",
            "section_06_safety": "RULE_006",
            "section_08_randomization": "RULE_009",
            "section_11_version": "RULE_012",
        }
    elif task_lower == "hard":
        selected_ids = [section_id for section_id, _ in SECTION_TITLES]
        violations = {
            "section_04_consent": "RULE_001",
            "section_06_safety": "RULE_006",
            "section_03_criteria": "RULE_004",
            "section_07_statistics": "RULE_010",
            "section_09_dmc": "RULE_007",
        }
    else:
        raise ValueError(f"Unsupported task: {task}")

    for section_id, rule_id in violations.items():
        section = all_sections[section_id]
        section.has_violation = True
        section.violated_rule_ids = [rule_id]
        section.severity = str(RULES[rule_id]["severity"])
        section.content = _violation_content(
            sponsor=sponsor,
            compound=compound,
            indication=indication,
            rule_id=rule_id,
            include_cascade_note=(section_id == "section_07_statistics" and task_lower == "hard"),
        )
        if section_id == "section_07_statistics" and task_lower == "hard":
            section.cascade_from = "section_03_criteria"

    return [all_sections[section_id] for section_id in selected_ids]


def generate_protocol(task: str, seed: int) -> ClinicalProtocol:
    task_lower = task.lower()
    rng = random.Random(seed)
    sponsor = rng.choice(SPONSORS)
    indication = rng.choice(INDICATIONS)
    compound = rng.choice(COMPOUNDS)
    phase = "Phase III" if task_lower == "hard" else rng.choice(PHASES)
    protocol_id = f"PROT-{seed:04d}-{task_lower[:3].upper()}"
    title = f"A {phase} Study of {compound} in Patients with {indication}"

    sections = _build_sections_for_task(task_lower, sponsor, compound, indication, phase)

    header = (
        f"PROTOCOL TITLE: {title}\n"
        f"SPONSOR: {sponsor}\n"
        f"PHASE: {phase}\n"
        f"PROTOCOL ID: {protocol_id}\n\n"
    )
    body_chunks: list[str] = []
    for i, section in enumerate(sections):
        body_chunks.append(f"SECTION {i + 1}: {section.section_title}\n{section.content}\n\n")
    full_text = header + "".join(body_chunks)

    # Hard-mode obfuscation for odd seeds in medium/hard tasks
    if task_lower in ("medium", "hard") and (seed % 2 == 1):
        filler_sentences = [
            f"The sponsor {sponsor} is committed to conducting this study in full accordance with applicable regulatory requirements.",
            "All study procedures will be overseen by the principal investigator and documented in the trial master file.",
            "Site staff will receive comprehensive training prior to subject enrollment.",
            "All data will be captured in the electronic data capture system within 24 hours of collection.",
            "Quality assurance visits will be conducted quarterly throughout the study duration.",
            f"The study will be conducted across {rng.randint(3, 8)} investigational sites globally.",
            "Standard of care procedures will not be modified for study participants.",
            "All biological samples will be processed and stored according to the laboratory manual.",
            "Participant medical records will be made available to the sponsor for source data verification.",
            f"The study timeline anticipates a {rng.randint(12, 36)}-month enrollment period.",
        ]
        new_sections: list[ProtocolSection] = []
        for section in sections:
            if section.has_violation:
                filler = rng.choice(filler_sentences)
                section.content = filler + " " + section.content
            new_sections.append(section)
        sections = new_sections
        full_text = (
            f"PROTOCOL TITLE: {title}\n"
            f"SPONSOR: {sponsor}\n"
            f"PHASE: {phase}\n"
            f"PROTOCOL ID: {protocol_id}\n\n"
        )
        for i, section in enumerate(sections):
            full_text += f"SECTION {i + 1}: {section.section_title}\n{section.content}\n\n"

    return ClinicalProtocol(
        protocol_id=protocol_id,
        title=title,
        sponsor=sponsor,
        phase=phase,
        sections=sections,
        full_text=full_text,
        seed=seed,
        task=task_lower,
    )


def generate_protocol_batch(task: str, seeds: list[int]) -> list[ClinicalProtocol]:
    """Generate a batch of protocols for the same task with different seeds.
    Useful for evaluating score variance. Returns list in same order as seeds."""
    return [generate_protocol(task, seed) for seed in seeds]
