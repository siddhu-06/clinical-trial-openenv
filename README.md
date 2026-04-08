---
title: Clinical Trial Protocol Review Environment
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
  - healthcare
  - clinical-trials
  - reinforcement-learning
  - agent-evaluation
license: mit
---

# Clinical Trial Protocol Review Environment

## 1. Overview
Clinical Trial Protocol Review Environment is an OpenEnv-compatible benchmark where an agent acts as a regulatory affairs reviewer. The environment generates synthetic clinical trial protocols from seeded Python logic and evaluates agent outputs against deterministic, embedded rules derived from simplified ICH E6 GCP and FDA 21 CFR Part 50 principles. The agent must identify violations, classify severity, suggest corrective language, and submit findings reports.

## 2. Why This Matters
Protocol defects are a major driver of trial delays, amendments, and avoidable compliance risk. Industry analyses consistently estimate multi-billion-dollar annual impact from protocol amendments, recruitment failures, and preventable operational deviations. This environment gives teams a deterministic way to train and evaluate regulatory reasoning before deployment in real clinical workflows.

## 3. Quick Start

### Local Python
```bash
cd clinical_trial_env
python -m venv .venv
. .venv/Scripts/activate
pip install -e .
python -m uvicorn clinical_trial_env.server.app:app --host 0.0.0.0 --port 8000
```

In another terminal:
```bash
python inference.py
```

### OpenEnv Validation
```bash
pip install openenv-core
openenv validate --verbose
```

### Docker (server image)
```bash
cd clinical_trial_env
docker build -t clinical-trial-env-server .
docker run --rm -p 8000:8000 clinical-trial-env-server
```

### Hugging Face Router inference mode
Set `HF_TOKEN` and run:
```bash
python inference.py
```

## 4. Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | OpenAI-compatible inference endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model used by `inference.py` |
| `HF_TOKEN` | empty | Preferred authentication token |
| `LOCAL_IMAGE_NAME` | empty | Optional local image/model selector for alternative runner setups |

If `HF_TOKEN` is not set, `inference.py` falls back to a deterministic local heuristic policy so smoke runs still complete.
That same deterministic heuristic policy is sufficient to achieve the documented baseline scores without making any LLM calls.

## 5. Observation Space

| Field | Type | Description |
|---|---|---|
| `task` | `str` | Task name: `easy`, `medium`, or `hard` |
| `step` | `int` | Current step index |
| `max_steps` | `int` | Maximum steps allowed for current task |
| `protocol_id` | `str` | Deterministic protocol identifier |
| `protocol_text` | `str` | Full protocol text for current episode |
| `available_actions` | `list[str]` | Valid action types for this step |
| `reviewer_feedback` | `str` | Feedback memo for hard-task negotiation |
| `cumulative_reward` | `float` | Episode reward aggregate in `[0.0, 1.0]` |
| `violations_found_so_far` | `int` | Count of correctly matched violation flags |
| `negotiation_round` | `int` | Hard-task feedback round |
| `calibration_hint` | `str` | Severity mix hint shown after step 2 when flags exist |
| `episode_done` | `bool` | Episode completion status |
| `reward` | `float` | Most recent step reward |

## 6. Action Space

| Action Type | Description | When to Use |
|---|---|---|
| `flag_violation` | Submit one or more `ViolationFlag` entries | Early and mid-episode detection |
| `submit_report` | Submit full textual findings report | Medium and hard finalization |
| `accept_protocol` | Accept protocol as compliant | Only when no unresolved critical risks remain |
| `request_clarification` | Request additional clarification context | Hard-task negotiation cycle |

Each action supports:

| Field | Type | Description |
|---|---|---|
| `action_type` | `str` | One action from the enum above |
| `violation_flags` | `list[ViolationFlag]` | Structured findings with section, rule, severity, explanation, correction |
| `report_text` | `str` | Final narrative review payload |
| `explanation` | `str` | Additional rationale for the chosen action |

## 7. Tasks

| Name | Difficulty | Max Steps | Violations | Description | Expected Baseline |
|---|---|---:|---:|---|---|
| Clause Violation Identification | easy | 4 | 2 | Section-level violation detection in short protocol | Model-dependent |
| Full Protocol Review with Corrections | medium | 8 | 4 | End-to-end review with correction quality scoring | Model-dependent |
| Multi-Round Review with Cascade Detection | hard | 14 | 5 | Includes cascade dependency and reviewer negotiation | Model-dependent |

## 8. Reward Function
Step rewards are dense and deterministic:

- Correct flag pair: `+0.12`
- Correct severity on a true positive: `+0.05`
- Correction text includes fix keywords: `+0.04`
- False positive: `-0.10`
- Repeated action or repeated flag: `-0.05`
- Hard task clarification action: `+0.01`
- Accept with unresolved critical violations: `-0.15`

Final graders:

- `grade_easy`: `0.5 * F1 + 0.3 * severity_accuracy + 0.2 * calibration`
- `grade_medium`: weighted blend of detection F1, severity accuracy, correction quality, calibration, and efficiency
- `grade_hard`: weighted blend of round-1 performance, cascade detection, correction quality, calibration, adaptation, report completeness, and efficiency

Final episode scoring uses complementary headroom scaling:

- `final_reward = normalized_grade * (1.0 - cumulative_step_rewards)`

The medium-task raw grader is normalized to its task ceiling before headroom scaling so a perfect medium review can still reach a total score of `1.0` without changing the grader itself.

Safety mechanic:

- If protocol is accepted with unresolved critical findings, hard-task final score is capped at `0.25`.

Example:

- Agent flags one correct major violation with correct severity and one fix keyword match.
- Step reward contribution: `0.12 + 0.05 + 0.04 = 0.21`.

## 9. Regulatory Rules

| Rule ID | Description | Severity |
|---|---|---|
| `RULE_001` | Informed consent must be obtained before any study procedures begin. | critical |
| `RULE_002` | Primary endpoint must be measurable and explicitly defined. | major |
| `RULE_003` | Sample size must be justified with power calculation. | major |
| `RULE_004` | Inclusion and exclusion criteria must be clearly listed. | major |
| `RULE_005` | Investigator qualifications must be documented. | critical |
| `RULE_006` | Adverse event reporting procedures must be defined. | critical |
| `RULE_007` | Phase III protocols must specify a Data Monitoring Committee. | major |
| `RULE_008` | Protocol must define stopping rules for early termination. | major |
| `RULE_009` | Randomization procedures must be described for comparative studies. | major |
| `RULE_010` | Statistical analysis plan must be pre-specified. | major |
| `RULE_011` | Blinding procedures must be described if blinded. | major |
| `RULE_012` | Protocol version and date must be clearly stated. | minor |

## 10. Example Episode

### Reset request
```json
POST /reset
{
  "task": "easy",
  "seed": 42
}
```

### Reset response
```json
{
  "task": "easy",
  "step": 0,
  "max_steps": 4,
  "protocol_id": "PROT-0042-EAS",
  "protocol_text": "PROTOCOL TITLE: A Phase II Study of BPC-771 in Patients with non-small cell lung cancer stage IIIB\nSPONSOR: Novagen Therapeutics\nPHASE: Phase II\nPROTOCOL ID: PROT-0042-EAS\n\nSECTION 1: Study Objectives and Primary Endpoints\nThe primary objective of this Phase II trial sponsored by Novagen Therapeutics is to evaluate the efficacy of BPC-771 in participants with non-small cell lung cancer stage IIIB. The primary endpoint is prospectively defined as change from baseline at Week 24 on a validated disease activity scale with centralized endpoint adjudication. Secondary objectives assess safety, tolerability, and quality-of-life outcomes using pre-specified clinical assessment schedules.\n\n",
  "available_actions": ["flag_violation", "accept_protocol"],
  "reviewer_feedback": "",
  "cumulative_reward": 0.0,
  "violations_found_so_far": 0,
  "negotiation_round": 1,
  "episode_done": false,
  "reward": 0.0
}
```

### Step request
```json
POST /step
{
  "action_type": "flag_violation",
  "violation_flags": [
    {
      "section_id": "section_03_criteria",
      "rule_id": "RULE_004",
      "severity": "major",
      "explanation": "Eligibility is delegated to physician discretion and written criteria are missing.",
      "suggested_correction": "Add explicit inclusion criteria and exclusion criteria with age 18 threshold and written criteria before enrollment."
    },
    {
      "section_id": "section_05_investigator",
      "rule_id": "RULE_005",
      "severity": "critical",
      "explanation": "Investigator qualifications are deferred until after site initiation.",
      "suggested_correction": "Provide investigator CV, training certificate, and documented qualifications before site initiation."
    }
  ],
  "report_text": "",
  "explanation": "Initial compliance review."
}
```

### Step response
```json
{
  "observation": {
    "task": "easy",
    "step": 1,
    "max_steps": 4,
    "protocol_id": "PROT-0042-EAS",
    "protocol_text": "PROTOCOL TITLE: A Phase II Study of BPC-771 in Patients with non-small cell lung cancer stage IIIB\nSPONSOR: Novagen Therapeutics\nPHASE: Phase II\nPROTOCOL ID: PROT-0042-EAS\n\nSECTION 1: Study Objectives and Primary Endpoints\nThe primary objective of this Phase II trial sponsored by Novagen Therapeutics is to evaluate the efficacy of BPC-771 in participants with non-small cell lung cancer stage IIIB. The primary endpoint is prospectively defined as change from baseline at Week 24 on a validated disease activity scale with centralized endpoint adjudication. Secondary objectives assess safety, tolerability, and quality-of-life outcomes using pre-specified clinical assessment schedules.\n\n",
    "available_actions": ["flag_violation", "accept_protocol"],
    "reviewer_feedback": "",
    "cumulative_reward": 0.42,
    "violations_found_so_far": 2,
    "negotiation_round": 1,
    "episode_done": false,
    "reward": 0.42
  },
  "reward": 0.42,
  "done": false
}
```

### Finalize
```json
POST /step
{
  "action_type": "accept_protocol",
  "violation_flags": [],
  "report_text": "",
  "explanation": "All identified findings have been recorded."
}
```

```json
{
  "observation": {
    "task": "easy",
    "step": 2,
    "max_steps": 4,
    "protocol_id": "PROT-0042-EAS",
    "cumulative_reward": 1.0,
    "violations_found_so_far": 2,
    "episode_done": true,
    "reward": 0.58
  },
  "reward": 0.58,
  "done": true
}
```

## 11. Baseline Scores

Scores below were produced by `python inference.py` using the current repository state on seed `42`.

| Task | Score | Notes |
|---|---:|---|
| Easy | `1.0000` | Deterministic heuristic baseline, single-round review |
| Medium | `1.0000` | All four violations recovered with compliant corrections |
| Hard | `1.0000` | Cascade finding and reviewer-feedback adaptation handled correctly |
| Average | `1.0000` | Reproducibility and variance checks pass |

The inference script also confirms that easy-task results are reproducible for `seed=42` and non-constant across seeds `42/43/44`. When API credentials are unavailable, these baseline scores are produced by the built-in deterministic heuristic review policy.

## 12. Citation
If you use this environment in research or evaluation, cite:

```text
Clinical Trial Protocol Review Environment (v1.0.0), OpenEnv-compatible benchmark.
Synthetic seed-deterministic clinical protocol review environment with cascade violation mechanics.
```
