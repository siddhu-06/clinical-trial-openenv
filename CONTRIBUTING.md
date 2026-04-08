# Contributing to Clinical Trial Protocol Review Environment

## Principles

- Preserve deterministic behavior for the same seed.
- Keep the benchmark domain-specific and clinically grounded.
- Do not weaken OpenEnv compatibility.
- Prefer small, test-backed changes over broad rewrites.

## Protocol Generation

Synthetic protocols are generated entirely in Python and must remain deterministic.
Use `random.Random(seed)` for any protocol-generation randomness. Do not introduce
module-level randomness, external files, or network calls into episode generation.

## Grading Requirements

All graders must:

- Return `float` values in `[0.0, 1.0]`
- Remain deterministic for identical inputs
- Produce meaningful variance across seeds and trajectories
- Preserve the cascade mechanic and safety cap behavior

The environment uses complementary headroom scaling for the final episode reward:

`total_reward = step_rewards + normalized_grade * (1.0 - step_rewards)`

This allows a perfect policy to reach a total score of `1.0` while preserving dense
intermediate learning signal. The medium-task raw grader is normalized to its task
ceiling before headroom scaling so perfect medium-task performance can still achieve
`1.0` without changing the grader itself.

## Testing Workflow

```bash
pip install -e .
pytest tests/ -v
python inference.py
```

## Contribution Scope

Good contributions:

- clearer deterministic protocol templates
- stronger tests
- documentation accuracy improvements
- OpenEnv packaging and compatibility fixes

Avoid:

- architecture rewrites
- nondeterministic behavior
- benchmark-generic abstractions
- unverified dependency bloat
