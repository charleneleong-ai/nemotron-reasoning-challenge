# nemotron-reasoning

Single-turn RLVR environment for the NVIDIA Nemotron Model Reasoning Challenge.

- **Task:** a reasoning puzzle prompt → the model thinks, then emits `\boxed{answer}`.
- **Reward:** `boxed_reward` (1.0 if the boxed answer matches gold, exact or ±1e-2) +
  `format_weight` × `format_reward` (1.0 for `<think>…</think>` then `\boxed{}`).
  The answers are machine-checkable, so the grader **is** the verifier — no reward model.
- **Data:** `data/train.csv` (`id,prompt,answer`), bundled into the package; `n_tasks`
  selects a training subset (default 4000; `None` = all rows).

## Usage

```bash
prime env push --path environments/nemotron_reasoning   # publish to chaleong/nemotron-reasoning
prime train     configs/rl/nemotron.toml                # hosted GRPO run on the Nemotron MoE
```

`load_environment(n_tasks=4000, start=0, format_weight=0.1)` returns a
`verifiers.SingleTurnEnv`. Override via the TOML `[[env]] args`.
