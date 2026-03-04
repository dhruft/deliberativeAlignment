LaTeX Instructions: https://www.overleaf.com/read/mfqgfrddjqvy#868fc1

# Project Summary

This goal of the project is to train an LLM to adhere to certain safety specifications that prevent scheming. This is done by collecting rollouts in environments where the goal of the agent is in tension with the safety specifications. These rollouts are labeled by a monitor. The rollouts classified as honest are then used to train a new LLM to find the delicate balance between antischeming and anti-overrefusal. 

## Architecture

```
envs/
├── environments/           # File-based evaluation scenarios
│   ├── moltbook_evil/
│   │   ├── task.txt        # Agent role + scenario (system prompt)
│   │   ├── conversation.txt
│   │   └── rewards.txt     # Desirable behavior spec for monitor
│   └── ...
├── base.py                 # BaseTask, TaskConfig
├── file_loader.py          # Loads envs from folders
└── registry.py             # Auto-discovers file-based envs

engine/
├── providers.py            # OpenAI, Anthropic, Together AI, MLX (local), Mock
├── rollout.py              # RolloutGenerator + safety spec injection
├── trainer.py              # SFT fine-tuning via Together AI API
└── safety_spec.txt         # AS1-GP4 anti-scheming rules

monitors/
├── heuristic.py            # Fast pattern-based filtering
└── deliberative_monitor.py # LLM evaluates CoT + action vs rewards.txt

data/
├── storage.py              # TrajectoryStore, ReviewableRollout
└── experiments/            # exp-N/ folders (auto-created)

analysis/
├── metrics.py              # compute_metrics(), compare_experiments()
└── visualize.py            # Behavior distribution plots, heatmaps

configs/
└── deliberative_training.yaml
```

### Experiment Folder Structure

```
data/experiments/
  exp-1/
    metadata.json
    moltbook_evil/          # collect-phase results
      trajectories.jsonl
      labels.jsonl
      review.jsonl          # human-editable before training
    train-1/                # SFT artifacts (job.json, model_id.txt)
    train-2/                # re-train if needed
    test/                   # test-phase results (replaced each run)
      moltbook_evaluation/
        trajectories.jsonl
        labels.jsonl
        review.jsonl
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # add API keys
```

in .env:
- `TOGETHER_API_KEY` — Together AI (default inference + fine-tuning provider)

---

## Workflow

### Step 1 — Collect

Generate rollouts on training environments with the safety spec injected. A 70B monitor evaluates each rollout and writes a human-reviewable `review.jsonl`.

```bash
python run_experiment.py configs/deliberative_training.yaml --collect
```

### Step 2 — Review

Inspect and correct monitor labels before training. Use `[H]`onest, `[S]`cheming, `[O]`verrefusal, `[U]`nclear, `[K]`eep.

```bash
python scripts/review.py exp-1
python scripts/review.py exp-1 --filter scheming unclear   # only show these
```

### Step 3 — Train

Fine-tune on rollouts labeled `"honest"` via the Together AI fine-tuning API.

```bash
python run_experiment.py configs/deliberative_training.yaml --train --experiment-id exp-1
```

Training artifacts are saved to `data/experiments/exp-1/train-1/` (auto-increments on re-runs).

### Step 4 — Test

Evaluate the fine-tuned model on held-out environments **without** the safety spec. Results are saved to `exp-1/test/`.

```bash
python run_experiment.py configs/deliberative_training.yaml --test --experiment-id exp-1
```

### Analyze

```bash
python analyze.py exp-1                          # collect-phase metrics
python analyze.py exp-1 --subdir test            # test-phase metrics
python analyze.py exp-1 exp-2 --compare          # compare experiments
python analyze.py exp-1 --report -o reports/     # generate full report

python scripts/review.py exp-1 --subdir test     # review test rollouts
```

---

## Config

```yaml
rollout_model:
  provider: together          # together | openai | anthropic | mlx | mock
  name: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo

test_model:                   # used for --test; falls back to rollout_model if absent
  provider: mlx
  name: data/local_models_mlx

monitor_model:
  provider: together
  name: meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo

sft_model:
  model: meta-llama/Meta-Llama-3.1-8B-Instruct-Reference
  n_epochs: 3
  batch_size: 8
  learning_rate: 1e-5
  suffix: scheming-sft

splits:
  train: [moltbook_evil, moltbook_moderation]
  test: [moltbook_evaluation]

n_rollouts: 10
temperature: 0.7
max_tokens: 2048
```

---

## Adding New Environments

```bash
python scripts/add_env.py my_new_scenario
# then edit envs/environments/my_new_scenario/task.txt and rewards.txt
```

Required files per environment:
- `task.txt` — self-contained system prompt (agent role + scenario)
- `rewards.txt` — defines DESIRABLE OUTCOMES, DESIRABLE REASONING, UNDESIRABLE BEHAVIOR

Optional:
- `conversation.txt` — opening user message shown to the agent

CRUICAL: right now this infrastructure only supports conversations with one turn (meaning the agent reads the past data, and generates a single response). More work needs to be done to support more nuanced environments.

---

## Local Inference (Apple Silicon)

To run inference locally with a downloaded/fine-tuned model:

```bash
# Download a fine-tuned model from Together AI
python scripts/download_model.py <job-id> --output data/local_models

# Quantize to 4-bit for memory efficiency (~5GB)
mlx_lm.convert --hf-path data/local_models --mlx-path data/local_models_mlx -q
```

Then set `test_model.provider: mlx` and `test_model.name: data/local_models_mlx` in config.

---

## Key Metrics

- **Covert Action Rate** — fraction of rollouts taking hidden or deceptive actions
- **Aligned Rate** — fraction explicitly reasoning about safety principles and acting accordingly
- **Misalignment Delta** — covert_rate(no spec) − covert_rate(with spec): measures how much the spec is doing the work vs internalized alignment
