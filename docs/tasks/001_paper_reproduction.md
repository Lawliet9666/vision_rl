# Task 001: DiffCVaR Reproduction Audit and JAX Baselines

## Purpose

Establish a traceable reproduction baseline for the [DiffCVaR paper](https://lawliet9666.github.io/rlcvarbf/) using the code in this repository. First map the paper's claims to the released implementation, then train and evaluate the two JAX MLP methods that this snapshot supports. The paper's proposed method uses a graph neural network (GNN); this repository currently dispatches only `diff_cvar_mlp` and `ppo_mlp`. Do not label MLP measurements as a reproduction of the paper's proposed GNN row.

This is a state-observation reproduction task. Image observations and a vision policy are future work.

## Learning Goals

- Set up the JAX environment and run a short training and evaluation smoke check.
- Trace training configuration, model selection, checkpoint loading, and evaluation metrics to source code.
- Distinguish a reproduced result from a related baseline and from an unavailable experiment.
- Record seeds, checkpoint selection, hardware, commands, configuration, raw metrics, and failure modes so another person can rerun the experiment.
- Use an issue, a task branch, small commits, and a PR for review.

## Reading and References

- [Paper and project page](https://lawliet9666.github.io/rlcvarbf/) — especially Section V, Table I, Figure 2, Table II, and Table III.
- [`main_jax` source branch](https://github.com/anonymousrobotics9666/Reinforcement-Learning-Adaptive-CVaR-Barrier-Function/tree/main_jax), imported here from commit `291c9162dae8ed04c070364a94a5a586f720ce95`.
- `README.md`, `config/`, `scripts/run_ppo.sh`, `scripts/eval.py`, `model/__init__.py`, and `trainer/jax_ppo_eval.py` in this repository.

## What the Starting Code Supports

`model/__init__.py` selects `diff_cvar_mlp` or `ppo_mlp`. The default configuration uses the unicycle robot, 20 dynamic obstacles, a 0.1 s time step, and 400 steps per episode. `scripts/eval.py` can evaluate a JAX `.pkl` checkpoint with ten default seed groups and 50 episodes per group, reporting return, success, collision, timeout, successful trajectory time, and successful minimum clearance.

No checkpoint is tracked in this repository. The paper's Diff-CVaR-GNN, its other comparison methods, and its ablations are not present in this source snapshot. Record each missing method or artifact in the audit; do not substitute an MLP result without naming the difference.

## Repository Locations

Put the task report, experiment matrix, commands, and small result tables in:

```text
experiments/exp_001_paper_reproduction/README.md
```

Put reusable scripts added for this task under `scripts/`. Keep machine-specific checkpoints and W&B logs under ignored `outputs/` and `wandb/`. Include a small, machine-readable summary under `experiments/exp_001_paper_reproduction/` if useful; do not commit large checkpoints or generated video files.

## GitHub Workflow

1. Open an issue titled `Task 001: DiffCVaR reproduction audit and JAX baselines`. State the hardware, proposed experiment matrix, estimated compute budget, and the result that will count as a successful check-in.
2. Create `task-001-paper-reproduction` from `main`. Do not push assignment work directly to `main`.
3. Make small commits for the audit, smoke check, each model run, and the report. Push the task branch so the commits are visible for review.
4. Open a PR with **base** `main` and **compare** `task-001-paper-reproduction`. Link the issue. The instructor can review and request changes without merging.

## Work Plan

### 1. Audit the paper against the source

Create a matrix with one row per paper result: Table I, Figure 2, Table II, and Table III. For each, list the model/baselines, scenario, obstacle count, metrics, matching code/config/checkpoint, and status: runnable, requires implementation, or missing artifact. Confirm the distinction between the paper's GNN policy and this branch's MLP policies from `model/__init__.py` and the paper's Section V.

### 2. Verify the local environment

Follow the root README. Record Python, JAX, Optax, and device versions. Run a short smoke training job for each model before committing to full runs. Show that a checkpoint is written and that `scripts/eval.py` can load it. If a command fails, include the exact command, error, and diagnosis in the report; do not silently change the experiment definition.

### 3. Train the available baselines

Train `diff_cvar_mlp` and `ppo_mlp` with the same unicycle and 20-obstacle environment settings. The default full training budget is `total_timesteps: 20000000` in `config/base.yaml`. Use offline W&B logging unless online tracking has been arranged. Record every override and the selected checkpoint step. If hardware limits prevent a full run, report the completed timestep budget and label the result as a partial run.

```bash
WANDB_MODE=offline RUN_NAME=task001_diff_cvar_mlp bash scripts/run_ppo.sh
WANDB_MODE=offline RUN_NAME=task001_ppo_mlp bash scripts/run_ppo.sh model=ppo_mlp
```

### 4. Evaluate with a fixed protocol

Evaluate each selected JAX checkpoint with seeds `100,200,...,1000` and 50 episodes per seed group (500 episodes total). Keep the saved run `config.yaml` with its checkpoint. Report success, collision, and timeout rates; average return; successful trajectory time; and successful minimum clearance. Check that success + collision + timeout is approximately 100%. State how any `mean ± std` values are aggregated. The existing evaluator pools episodes; do not present its episode-level standard deviation as a standard deviation over ten independent training runs.

```bash
python scripts/eval.py \
  --save-dir outputs/crowd_dyn_var_num_env/runs/<run> \
  --checkpoint outputs/crowd_dyn_var_num_env/runs/<run>/ckpt_<step>.pkl \
  --seeds 100,200,300,400,500,600,700,800,900,1000 \
  --episodes 50
```

Use the same evaluation protocol for both models. Report the run names and checkpoint paths, but do not include absolute paths specific to one computer in runnable instructions.

### 5. Compare and explain

Place the two measured MLP rows beside the relevant paper numbers, with explicit architecture labels. For context, the paper's Table I reports 58.2% success for Vanilla RL and 97.0% for the **GNN-based** proposed method in the 20-obstacle unicycle setting. Those paper rows are reference values, not pass thresholds for these MLP models. Explain any config, architecture, checkpoint, training-seed, or metric mismatch before interpreting a numerical difference. Finish the audit matrix with the work and artifacts needed to reproduce Figure 2, Table II, Table III, and all Table I baselines faithfully.

## What to Submit in the PR

- `experiments/exp_001_paper_reproduction/README.md` with the audit matrix, exact commands, environment/device versions, run configs, checkpoint steps, and a two-model results table.
- Evidence of successful smoke training and checkpoint evaluation, or a clear failure record with reproduction steps.
- Raw evaluation JSON or a small script and output that regenerates the reported metrics; omit large checkpoints.
- A short answer to each question: Which paper rows are genuinely reproducible from this snapshot? How does MLP differ from the paper's GNN policy? What additional code or artifacts are required for full paper reproduction? How were seeds and standard deviations handled?

## PR Description Template

```markdown
## Summary
- Source commit and code changes:
- Models and scenarios evaluated:

## Reproduction audit
- Paper results supported by this snapshot:
- Missing implementations or artifacts:

## Protocol
- Hardware, Python/JAX/Optax versions:
- Training commands, seeds, timestep budgets, and overrides:
- Checkpoint selection and evaluation command:

## Results
| Model | Architecture | Episodes | Success % | Collision % | Timeout % | Avg. return | Traj. time | Min. clearance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PPO | MLP | | | | | | | |
| DiffCVaR | MLP | | | | | | | |

## Interpretation and limits
- Comparison to Table I:
- Metric/architecture differences:
- Work needed for full paper reproduction:

## Checklist
- [ ] Issue and task branch are linked
- [ ] Commands and configs are recorded
- [ ] Both available models were evaluated or failures are documented
- [ ] Rates sum to approximately 100%
- [ ] MLP results are not labeled as GNN results
- [ ] PR targets `main`
```

The assignment is complete when another person can trace every reported number to a run, checkpoint, configuration, and evaluation command, and when unavailable paper results are clearly identified. A successful smoke check or MLP comparison alone is not a claim that the full paper has been reproduced.
