# Task 003: Visual PPO with a Differentiable CBF-QP Layer

## Goal

Extend your `ppo_cnn` implementation from [Task 002](002_visual_ppo.md) with a differentiable Control Barrier Function Quadratic Program (CBF-QP) layer. The CNN proposes a continuous action, and the QP adjusts it using estimated obstacle geometry.

Proposed architecture:

```text
Image history -> CNN -> proposed action -----------------+
                                                        v
Image history -> geometry estimation -> geometry -> CBF-QP -> executed action
```

The CNN actor must continue to receive only images. The QP requires geometry estimated from observations, so convert sensor data into geometric information that the QP can use, such as distances and robot states. Since GMM parameters are often difficult to obtain, start with a standard CBF-QP rather than a CVaR-BF-QP.

## What to Do

### 1. Differentiable CBF-QP

**Convert** `diff_cvar_gnn` **from Task 1 into** `diff_cbf_gnn`**.** Keep the GNN architecture and PPO framework, but replace the GMM/CVaR constraints with deterministic CBF constraints. Following [BarrierNet's design, Section 5, Equations (6)-(10)](https://arxiv.org/html/2111.11277), let the network learn the **nominal action** and a **positive CBF gain $\kappa_i$** for each obstacle:

$$
\kappa_i=\kappa_{\min}+(\kappa_{\max}-\kappa_{\min})\sigma(\xi_i),
\qquad \dot h_i+\kappa_i h_i\geq-s_i,
$$

$$
h_i=\|q-p_i\|^2-R_i^2.
$$

Handle multiple obstacles exactly as in `diff_cvar_mlp`: retain nearest-$K$ selection, valid masks, per-obstacle slack, control bounds, and all selected constraints in one differentiable QP. Train through the QP and aim for performance comparable to `diff_cvar_gnn`.

### 2. CNN with a Differentiable CBF-QP (Ground-Truth Geometry)

First, provide the QP with **ground-truth obstacle positions and velocities**. The CNN actor must still receive only images.

### 3. Replace Ground-Truth Geometry with Observation-Based Estimates

choose one of them:

- Simulate 2D LiDAR. Convert ranges and angles into points, cluster them to estimate obstacle positions, and track them across frames to estimate velocities for the CBF-QP.
- Use RGB-D: depth measurements and tracking provide geometry for the CBF-QP. Account for robot motion when estimating obstacle velocities.

### 4. Run Comparisons

Compare the Task 002 `ppo_cnn` policy with `barriernet_cnn`, using matching environment settings, robot dynamics, training budgets, and evaluation scenarios.

## Answer These Questions in the PR

Answer briefly and link to the relevant code or results.

1. **CBF-QP:** What changed from CVaR-BF
2. **Inputs:** What do the CNN, and QP receive? Identify any ground-truth inputs.
3. **Geometry:** How are obstacle positions and velocities estimated?
4. **Results and videos:** Provide a results table for 1)`barriernet_cnn` and `ppo_cnn`, and 2) `barriernet_cnn`  with ground-truth geomerty and `barriernet_cnn` with estimated geometry from sensors.

## Submit Your Work

1. Fork the repository to your personal GitHub account.
2. Create a development branch in your fork.
3. Commit changes regularly with clear messages.
4. Open a PR from that branch into your fork's `main`.
5. Answer every **Task 003** question in the PR description. Include code links, sample images, results, and videos.
6. Email us the PR link for review.
