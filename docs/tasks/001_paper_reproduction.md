# Task 001: Reproduce DiffCVaR-GNN and Compare with MLP

## Goal

Add the GNN from the [DiffCVaR paper](https://arxiv.org/abs/2605.21257) to this JAX code. Compare it with the current `diff_cvar_mlp` policy using test results and videos. Does the GNN help the robot navigate better?

## Read First

- **CBFs:** [Ames et al.](http://ames.caltech.edu/ames2019control.pdf) and this [tutorial](https://arxiv.org/abs/2603.06954). Focus on the main idea; you can skip the dense derivations.
- **Safety and risk:** [BarrierNet](https://arxiv.org/abs/2111.11277), [DiffCVaR](https://arxiv.org/pdf/2605.21257) (page 4 for the CVaR constraint; Sections IV–V for the GNN and tests), and [CVaR intuition](https://arxiv.org/pdf/2504.06513) (pages 2–3 and Figure 2).
- **GNN examples:** [starter notebook](https://github.com/tkkim-robot/minimal-graph-attention-starter/blob/main/graph_attention_starter.ipynb), [GAT code](https://github.com/tkkim-robot/online_adaptive_cbf/blob/main/nn_model/penn/gat.py), and its [paper](https://arxiv.org/pdf/2504.03038) (page 7).

## What to Do

1. Read the README. Find where observations become policy inputs in `env/jax_social_nav.py` and `model/jax_diff_cvar_mlp.py`, then find where `model/jax_cvar_qp.py` computes the safe action. The policy sees all obstacle slots; the QP chooses a smaller set of nearby obstacles.
2. Follow the paper to build a graph from the robot, goal information, and all active obstacles. Use the three observation frames. Feed the GNN output to the policy so it still produces a nominal action and a risk level and safety margin for each obstacle. Inspect the slack variables, action bounds, and slack penalty for infeasiblity problem.
3. Train the MLP and GNN with the same environment, training steps, and random seeds. Test both in the paper's 20-obstacle unicycle setting. Report success, collision, and timeout rates; total reward; travel time; and closest distance to an obstacle. Include the settings, seeds, and checkpoints you used.
4. Compare the two policies on the same navigation scenarios. Use the videos to show how they avoid obstacles and why representative runs succeed or fail.

## Answer These Questions in the PR

Answer each question briefly. Point to the relevant code, configuration, or test result.

1. **Graph input:** What are the robot, obstacle, and goal node features? The edge definition. Explain the input dimensions and main branch in this [code](https://github.com/anonymousrobotics9666/Reinforcement-Learning-Adaptive-CVaR-Barrier-Function), the difference
2. **GNN design:** descript or draw the GNN structure with the parameter setting, e.g., size of each layer .
3. **QP:** comapred with the original QP design of main branch in this [code](https://github.com/anonymousrobotics9666/Reinforcement-Learning-Adaptive-CVaR-Barrier-Function), what is the differnce, e.g., action limit, slack variable, constraint setting, how the qp related parameter ranges are set
4. **Reward:** explain reward design and related parameter setting.
5. **Output:** what is the output and how is it different from the final output of ppo mlp, explain the forward process and how the gradient information is passed
6. **record Results :** record your training results in a table like the paper does compare it with the diffcvar_mlp and ppo_mlp and save the corresponding video. explain the trajectory in the video.

## Submit Your Work

1. Fork the repository to your personal GitHub account.
2. Create a development branch in your fork.
3. Commit changes regularly with clear messages.
4. Open a PR from that branch into your fork’s `main`.
5. Answer every **Task 001** question in the PR description. Include code links, results, and videos.
6. Email us the PR link for review.

