# Task 002: Visual PPO for Crowd Navigation

## Goal

Add a `ppo_cnn` policy to the existing DiffCVaR crowd-navigation simulator. The actor must choose the robot's continuous action from only the latest **3 or 4 clean top-down images**. Compare it with the current state-based `ppo_mlp` policy: can the robot learn to reach the goal and avoid obstacles from images alone?

## Read First

- **This repository:** Read the README, then trace observations, actions, and PPO updates through `env/jax_social_nav.py`, `model/jax_ppo_mlp.py`, `model/jax_actor_critic.py`, `trainer/jax_ppo_rollout.py`, and `trainer/jax_ppo_eval.py`.
- **CleanRL:** Use [image PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py) for CNN and image-rollout ideas, [continuous-action PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py) for action sampling and PPO updates, and the [JAX image PPO example](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax_scan.py) for JAX structure. The image examples use **discrete** actions; adapt the ideas to this simulator's continuous actions.
- **JAX visual RL:** See [SERL's image-observation simulation guide](https://github.com/rail-berkeley/serl/blob/main/docs/sim_quick_start.md) for image-pipeline ideas. It is a different RL algorithm.
- **Optional:** [DrQ-v2](https://github.com/facebookresearch/drqv2) for image-based RL ideas and [SDPG](https://github.com/HaoxiangYou/SDPG) for visual-RL training efficiency. Neither is a drop-in PPO implementation for this task.



## What to Do

1. **Make policy images.** Render the robot, goal, and currently visible, active obstacles in a consistent top-down view. The pixels must not show past trajectories, future states, safety regions, QP diagnostics, text, or numerical simulator-state values. Do not use the diagnostic video renderer in `scripts/eval.py` as the policy observation. Generate a clean image at every training step and feed the latest few images to `ppo_cnn`. Save a few example images so we can check what the policy actually sees.
2. **Add** `ppo_cnn`**.** Stack the latest 3 or 4 frames, including correct initialization and reset at episode boundaries. Feed the image stack to a CNN actor that produces the same continuous robot action as `ppo_mlp`. The actor must never receive state vectors, positions, distances, or other ground-truth features outside the pixels.
3. **Test.** First check that the saved images show the right objects, hidden obstacles do not appear, the 3–4 frames are in order and reset between episodes, actions stay within their limits, and one training update runs with finite gradients. Then train `ppo_cnn` and `ppo_mlp` with the same environment settings, random seeds. Report success, collision, and timeout rates, total reward, travel time, and closest distance to an obstacle. Save videos of both policies on the same test scenarios, show the clean images seen by `ppo_cnn`, and explain representative successes and failures.



## Answer These Questions in the PR

Answer briefly and point to code, configuration, images, or results.

1. **Images:** explain  image size, channels, pixel range, coordinate view, and obstacle-visibility rule? Show sample frames and explain how you checked that forbidden information is absent.
2. **Frame history:** explain input shape, including batch and time dimensions. At the start of each episode, how do you fill the 3–4 image slots? Show that all old images are cleared when a new episode begins.
3. **CNN:** Draw or list the CNN layers and their sizes. What do the actor and critic receive and output?
4. **PPO:** explain your changes related to PPO, e.g., reward design, action bounds, or any parameter settings
5. **Results and videos:** Provide a results table for `ppo_cnn` and `ppo_mlp`, with seeds, training steps, settings, and checkpoint links. explain the trajectory in the video.



## Submit Your Work

1. Fork the repository to your personal GitHub account.
2. Create a development branch in your fork.
3. Commit changes regularly with clear messages.
4. Open a PR from that branch into your fork's `main`.
5. Answer every **Task 002** question in the PR description. Include code links, sample images, results, and videos.
6. Email us the PR link for review.

