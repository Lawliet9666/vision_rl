# Task 004: End-to-End Sensor-Based PPO with a Differentiable CBF-QP

## Goal

**Task 003:** Use a separate perception module to convert sensor information into geometric information so that the RL policy network's observations remain unchanged. In Task 004, extend [Task 003](003_visual_ppo_cbf.md) into an **end-to-end sensor-to-action policy** using **either LiDAR or depth/RGB-D images**. Learn features from sensor history to output a nominal action and per-obstacle CBF gains. You may need to extract geometric information from sensor data to calculate QP constraints.


## What to Do

### 1. Preliminaries: Understand the CBF Constraint

Start with Task 003's circle barrier and fixed combined radius:

$$
h_i=\|q-p_i\|^2-R_i^2,
\qquad \dot h_i+\kappa_i h_i\geq-s_i.
$$

First, derive $\dot h_i$ using robot dynamics and obstacle velocity, and then write a linear inequality in the QP control variables. You can start with a single-integrator model for both the robot and obstacles.

List the required information and identify measured, known, and estimated terms.
Check one static-obstacle example and one moving-obstacle example, including control bounds.

### 2. E2E Policy Training: Learn Actions from Sensor Observations
LiDAR scans or RGB-D images can directly provide distance information, but that is not enough for the CBF-QP.
If you understand how CBFs work, you will see that we also need obstacle velocities. The simplest approach is to reuse Task 003's velocity estimator. After working out the CBF-QP part, you can implement the RL policy's mapping from observations to actions.

Implement **one** of these simple examples (you can start with LiDAR because it is simpler, but we will eventually need image information to understand semantic safety, e.g., to predict risks involving humans):

#### 2.1
```text
LiDAR history + robot state/goal
               ↓
      shared MLP hidden layers
               ↓
       action head + gain head
```
#### 2.2
```text
Recent depth images or RGB frames from an RGB-D camera
               ↓
      same CNN applied to each frame
               ↓
concatenate features + robot state/goal
               ↓
       MLP -> action head + gain head
```

Retain Task 003's **bounded positive gain $\kappa_i$ for each selected obstacle**, using the same bounds. Align gain outputs with the QP's obstacle ordering and masks.

Notes:
1. For a more complex design, you may consider GRUs, Transformers, and pretrained encoders, but these are optional because they can be computationally demanding.
2.  Preserve Task 003's dynamics, nearest-K selection, masks, fixed radii, slack penalties, control bounds, and PPO action-distribution handling.
### 3. Compare and Explain

Compare Task 003's perception + geometry-based diff-cbf-qp policy with Task 004's end-to-end diff-cbf-qp policy using the same sensor and exactly the same settings.

Report success, collisions, timeouts, clearance, QP failures/slack, and runtime, and include a demonstration figure.

# 4. Run Task 003 and Task 004 in MuJoCo 4.0

## Reading Materials (Optional)
Modular method: perception + geometry-based policy learning (image input)
1.  **[Xiao et al., Figure 2 and Sections V-VI-B](https://arxiv.org/html/2203.02401):** Understand images -> physical estimates/reference controls/gains -> differentiable CBF-QP. Their training is supervised; retain our PPO framework.

**Alternative ways to construct the barrier from sensors:**

2. **[Depth-CBF, Section III](https://hybrid-robotics.berkeley.edu/publications/ICRA2024_Point_Cloud_CBF.pdf):** Consider replacing our circle-based CBF with a barrier fitted from LiDAR/depth point clouds. Understand how to obtain $h$ and its gradient and construct the QP constraint.

3. **[Neural CBFs for Safe Navigation, Sections IV-V](https://arxiv.org/html/2407.19907):** Consider learning $h_\theta(o,x)$ from LiDAR observations and robot state. Understand how the network is trained, how its gradient enters the QP, and how observation updates are handled.

These are alternative barrier constructions, not ready-made PPO-through-QP implementations. Implementing either is optional after the baseline works.

## Answer These Questions in the PR

1. **Constraint:** Derive $\dot h_i$, write the QP, and show the derivation steps.
2. **Perception:** Show how you process LiDAR or RGB-D observations and obtain the geometric information required by the QP. What is the input to the CBF-QP layer? What is the RL action?
3. **Network and training:** Provide a network diagram and describe the architecture.
4. **Results:** Compare Task 003's geometry-input policy with Task 004's sensor-input policy using the same sensors, perception module, QP settings, environments, seeds, and training budgets. Report success, collisions, timeouts, clearance, QP failures/slack, runtime, and nominal-action constraint violations or QP interventions. Include representative success/failure videos and explain the main differences.

## Submit Your Work

Continue in your existing fork. Open a PR into your fork's `main`, answer every **Task 004** question, and include code/configuration links, results, and videos. Email us the PR link for review.
