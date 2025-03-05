# Monte Carlo Reinforcement Learning for Racetrack Simulation

## Author
**Ben Lam** (📧 [bach.lam@tufts.edu](mailto:bach.lam@tufts.edu))

## Introduction
This project implements a **Monte Carlo Reinforcement Learning (RL)** agent to navigate a racetrack. The goal is for the agent to reach the finish line as efficiently as possible while minimizing the number of moves. 

After **20,000** rounds of training, the agent successfully learns a near-optimal policy using the **Exploring Starts (ES) Monte Carlo** approach. The trained agent completes:
- **Small Track:** ~6 moves
- **Large Track:** ~17 moves

## Methodology

### Racetrack Environment
The environment is a **grid-based racetrack** with the following elements:
- `S`: Possible **starting positions** (blue squares).
- `F`: **Finish line** squares (green squares).
- `#`: **Walls** that the car **cannot** pass through (black squares).
- `.`: **Valid track** spaces (white squares).

Two racetrack layouts are used in this project:
1. **Small Track** (16 × 14)
2. **Large Track** (30 × 20)

The large track presents a greater challenge due to an additional sharp right turn.

### Action Space & Objectives
- The car starts with **velocity (0,0)**.
- Each action modifies the velocity using **(ay, ax) ∈ {−1, 0, 1}**, allowing **9 possible actions** at every step.
- The velocity updates as:  
  **v_new = v_old + a**  
  with a **speed cap of ±5** in both x and y directions.
- There is a **10% probability** that the agent is forced into action (0,0), meaning no movement occurs.
- **Rewards:**
  - **+100** for crossing the finish line.
  - **-20** for crashing into walls or going out of bounds.
  - **-1** for each step taken (step penalty).

## Implementation

### Monte Carlo Agent
The agent uses the **Exploring Starts Monte Carlo** method with:
- **Q-table** for state-action value estimates.
- **ϵ-greedy policy** for decision-making.
- **First-visit Monte Carlo updates** for Q-value improvements.

### Training Process
1. Reset environment and initialize state.
2. Run an episode where the agent follows its learned policy or explores.
3. Calculate **total discounted return (G)** at each timestep.
4. Update Q-table using the mean of observed returns (for first visits only).
5. Adjust the policy based on updated Q-values.

### Hyperparameters
- **Epsilon (ϵ):** 0.1  
- **Gamma (γ):** 0.9  
  _(These values remained constant throughout the experiment.)_

## Results
After **20,000** rounds:
- **Completion Rate:**
  - **Small Track:** ~80%
  - **Large Track:** ~70%  
  _(Completion rate is affected by the 10% probability of forced (0,0) actions.)_
- The agent completes:
  - **Small Track:** ~6 moves  
  - **Large Track:** ~17 moves  
- The trained policy can be visualized in **real time** by uncommenting the last four lines in `monte_carlo_agent.py`.

### Reward Trends
Every **1,000** training episodes, the agent's **average reward** and **completion rate** were recorded.

| Agent  | Track  | Average Reward | Completion Rate |
|--------|--------|---------------|----------------|
| Agent 1 | Small  | **Higher** (shorter track) | **Faster learning** |
| Agent 2 | Large  | **Lower** (longer track, step penalty) | **Slower learning** |

## Discussion
While Monte Carlo learning proved effective for this task, its limitation is that it requires the agent to **complete episodes before updating the policy**. This means that for some problems, where episodes take too long or may never complete, **Monte Carlo methods may not be feasible**.

In such cases, alternative **online learning methods** that update after every step (instead of after each episode) could be more effective. Examples include:
- **Temporal Difference (TD) Learning** (e.g., **SARSA**, **Q-Learning**)
- **Deep Q-Networks (DQN)** for complex environments

However, for this task, Monte Carlo learning performed **well**, and the agent's performance was **satisfactory**.

## Conclusion
This project successfully implemented a Monte Carlo RL agent for racetrack navigation. The results demonstrate that the agent:
- **Learns an efficient policy**
- **Adapts to different track complexities**
- **Shows clear improvement over training episodes**

### Future Work
- **Implement TD-learning** methods such as **SARSA** or **Q-Learning**.
- **Use function approximation** (e.g., **Neural Networks**) instead of Q-tables.
- **Test on more complex racetrack layouts**.

## Running the Code
To visualize the trained policy:
1. Ensure `monte_carlo_agent.py` is in your working directory.
2. Uncomment the last **four lines** in `monte_carlo_agent.py`.
3. Run:
   ```bash
   python monte_carlo_agent.py


## Further Inqueries
For further questions or comments regarding this project, feel free to reach out to me at (bach.lam@tufts.edu)!