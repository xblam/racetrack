import numpy as np
import random
from collections import defaultdict
from racetrack import Racetrack
from actions import *
from pprint import pprint



class MonteCarloAgent:
    def __init__(self, env, gamma=0.9, epsilon=0.1, episodes=10000):
        # initialize values
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.Q = defaultdict(lambda: np.zeros(len(ACTION_SPACE)))
        self.returns = defaultdict(list) # keep track of returns for each state-action pair
        self.policy = {} # we will use this to keep track of the best move for each state, use to fast lookup



    def get_action(self, state):
        # if state unknown or random, chose random action
        if state not in self.Q or np.random.rand() < self.epsilon:
            return random.choice(ACTION_SPACE)

        # otherwise chose best action
        return ACTION_SPACE[np.argmax(self.Q[state])]



    def generate_episode(self):
        episode = [] # stores the states, actions, and rewards for each step
        state = self.env.reset() # reset the state
        done = False
        num_steps = 0

        # loop until the episode ends
        while not done and num_steps < 1000:
            num_steps += 1
            action = self.get_action(state)  # get an action
            next_state, action, reward, done = self.env.step(action) # take a step
            episode.append((state, action, reward)) # previous state will be paired with current action
            state = next_state # then we move on to the next state

        return episode # return all the states and the corresponding actions and rewards


    # once the episode is done we will calculate the reward for each step in episode
    def update_q_values(self, episode):
        G = 0
        visited = set()

        # starting from the back we will calculate the reward for each step (with discount)
        for state, action, reward in reversed(episode): 
            G = self.gamma * G + reward
            
            if (state, action) not in visited: # if not visited yet thne we will update q value
                self.returns[(state, action)].append(G)
                self.Q[state][ACTION_TO_INDEX[action]] = np.mean(self.returns[(state, action)])
                visited.add((state, action))  # Mark as visited
    
    
    
    
    def update_q_values(self, episode, alpha=0.1):
        """
        Updates the Q-values using First-Visit Monte Carlo method.
        - `alpha`: Learning rate for incremental updates (default=0.1)
        """
        G = 0  
        visited = set()

        # calculate reward for each step in reverse
        for state, action, reward in reversed(episode):  
            G = self.gamma * G + reward  # Compute return

            if (state, action) not in visited:  # first visit update
                if (state, action) not in self.returns:
                    self.returns[(state, action)] = []  # Initialize if missing
            
                if state not in self.Q:
                    self.Q[state] = np.zeros(len(ACTION_SPACE))  # Initialize Q-values

                # Store return for averaging
                self.returns[(state, action)].append(G)

                # Compute Q-value update
                if alpha > 0:  # Incremental update
                    self.Q[state][ACTION_TO_INDEX[action]] += alpha * (G - self.Q[state][ACTION_TO_INDEX[action]])
                else:  # Full averaging
                    self.Q[state][ACTION_TO_INDEX[action]] = np.mean(self.returns[(state, action)])

                visited.add((state, action))  # Mark as visited

        return {"total_reward": sum(r for _, _, r in episode), "length": len(episode)}


    return {"total_reward": sum(r for _, _, r in episode), "length": len(episode)}


    def improve_policy(self):
        """Updates the policy based on the learned Q-values."""
        for state in self.Q:
            best_action_idx = np.argmax(self.Q[state])
            self.policy[state] = ACTION_SPACE[best_action_idx]  # Store best action

    def train(self, episodes=10000):
        """Runs Monte Carlo Control to train the agent."""
        for episode_num in range(episodes):
            episode = self.generate_episode()  # Generate episode
            self.update_q_values(episode)  # Update Q-table
            self.improve_policy()  # Improve policy

            if episode_num % 1000 == 0:
                print(f"Training Progress: {episode_num}/{episodes} episodes")

    def simulate(self):
        """Runs the agent using the learned policy."""
        state = self.env.reset()
        done = False
        path = [state]

        while not done:
            action = self.policy.get(state, random.choice(ACTION_SPACE))  # Use policy
            next_state, _, _, done = self.env.step(action)
            path.append(next_state)
            state = next_state

        return path  # Return path for visualization



if __name__ == "__main__":
    track1 = [
        "################",
        "#.............F#",
        "#.............F#",
        "#.............F#",
        "#.............F#",
        "#.............F#",
        "#.............F#",
        "#......#########",
        "#......#########",
        "#......#########",
        "#......#########",
        "#......#########",
        "#SSSSSS#########",
        "################"
    ]

    env = Racetrack(track1) 
    # print(action_to_index((0, 0)))
    # print(index_to_action(4))
    agent = MonteCarloAgent(env)
    state, action, reward, done = agent.env.step((0, 0))

    pprint(agent.Q)