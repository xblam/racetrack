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
        self.returns = defaultdict(list)
        self.policy = {}



    def generate_episode(self):
        """
        Generates an episode using the current policy with exploring starts.
        """
        episode = []
        state = self.env.reset()

        # Exploring starts: Random velocity from start state
        start_velocity = (random.randint(-1, 1), random.randint(-1, 1))
        self.env.velocity = list(start_velocity)

        done = False
        while not done:
            action = self.get_action(state)
            next_state, velocity, reward, done = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state

        return episode

    def update_q_values(self, episode):
        """
        Updates Q-values using the returns from an episode.
        """
        G = 0  # Return (discounted sum of rewards)
        visited = set()  # Track (state, action) pairs visited

        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward  # Compute return
            if (state, action) not in visited:  # First-visit MC
                self.returns[(state, action)].append(G)
                self.Q[state][ACTION_SPACE.index(action)] = np.mean(self.returns[(state, action)])
                visited.add((state, action))

    def get_action(self, state):
        """
        Selects an action using the ε-greedy policy.
        """
        if state not in self.policy or random.random() < self.epsilon:
            return random.choice(self.env.action_space)  # Explore

        return self.policy[state]  # Exploit

    def improve_policy(self):
        """
        Updates the policy based on the learned Q-values.
        """
        for state in self.Q:
            best_action_idx = np.argmax(self.Q[state])
            self.policy[state] = self.env.action_space[best_action_idx]

    def train(self):
        """
        Runs Monte Carlo control for a specified number of episodes.
        """
        for _ in range(self.episodes):
            episode = self.generate_episode()
            self.update_q_values(episode)
            self.improve_policy()

    def simulate(self):
        """
        Simulates a race using the learned policy.
        """
        state = self.env.reset()
        done = False
        path = [state]

        while not done:
            action = self.get_action(state)
            state, _, _, done = self.env.step(action)
            path.append(state)

        return path


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

    pprint(agent.Q)