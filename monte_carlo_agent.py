import numpy as np
import random
from collections import defaultdict
from racetrack import Racetrack
from actions import *
import pickle

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
        # if state unknown or under epsilon, chose random action
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
            next_state, reward, done = self.env.step(action) # take a step
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
            
            if (state, action) not in visited: # if not visited yet then we will update q value
                self.returns[(state, action)].append(G)
                # if it is your first time we take average of observed rewards
                self.Q[state][ACTION_TO_INDEX[action]] = np.mean(self.returns[(state, action)])
                visited.add((state, action)) # mark as visited

        # and then we will just return total reward in episode
        return {"total_reward": sum(reward for _, _, reward in episode), "length": len(episode)}



    # this is just so that we can run the best model after training
    def improve_policy(self):
        for state in self.Q:
            best_action_idx = np.argmax(self.Q[state])
            self.policy[state] = ACTION_SPACE[best_action_idx]



    def train(self, episodes=10000):
        rewards = []
        for episode_num in range(episodes):
            episode = self.generate_episode()
            stats = self.update_q_values(episode) # update q table with episode
            self.improve_policy() # improve policy

            rewards.append(stats["total_reward"])
            
            # lets us see how the model is doing
            if episode_num % 1000 == 0:
                avg_reward = np.mean(rewards[-1000:])
                print(f"Episode {episode_num}/{episodes}: Avg Reward = {avg_reward:.2f}")

    def simulate(self, render=True):
        """Runs the trained agent using the learned policy and returns the path taken."""
        state = self.env.reset()  # Start at a random position
        done = False
        path = [state]  # Track the states visited

        while not done:
            action = self.policy.get(state, random.choice(ACTION_SPACE))  # Use learned policy
            next_state, _, done = self.env.step(action)  # Take step
            path.append(next_state)  # Store the new state
            state = next_state  # Move to the next state

            if render:
                self.env.render()  # ✅ Optional: Print track visualization (if implemented)

        return path  # Return the full path taken by the agent
    


    # save model so we dont have to rerun all the time
    def save_model(self, filename="monte_carlo_model.pkl"):
        model_data = {
            "Q": dict(self.Q),  # Convert defaultdict to a normal dict
            "policy": self.policy
        }
        with open(filename, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")


    # Load saved model
    def load_model(self, filename="monte_carlo_model.pkl"):
        with open(filename, "rb") as f:
            model_data = pickle.load(f)
        self.Q = defaultdict(lambda: np.zeros(len(ACTION_SPACE)), model_data["Q"])
        self.policy = model_data["policy"]
        print(f"Model loaded from {filename}")


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
        agent = MonteCarloAgent(env)

        agent.train(10000)
        agent.save_model()
        
        agent.load_model()
        print(agent.simulate(False))




