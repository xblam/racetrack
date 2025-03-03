import numpy as np
import random
import matplotlib.pyplot as plt


ACTION_SPACE = [(up_change, right_change) for up_change in [-1, 0, 1] for right_change in [-1, 0, 1]]

class Racetrack:
    # get the starting and ending blocks of track, initialize position and velocity of car
    def __init__(self, track):
        self.track = track
        # get a list of all the possible starting positions of the car
        self.start_positions = [[r, c] for r, row in enumerate(track) for c, cell in enumerate(row) if cell == 'S']
        self.finish_positions = [[r, c] for r, row in enumerate(track) for c, cell in enumerate(row) if cell == 'F']

        self.reset()




    # reset car position to random and velocity to 0 0
    def reset(self):
        self.position = random.choice(self.start_positions)
        self.velocity = [0,0]

        return self.position, self.velocity


    # checks to see if car is in bounds and is not on a wall
    def is_valid_position(self, x, y):
        in_bounds = 0 <= x < len(self.track) and 0 <= y < len(self.track[0])
        not_wall = self.track[x][y] != '#' if in_bounds else False # need to bounds check first

        return in_bounds and not_wall




    # for checking if we have crossed the finish line it is a bit more tricky,
    # we have to make sure that none of the intermediate positions are on the finish line
    def crosses_finish_line(self, y, x, vy, vx):

        # first check if we are on the finish line
        if [y,x] in self.finish_positions:
            return True

        # then check to see if we have overshot the finish
        steps = max(abs(vx), abs(vy))
        for i in range(1, steps + 1):  # check each intermediate position
            intermediate_y = y + round(i * vy / steps)
            intermediate_x = x + round(i * vx / steps)
            if (intermediate_y, intermediate_x) in self.finish_positions:
                return True  # finish line detected

        return False  # no crossing detected




    # next we need to determine what happens when we take a step
    # for each step we will have to return the state, the velocity, the reward, and whether or not we are done
    def step(self, action):
        # check for funny business
        if action not in ACTION_SPACE:
            raise ValueError(f'Invalid action: {action}')

        # there will be a 10% chance that the car will not accelerate
        if random.random() < 0.1:
            action = (0,0)

        # apply changes to velocity
        self.velocity[0] += action[0]
        self.velocity[1] += action[1]

        # enforce speed limits
        max_speed = 5
        self.velocity[0] = max(-max_speed, min(max_speed, self.velocity[0]))
        self.velocity[1] = max(-max_speed, min(max_speed, self.velocity[1]))

        # check if we have crossed the finish line
        if self.crosses_finish_line(self.position[0], self.position[1], self.velocity[0], self.velocity[1]):
            print('FINSIH')
            return self.position, self.velocity, 100, True

        # check head to see if we are out of bounds
        new_y_position = self.position[0] - self.velocity[0] # minus to make it more intuitive what direction we are going in
        new_x_position = self.position[1] + self.velocity[1]

        if not self.is_valid_position(new_y_position, new_x_position):
            print("POSITION INVALID")
            position, velocity = self.reset()
            return position, velocity, -20, False

        # then if we are still in play then we will update the position
        self.position = [new_y_position, new_x_position]
        return self.position, self.velocity, -1, False # impose small step penalty

        


def testing_function():
    # simple track with right turn 14 x 16
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

    # checking init
    env = Racetrack(tiny_track)
    print('starting positions: ', env.start_positions)
    print('finishing positions: ', env.finish_positions)

    # # checking reset
    # state = env.reset()
    # print(f'starting state: {state}')

    # # checking valid positions
    # print(env.track[1][1])
    # print(env.track[1][0])
    # print(env.track[12][1])
    # print(env.track[1][14])
    # print(env.is_valid_position(1, 1)) # road
    # print(env.is_valid_position(1, 0)) # wall
    # print(env.is_valid_position(12, 1)) # start block
    # print(env.is_valid_position(1, 14)) # end block
    # print(env.is_valid_position(1, 19)) # out of bounds

    # checking action space
    print(env.action_space)
    print(env.velocity)
    print('pos', env.position)
    env.step((-1,1))
    print(env.velocity)
    print('pos', env.position)
    env.step((1,1))
    print(env.velocity)
    print('pos', env.position)
    env.step((1,1))
    print(env.velocity)
    print('pos', env.position)

    # test step function

def print_track(env):
    track_copy = [list(row) for row in env.track]  # Create a copy of the track
    y, x = env.position
    track_copy[y][x] = 'C' 

    # Print the updated racetrack
    for row in track_copy:
        print(''.join(row))





# takes two space separated integers as input
def get_user_action():
    while True:
        try:
            # take input and turn into int list
            action = tuple(map(int, input("enter vel command: ").split()))
            print('\n')
            if action in ACTION_SPACE:
                return action
            else:
                print("Invalid action")
        except ValueError:
            print("Invalid input")

def visual_test():
    tiny_track = [
    "#####",
    "#S  #",
    "#  F#",
    "#   #",
    "#####"
    ]

    env = Racetrack(tiny_track) 
    print(env.finish_positions)

    print_track(env)

    done = False
    print(done)

    while not done:
        action = get_user_action()  # Get user action
        position, velocity, reward, done = env.step(action)

        print_track(env)  # Update track with new position
        print(f'Position: {env.position}')
        print(f"Velocity: {env.velocity}")
        print(f'reward: {reward}')
        print(done, "\n")




if __name__ == "__main__":
    visual_test()

