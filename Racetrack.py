import numpy as np
import random
import matplotlib.pyplot as plt
from actions import *
import pygame
import sys


class Racetrack:
    # get the starting and ending blocks of track, initialize position and velocity of car
    def __init__(self, track):
        self.track = track
        # get a list of all the possible starting positions of the car
        self.start_positions = [(r, c) for r, row in enumerate(track) for c, cell in enumerate(row) if cell == 'S']
        self.finish_positions = [(r, c) for r, row in enumerate(track) for c, cell in enumerate(row) if cell == 'F']
        self.action_space = ACTION_SPACE
        self.reset()


    # reset car position to random and velocity to 0 0 and returns the state
    def reset(self):
        self.position = random.choice(self.start_positions)
        self.velocity = (0,0)
        return (self.position, self.velocity)


    # checks to see if car is in bounds and is not on a wall
    def is_valid_position(self, y, x):
        in_bounds = 0 <= y < len(self.track) and 0 <= x < len(self.track[0])
        not_wall = self.track[y][x] != '#' if in_bounds else False # need to bounds check first
        return in_bounds and not_wall


    # check to see if car is on or crosses finish line
    def crosses_finish_line(self, y, x, vy, vx):
        if [y,x] in self.finish_positions:
            return True

        # calculate intermediate positions to see if we cross finish
        steps = max(abs(vx), abs(vy))
        for i in range(1, steps + 1):
            intermediate_y = y + round(i * vy / steps)
            intermediate_x = x + round(i * vx / steps)
            if (intermediate_y, intermediate_x) in self.finish_positions:
                return True
        return False


    '''
    1. check for valid action
    2. apply changes to velocity, clip velocity
    3. check if we have crossed the finish line
    4. check if we are out of bounds
    5. update position
    '''
    def step(self, action):
        if action not in ACTION_SPACE:
            raise ValueError(f'Invalid action: {action}')

        # 10% chance no change
        if random.random() < 0.1:
            action = (0,0)

        self.velocity = (self.velocity[0] + action[0], self.velocity[1] + action[1])

        # clip speed
        max_speed = 5
        self.velocity = max(-max_speed, min(max_speed, self.velocity[0])), max(-max_speed, min(max_speed, self.velocity[1]))

        # check finish line
        if self.crosses_finish_line(self.position[0], self.position[1], self.velocity[0], self.velocity[1]):
            return (self.position, self.velocity), 100, True

        # check out of bounds
        new_y_position = (self.position[0] + self.velocity[0])
        new_x_position = (self.position[1] + self.velocity[1])

        if not self.is_valid_position(new_y_position, new_x_position):
            return (self.reset), -20, True

        # if all checks pass then udpate position
        self.position = (new_y_position, new_x_position)
        return (self.position, self.velocity), -1, False # impose small step penalty


    # prints out current position of car and track
    def print_track(self):
        track_copy = [list(row) for row in self.track]
        y, x = self.position
        track_copy[y][x] = 'C' 

        for row in track_copy:
            print(''.join(row))
        import pygame


class RacetrackVisualizer:
    def __init__(self, env):
        pygame.init()
        self.env = env
        self.cell_size = CELL_SIZE
        self.width = len(env.track[0]) * self.cell_size
        self.height = len(env.track) * self.cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Racetrack Simulator")

        self.clock = pygame.time.Clock()
        self.car_color = RED
        self.track_color = WHITE
        self.wall_color = BLACK
        self.start_color = BLUE
        self.finish_color = GREEN


    # go over the gameboard and draw appropriate colors
    def draw_track(self):
        for row in range(len(self.env.track)):
            for col in range(len(self.env.track[row])):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                if self.env.track[row][col] == "#":
                    pygame.draw.rect(self.screen, self.wall_color, rect)
                elif self.env.track[row][col] == "F":
                    pygame.draw.rect(self.screen, self.finish_color, rect)
                elif self.env.track[row][col] == "S":
                    pygame.draw.rect(self.screen, self.start_color, rect)
                else:
                    pygame.draw.rect(self.screen, self.track_color, rect)


    def draw_car(self):
        y, x = self.env.position
        car_rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.car_color, car_rect)


    # need completed agent to run
    def run_simulation(self, agent):
        state = self.env.reset()
        done = False
        step_count = 0

        while not done:
            step_count += 1
            self.screen.fill(WHITE)
            self.draw_track()
            self.draw_car()
            pygame.display.flip()

            action = agent.policy.get(state, (0, 0))  # Get action from policy or do nothing
            next_state, reward, done = self.env.step(action)

            print(f"Position: {state[0]}, Velocity = {state[1]}, Action: {action}, "
            f"reward: {reward}, Next state: {next_state[0]}, Step count: {step_count}")

            state = next_state
            self.clock.tick(2)
        pygame.quit()
        sys.exit()


# takes two space separated integers as input, used for testing
def get_user_action():
    while True:
        try:
            # take input and turn into int list
            action = tuple(map(int, input("enter vel command: ").split()))
            print('\n')
            return action
        except ValueError: print("Invalid input")

def visual_test():
    tiny_track = [
    "#####",                                       
    "#S  #",
    "#  F#",
    "#   #",
    "#####"
    ]

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
    print('start positions: ', env.start_positions)
    print('finish positions: ', env.finish_positions)
    done = False

    while not done:
        env.print_track()
        print('pos: ', env.position)
        print('vel: ', env.velocity)
        action = get_user_action()  # Get user action
        state, reward, done = env.step(action)

        print(f'Position: {env.position}')
        print(f'Velocity: {env.velocity}')
        print(f'reward: {reward}')
        print(done, "\n")

    # visual_test()
    tiny_track = [
    "#####",                                       
    "#S  #",
    "#  F#",
    "#   #",
    "#####"
    ]

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

    track2 = [
        "#############################",
        "#...........................#",
        "#...........................#",
        "#...........................#",
        "#...........................#",
        "#...........................#",
        "#...........................#",
        "#...........................#",
        "#...........................#",
        "#...........................#",
        "#........###########........#",
        "#........###########........#",
        "#........###########........#",
        "#........###########........#",
        "#........###########........#",
        "#........###########........#",
        "#........###########........#",
        "#........###########........#",
        "#SSSSSSSS###########FFFFFFFF#",
        "#############################"
    ]

    env = Racetrack(track2)

    visual = RacetrackVisualizer(env)

    visual.draw_track()

