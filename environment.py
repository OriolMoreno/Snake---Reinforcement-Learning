# importing libraries
import numpy as np
import pygame
import time
import random
from player import Player
import gym

SNAKE = 1
FRUIT = 2
WALLS = 3

SIZE_X = 25
SIZE_Y = 25

SPEED = 100  # Default = 20


def set_board():
    board = np.zeros((SIZE_X + 1, SIZE_Y + 1))
    board[0, :] = WALLS
    board[SIZE_X, :] = WALLS
    board[:, 0] = WALLS
    board[:, SIZE_Y] = WALLS
    board = board.flatten()

    return board


class Environment(gym.Env):
    # RL things
    # ENVIRONMENT_SHAPE = (72, 48, 1)
    # OBSERVATION_SHAPE = (72, 48, 1)
    observation_space = gym.spaces.Box(shape=(12,), low=0, high=1)
    action_space = gym.spaces.Discrete(len([0, 1, 2, 3, 4]))
    action_space_size = len([0, 1, 2, 3, 4])
    punishment = -100
    reward = 100
    score = 0

    def __init__(self):
        self.snake_speed = SPEED
        self.last_distance = 100

        # board
        self.board = set_board()

        # Window size
        self.window_x = SIZE_X * 10
        self.window_y = SIZE_Y * 10

        # defining colors
        self.black = pygame.Color(0, 0, 0)
        self.white = pygame.Color(255, 255, 255)
        self.red = pygame.Color(255, 0, 0)
        self.green = pygame.Color(0, 255, 0)
        self.blue = pygame.Color(0, 0, 255)
        self.light_pink = pygame.Color(248, 171, 186)

        # Player
        self.player = Player()
        self.game_over = False
        self.game_window = pygame.display.set_mode((self.window_x, self.window_y))

        # FPS (frames per second) controller
        self.fps = pygame.time.Clock()

        # fruit posiiton
        self.fruit_position = self.spawn_fruit()

        self.fruit_spawn = True
        self.action = 0
        self.currentReward = 0

        # Define Array of State as the following: [Snake go left, right, up, down, Food left, right, up, down,
        # Danger in front, left, right, On fruit]
        self.state = np.zeros((12,))

        # setting default snake direction
        self.direction = 'RIGHT'
        self.state[1] = 1

        # Add things to env board
        self.add_to_board()

        # Initialising pygame
        pygame.init()

        # Initialise game window
        pygame.display.set_caption('Snake')

    def spawn_fruit(self):
        fruit_position = [random.randrange(1, (self.window_x // 10)) * 10,
                          random.randrange(1, (self.window_y // 10)) * 10]
        while fruit_position in self.player.snake_body:
            fruit_position = [random.randrange(1, (self.window_x // 10)) * 10,
                              random.randrange(1, (self.window_y // 10)) * 10]

        return fruit_position

    def add_to_board(self):
        self.board[(self.fruit_position[0] // 10) * SIZE_Y + self.fruit_position[1] // 10] = FRUIT
        for position in self.player.snake_body:
            self.board[(position[0] // 10) * SIZE_Y + position[1] // 10] = SNAKE

    def step(self, action=None):
        self.state = np.zeros((12,))
        got_fruit = False
        currentReward = -1
        self.fruit_spawn = True
        if action is not None:
            if isinstance(action, np.ndarray):
                bestAction = np.argmax(action)
            else:
                bestAction = action
            change_to = self.direction
            if bestAction == 0:
                if self.direction == 'UP':
                    self.player.snake_position[1] -= 10
                    self.state[2] = 1
                if self.direction == 'DOWN':
                    self.player.snake_position[1] += 10
                    self.state[3] = 1
                if self.direction == 'LEFT':
                    self.player.snake_position[0] -= 10
                    self.state[0] = 1
                if self.direction == 'RIGHT':
                    self.player.snake_position[0] += 10
                    self.state[1] = 1
            else:
                if bestAction == 1:
                    change_to = 'UP'
                if bestAction == 2:
                    change_to = 'DOWN'
                if bestAction == 3:
                    change_to = 'LEFT'
                if bestAction == 4:
                    change_to = 'RIGHT'

                if change_to == 'UP' and self.direction != 'DOWN':
                    self.direction = 'UP'
                    self.state[2] = 1
                if change_to == 'DOWN' and self.direction != 'UP':
                    self.direction = 'DOWN'
                    self.state[3] = 1
                if change_to == 'LEFT' and self.direction != 'RIGHT':
                    self.direction = 'LEFT'
                    self.state[0] = 1
                if change_to == 'RIGHT' and self.direction != 'LEFT':
                    self.direction = 'RIGHT'
                    self.state[1] = 1

                if self.direction == 'UP':
                    self.player.snake_position[1] -= 10
                if self.direction == 'DOWN':
                    self.player.snake_position[1] += 10
                if self.direction == 'LEFT':
                    self.player.snake_position[0] -= 10
                if self.direction == 'RIGHT':
                    self.player.snake_position[0] += 10

        self.player.snake_body.insert(0, list(self.player.snake_position))

        if self.player.snake_position[0] == self.fruit_position[0] and self.player.snake_position[1] == \
                self.fruit_position[1]:
            got_fruit = True
            self.state[11] = 1
            currentReward += self.reward
            self.fruit_spawn = False
            self.fruit_position = self.spawn_fruit()
        else:
            self.player.snake_body.pop()

        # Game Over conditions
        if self.player.snake_position[0] < 0 or self.player.snake_position[0] > self.window_x - 10:
            self.game_over = True
        if self.player.snake_position[1] < 0 or self.player.snake_position[1] > self.window_y - 10:
            self.game_over = True

        # Touching the snake body
        for block in self.player.snake_body[1:]:
            if self.player.snake_position[0] == block[0] and self.player.snake_position[1] == block[1]:
                self.game_over = True

        # Where is the food
        if self.fruit_position[0] > self.player.snake_position[0]:
            self.state[5] = 1
        if self.fruit_position[0] < self.player.snake_position[0]:
            self.state[4] = 1
        if self.fruit_position[1] > self.player.snake_position[1]:
            self.state[7] = 1
        if self.fruit_position[1] < self.player.snake_position[1]:
            self.state[6] = 1

        # If not gameover calculate reward
        if not self.game_over:
            self.board = set_board()
            # Add things to env board
            if not got_fruit:
                self.board[(self.fruit_position[0] // 10) * SIZE_Y + self.fruit_position[1] // 10] = FRUIT
            for position in self.player.snake_body:
                self.board[(position[0] // 10) * SIZE_Y + position[1] // 10] = SNAKE

            distance = 0
            reshaped = np.reshape(self.board, (SIZE_X + 1, SIZE_Y + 1))
            res = np.where(reshaped == FRUIT)
            player_pos = [self.player.snake_position[0] // 10, self.player.snake_position[1] // 10]
            if res[0]:
                distance = np.abs(player_pos[0] - res[0][0]) + np.abs(player_pos[1] - res[1][0])
            if distance < self.last_distance:
                currentReward += 3
            else:
                currentReward -= 3

            self.last_distance = distance

            # Danger near?
            reshaped = np.transpose(np.reshape(self.board, (SIZE_X + 1, SIZE_Y + 1)))
            front = 0
            left = 0
            right = 0
            pos = self.player.snake_position[0] // 10, self.player.snake_position[1] // 10

            if self.direction == 'UP':
                front = [pos[0], pos[1] - 1]
                left = [pos[0] - 1, pos[1]]
                right = [pos[0] + 1, pos[1]]
            elif self.direction == 'DOWN':
                front = [pos[0], pos[1] + 1]
                left = [pos[0] + 1, pos[1]]
                right = [pos[0] - 1, pos[1]]
            elif self.direction == 'LEFT':
                front = [pos[0] - 1, pos[1]]
                left = [pos[0], pos[1] + 1]
                right = [pos[0], pos[1] - 1]
            elif self.direction == 'RIGHT':
                front = [pos[0] + 1, pos[1]]
                left = [pos[0], pos[1] - 1]
                right = [pos[0], pos[1] + 1]

            if reshaped[front[0], front[1]] != 0 and reshaped[front[0], front[1]] != FRUIT:
                self.state[8] = 1
            if reshaped[left[0], left[1]] != 0 and reshaped[front[0], front[1]] != FRUIT:
                self.state[9] = 1
            if reshaped[right[0], right[1]] != 0 and reshaped[front[0], front[1]] != FRUIT:
                self.state[10] = 1

        done = False
        if self.game_over:
            currentReward += self.punishment
            self.score += currentReward
            done = True
            print('Game Over, Score:', self.score)

        self.score += currentReward
        return self.state, currentReward, done, {}

    def reset(self):
        self.player = Player()
        self.score = 0
        self.game_over = False
        self.last_distance = 100

        # fruit posiiton
        self.fruit_position = self.spawn_fruit()
        self.fruit_spawn = True
        self.action = 0

        self.board = set_board()
        # Add things to env board
        self.add_to_board()

        # setting default snake direction
        self.direction = 'RIGHT'
        self.state = np.zeros((12,))
        self.state[1] = 1

        return self.state

    # displaying Score function
    def show_score(self, choice, color, font, size):
        # creating font object score_font
        score_font = pygame.font.SysFont(font, size)

        # create the display surface object
        # score_surface
        score_surface = score_font.render('Score : ' + str(self.score), True, color)

        # create a rectangular object for the text
        # surface object
        score_rect = score_surface.get_rect()

        # displaying text
        self.game_window.blit(score_surface, score_rect)

    # game over function
    def game_over_func(self):
        # creating font object my_font
        my_font = pygame.font.SysFont('times new roman', 50)

        # creating a text surface on which text
        # will be drawn
        game_over_surface = my_font.render(
            'Your Score is : ' + str(self.score), True, self.red)

        # create a rectangular object for the text
        # surface object
        game_over_rect = game_over_surface.get_rect()

        # setting position of the text
        game_over_rect.midtop = (self.window_x / 2, self.window_y / 4)

        # blit wil draw the text on screen
        self.game_window.blit(game_over_surface, game_over_rect)
        pygame.display.flip()

        self.currentReward = self.punishment

        self.score += self.punishment

    def render(self, window=None, human=False, mode=''):
        # Main Function

        if human:
            change_to = self.direction

            # handling key events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        change_to = 'UP'
                    if event.key == pygame.K_DOWN:
                        change_to = 'DOWN'
                    if event.key == pygame.K_LEFT:
                        change_to = 'LEFT'
                    if event.key == pygame.K_RIGHT:
                        change_to = 'RIGHT'

            self.step()

        pygame.event.get()

        self.game_window.fill(self.black)

        for pos in self.player.snake_body:
            pygame.draw.rect(self.game_window, self.light_pink,
                             pygame.Rect(pos[0], pos[1], 10, 10))
        pygame.draw.rect(self.game_window, self.white, pygame.Rect(
            self.fruit_position[0], self.fruit_position[1], 10, 10))

        # Game Over conditions

        # displaying score countinuously
        self.show_score(1, self.white, 'times new roman', 20)

        # Refresh game screen
        pygame.display.update()

        # Frame Per Second / Refresh Rate
        self.fps.tick(self.snake_speed)
