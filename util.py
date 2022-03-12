'''
This file keeps track of the q learner
'''

__author__ = 'Cody Vollrath'
__version__ = 'Spring 2022'
__pylint__ = 'v2.12.2'

from tracemalloc import start
from turtle import pos
import random

class q_learner:
    '''
    Keeps track of the q learning stuff
    '''
    def __init__(self, alpha = 0, gamma = 0, epsilon = 0):
        '''
        Creates an instance of the q learner
        '''
        
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.moves_tracker = []
        self.visited = {}
        self.__create_map__()
    
    def __create_map__(self):
        world = [0] * 50
        self.reward_indices = {
            6: -10, 7: -10, 10: -10, 13: 25, 14: -10,
            24: -10000, 25: -10, 26: -10, 28: -10, 34: -10000, 
            38: -10, 42: -10, 44: -10000, 47: 1500, 48: -10}
        
        mappings = {}
        for index in range(len(world)):
            position = index + 1
            self.visited[position] = 0
            if position > 10:
                mappings[(position, 'N')] = 0
            if position % 10 != 1:
                mappings[(position, 'W')] = 0
            if position % 10 != 0:
                mappings[(position, 'E')] = 0
            if position <= 40:
                mappings[(position, 'S')] = 0
        self.q_table = mappings
    
    def train(self, epochs = 100):
        old_ep = self.epsilon
        if epochs >= 100_000:
            self.epsilon = 0.9
        for epoch in range(epochs):
            self.transition_function(31)
            if epoch % 1000 == 0:
                if self.epsilon >= old_ep:
                    self.epsilon -= 0.001
                else:
                    self.epsilon = old_ep
                print(epoch + 1)
                print(self.moves_tracker)
            self.moves_tracker = []
    
    def final_run(self):
        self.visited.fromkeys(self.visited, 0)
        return self.transition_function(training_mode=False)
        
    def transition_function(self, position = 31, training_mode = True):
        #add first position to moves tracker
        self.moves_tracker.append(position)

        #if the current reward of the position is 1500 (goal) or -100 (death) then end the learning
        while True:
            self.visited[position] += 1
            #get the next move based on the greedy policy
            move = self.greedy_policy(position, training_mode)
            # print(f'The move that will be made: {move}')
            #get next state from move
            next_state = self.get_next_state(move)
            #update the q table based on Belmont's formula
            self.q_table[move] = (1 - self.alpha) * self.q_table[move] + self.alpha * (self.__get_reward__(position) + self.gamma * self.argmax(next_state)) #get the max value of the direction

            #set the position to the next move
            position = next_state
            
            #add the new position to the moves tracker
            if self.__get_reward__(move[0]) == 1500 or self.__get_reward__(move[0]) == -10000:
                break
            self.moves_tracker.append(position)
            
        return self.moves_tracker

    def greedy_policy(self, position, training_mode):
        possible_positions = self.__find_cardinal_directions__(position)
        best_move = self.get_best_move(possible_positions)

        random_number = random.random()
        if random_number > self.epsilon and self.q_table[best_move] != 0:
            # print(f'Best move is: {best_move}')
            return best_move
        random_move = random.choice(possible_positions)
        # print(f'RANDOM move is: {random_move}')
        return random_move

    def __find_cardinal_directions__(self, position):
        applicable_cardinal_directions = []
        if position > 10:
            applicable_cardinal_directions.append((position, "N"))
        if position % 10 != 1:
            applicable_cardinal_directions.append((position, "W"))
        if position % 10 != 0:
            applicable_cardinal_directions.append((position, "E"))
        if position <= 40:
            applicable_cardinal_directions.append((position, "S"))
        return applicable_cardinal_directions
    
    def argmax(self, position):
        max = -9999999
        # print(f'Next Position: {position}')
        possible_maxes = []
        for position_direction in self.q_table:
            if position == position_direction[0]:
                possible_maxes.append(str((position_direction, self.q_table[position_direction])))
                if max < self.q_table[(position_direction)]:
                    max = self.q_table[(position_direction)]
                    # print(f'Potential Max: {max} | Action from next position: {position_direction[1]}')
        # print(f'Possible Maxes: {", ".join(possible_maxes)}' )
        # print(f'Actual Max: {max}')
        # print('-----' * 10)
        return max
    
    def get_next_state(self, move):
        if move[1] == "N":
            return move[0] - 10
        if move[1] == "W":
            return move[0] - 1
        if move[1] == "E":
            return move[0] + 1
        if move[1] == "S":
            return move[0] + 10
        
    def get_best_move(self, possible_moves):
        max = -99999
        max_move = None
        for possible_move in possible_moves:
            if self.q_table[possible_move] > max:
                max = self.q_table[possible_move]
                max_move = possible_move
        return max_move
        
    def __get_reward__(self, position):
        if position in self.reward_indices.keys():
            return self.reward_indices[position]
        return 0