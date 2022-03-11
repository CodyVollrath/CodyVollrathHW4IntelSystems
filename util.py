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
        #random.seed(0)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.moves_tracker = []
        self.__create_map__()
    
    def __create_map__(self):
        world = [0] * 50
        self.reward_indices = {
            6: -10, 7: -10, 10: 1500, 13: 25, 14: -10, 
            24: -100, 25: -10, 26: -10, 28: -10, 34: -100, 
            38: -10, 42: -10, 44: -100, 48: -10}
        
        mappings = {}
        for index in range(len(world)):
            position = index + 1
            if (position) in self.reward_indices.keys():
                world[index] = self.reward_indices[position]
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
        #cache the actual epsilon value
        store_epsilon = self.epsilon
        #set epsilon to zero for full random on first epoch
        self.epsilon = 0
        for epoch in range(epochs):
            self.transition_function(31)
            #after first set of transitions, set epsilon back to specfied params
            self.epsilon = store_epsilon
            #check if epoch is not the last epoch and reset moves tracker
            if epoch < epochs - 1:
                self.moves_tracker = []
        
        #display moves taken by the agent
        print(self.moves_tracker)
        
    def transition_function(self, position = 31):
        #add first position to moves tracker
        self.moves_tracker.append(position)

        #if the current reward of the position is 1500 (goal) or -100 (death) then end the learning
        while self.__get_reward__(position) != 1500 and self.__get_reward__(position) != -100:
            #get the next move based on the greedy policy
            move = self.greedy_policy(position)
            #get next state from move
            next_state = self.get_next_state(move)
            #update the q table based on Belmont's formula
            self.q_table[move] = (1 - self.alpha) * self.q_table[move] + self.alpha * (self.__get_reward__(position) + self.gamma * self.argmax(next_state)) #get the max value of the direction
            #set the position to the next move
            position = next_state
            #add the new position to the moves tracker
            self.moves_tracker.append(position)
    
    def greedy_policy(self, position):

        #get the possible moves from the position
        possible_positions = self.__find_cardinal_directions__(position)

        #generate a random number to compare to epsilon
        random_number = random.random()
        
        if random_number > self.epsilon:
            best_move = self.get_best_move(possible_positions)
            return best_move
        
        random_move = random.choice(possible_positions)
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
        max = -9999
        for position_direction in self.q_table:
            if position == position_direction[0]:
                if max < self.q_table[(position_direction)]:
                    max = self.q_table[(position_direction)]
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