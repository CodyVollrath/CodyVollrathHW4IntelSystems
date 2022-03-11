'''
This script runs the primary functions of the app
'''

__author__ = 'Cody Vollrath'
__version__ = 'Spring 2022'
__pylint__ = 'v2.12.2'
from util import q_learner

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.2

def main():
    '''
    The entry point of the application
    '''
    test = q_learner(ALPHA, GAMMA, EPSILON)
    test.train(1)

if __name__ == '__main__':
    main()