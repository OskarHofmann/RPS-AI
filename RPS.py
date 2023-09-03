import numpy as np
import itertools

# ideal response based on a predicted move of the oponnent
POSSIBLE_MOVES = ['R', 'P', 'S']
IDEAL_RESPONSE = {'P': 'S', 'R': 'P', 'S': 'R'}

# all possible 3 move sequences 
POSSIBLE_3_MOVES_SEQUENCES = list(itertools.product('RPS', repeat = 3))

#learns from opponent history to predict a move based on the opponents previous 3 moves
def q_learning_3_moves(opponent_history: list[str]) -> str:
    
    # state: oponnent's move at round n, n-1, n-2
    # action: (prediction of) opponent's move at round n+1
    STATES = len(POSSIBLE_3_MOVES_SEQUENCES) # 27
    ACTIONS = 3 # R, P, S

    Q = np.random.rand(STATES, ACTIONS)

    LEARNING_RATE = 0.95
    # GAMMA = 0.96
    HISTORY_STEPS = 12 # how many of the previous moves of the opponent should be used for learning? 0 means all steps

    # this uses only the last HISTORY_STEPS entries of opponent_history
    # still works if HISTORY_STEPS > opponent_history.size or if HISTORY_STEPS = 0 (just uses all entries in both cases)
    considered_opponent_history = opponent_history[-HISTORY_STEPS:]

    pass

#learns from opponent history to predict a move based on the opponents previous move
def q_learning_1_move(opponent_history: list[str]) -> str:

    def calc_reward(prediction: int, truth: int) -> int:
        if prediction == truth:
            return 1
        else:
            return -1
    
    # state: oponnent's move at round n
    # action: (prediction of) opponent's move at round n+1
    STATES = 3 # R, P, S
    ACTIONS = 3 # R, P, S

    Q = np.random.rand(STATES, ACTIONS)

    LEARNING_RATE = 0.95
    # GAMMA = 0.96
    HISTORY_STEPS = 10 # how many of the previous moves of the opponent should be used for learning? 0 means all steps

    # this uses only the last HISTORY_STEPS entries of opponent_history
    # still works if HISTORY_STEPS > opponent_history.size or if HISTORY_STEPS = 0 (just uses all entries in both cases)
    considered_opponent_history = opponent_history[-HISTORY_STEPS:]

    # if the oponent history only contains 1 move, then no real prediction can be made
    if len(considered_opponent_history) < 2:
        return 'R'
    
    # iterate over history of oponnent' moves
    for idx, move in enumerate(considered_opponent_history[:-1]):
        # get index of Q table
        state = POSSIBLE_MOVES.index(move)
        # the correct action is the next move of the opponent that should be predicted
        correct_action = POSSIBLE_MOVES.index(considered_opponent_history[idx+1])
        # for each possible action, update Q-table based on whether the prediction would be right or wrong
        for action in range(ACTIONS):
            reward = calc_reward(action, correct_action)
            Q[state,action] = (1-LEARNING_RATE) * Q[state,action] + LEARNING_RATE * reward

    calculated_response_index = np.argmax(Q[POSSIBLE_MOVES.index(considered_opponent_history[-1]), :])
    calculated_response = POSSIBLE_MOVES[calculated_response_index]
    return IDEAL_RESPONSE[calculated_response]


def player(prev_play, opponent_history=[]):
    if prev_play:
        opponent_history.append(prev_play)

    return q_learning_1_move(opponent_history)
