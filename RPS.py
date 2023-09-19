import numpy as np
import itertools

# ideal response based on a predicted move of the oponnent
POSSIBLE_MOVES = ['R', 'P', 'S']
IDEAL_RESPONSE = {'P': 'S', 'R': 'P', 'S': 'R'}



#learns from opponent history to predict a move based on the opponents previous n moves
def q_learning_enemy_moves(opponent_history: list[str], n_moves: int = 3) -> str:

    def calc_reward(prediction: int, truth: int) -> int:
        if prediction == truth:
            return 1
        else:
            return -1
        
    # all possible n move sequences 
    POSSIBLE_N_MOVES_SEQUENCES = list(itertools.product('RPS', repeat = n_moves))
    
    # state: oponnent's move at round n, n-1, n-2 ... n-(steps-1)
    # action: (prediction of) opponent's move at round n+1
    STATES = len(POSSIBLE_N_MOVES_SEQUENCES)
    ACTIONS = 3 # R, P, S

    # Q = np.zeros((STATES, ACTIONS))
    Q = np.random.rand(STATES, ACTIONS)

    LEARNING_RATE = 0.9
    # GAMMA = 0.96
    HISTORY_STEPS = 0 # how many of the previous moves of the opponent should be used for learning? 0 means all steps

    # this uses only the last HISTORY_STEPS entries of opponent_history
    # still works if HISTORY_STEPS > opponent_history.size or if HISTORY_STEPS = 0 (just uses all entries in both cases)
    considered_opponent_history = opponent_history[-HISTORY_STEPS:]

    # if the oponent history contains less than 4 moves, then no real prediction can be made
    if len(considered_opponent_history) < (n_moves + 1):
        return 'P'

    # iterate over history of oponnent' moves
    for idx, move in enumerate(considered_opponent_history[:-n_moves]):
        # build N move sequence
        move_sequence = tuple([considered_opponent_history[i] for i in range(idx, idx+n_moves)])    
        # get index of Q table
        state = POSSIBLE_N_MOVES_SEQUENCES.index(move_sequence)
        # the correct action is the next move of the opponent that should be predicted
        correct_action = POSSIBLE_MOVES.index(considered_opponent_history[idx+n_moves])
        # for each possible action, update Q-table based on whether the prediction would be right or wrong
        for action in range(ACTIONS):
            reward = calc_reward(action, correct_action)
            Q[state,action] = (1-LEARNING_RATE) * Q[state,action] + LEARNING_RATE * reward

    last_n_moves = tuple(considered_opponent_history[-n_moves:])
    calculated_move_index = np.argmax(Q[POSSIBLE_N_MOVES_SEQUENCES.index(last_n_moves), :])
    calculated_move = POSSIBLE_MOVES[calculated_move_index]

    return IDEAL_RESPONSE[calculated_move]


def player(prev_play, opponent_history=[]):
    if prev_play:
        opponent_history.append(prev_play)
    else:
        # reset the opponent history after an empty prev_play as it indicates playing against a new enemy
        opponent_history = []

    # return q_learning_1_move(opponent_history)
    return q_learning_enemy_moves(opponent_history, 3)

#TODO:
# 1) player uses Q_table = [Q] and only calculates based on latest move
# 2) player calculates next move based on history of own moves, use the one with highest confidence
# 3) add calculation of enemy move based on history of 2 player moves and use best of all 3

