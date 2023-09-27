import numpy as np
import itertools

# ideal response based on a predicted move of the oponnent
POSSIBLE_MOVES = ['R', 'P', 'S']
IDEAL_RESPONSE = {'P': 'S', 'R': 'P', 'S': 'R'}
ACTIONS = 3 # R, P, S


#learns from opponent history to predict a move based on the opponents previous n moves
def q_learning_enemy_moves(opponent_history: list[str], n_moves: int = 3, q_table = list[np.ndarray]) -> str:

    # all possible n move sequences 
    possible_n_moves_sequences = list(itertools.product('RPS', repeat = n_moves))
    # state: move at round n, n-1, n-2 ... n-(steps-1)
    # action: (prediction of) move at round n+1
    states = len(possible_n_moves_sequences)
 

    def calc_reward(prediction: int, truth: int) -> int:
        if prediction == truth:
            return 1
        else:
            return -1


    def initialize_q_table() -> np.ndarray:
        # Q = np.zeros((STATES, ACTIONS))
        Q = np.random.rand(states, ACTIONS)
        return Q
    

    # updates q_table (inplace)
    def update_q_table(Q: np.ndarray, previous_steps: list[str], true_outcome: str) -> None:
        learning_rate = 0.9
        # GAMMA = 0.96

        # create tuple as itertools.product returns tuples, i.e. POSSIBLE_N_MOVES_SEQUENCES is a list of tuples
        move_sequence = tuple(previous_steps)
        # get index of Q table
        state = possible_n_moves_sequences.index(move_sequence)
        correct_action = POSSIBLE_MOVES.index(true_outcome)

        # for each possible action, update Q-table based on whether the prediction would be right or wrong
        for action in range(ACTIONS):
            reward = calc_reward(action, correct_action)
            Q[state,action] = (1-learning_rate) * Q[state,action] + learning_rate * reward

    
    def predict_move(Q: np.ndarray, previous_steps: list[str]) -> str:
        last_n_moves = tuple(previous_steps)
        calculated_move_index = np.argmax(Q[possible_n_moves_sequences.index(last_n_moves), :])
        calculated_move = POSSIBLE_MOVES[calculated_move_index]
        return calculated_move

        

    # if the oponent history contains less than n moves, then no real prediction can be made
    if len(opponent_history) < (n_moves + 1):
        return 'P'
    
    if q_table:
        Q = q_table[0]
    else:
        Q = initialize_q_table()
        q_table.append(Q)

    # get the last n+1 moves
    # the first n moves define the state for training, the last move is outcome that should be predicted (action)
    move_sequence_training =[opponent_history[i] for i in range(-n_moves-1 , -1)]
    correct_action = opponent_history[-1]
    update_q_table(Q, move_sequence_training, correct_action)

    last_steps = [opponent_history[i] for i in range(-n_moves , 0)]
    calculated_move =  predict_move(Q, last_steps)

    return IDEAL_RESPONSE[calculated_move]


# Use default empty list values for opponent_history and q_table as this gives static access to one list respectively that can be updated
# even outside the life cycle of the player function.
# This works as default values are initialized once (at first call of the function) and lists are mutable so that when the list is changed
# it is still the same object the initialized variable is referencing.
def player(prev_play, opponent_history = [], player_history = [], q_table = []):
    if prev_play:
        opponent_history.append(prev_play)
    else:
        # reset the opponent history and q_table after an empty prev_play as it indicates playing against a new enemy
        opponent_history = []
        player_history = []
        q_table = []

    calculated_move = q_learning_enemy_moves(opponent_history, n_moves = 3, q_table = q_table)
    player_history.append(calculated_move)
    return calculated_move

#TODO:
# 1) add calculation of enemy move based on history of player moves
# 2) use prediction with highes probabilty (return probability in predict_move function)

