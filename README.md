# RPS-AI
This project is based on the freeCodeCamp "Rock Paper Scissors" project:
https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/rock-paper-scissors

The game is Rock-Paper-Scissors. Two players select either rock, paper or scissors. Rock beats scissors, scissors beat paper and paper beats rock. If both players select the same, the round is a tie.

The goal of this project is to program an "AI" RPS player that can beat other human or AI players based on a list of their last moves. 

## How to use

https://github.com/OskarHofmann/RPS-AI/blob/19612dff9d7492a2ade5e0b66c1fafd238904a29/RPS.py#L3
defines a "player" that must return "R", "P" or "S". The input is the previous play `prev_play` of the enemy. The parameter `opponent_history` is never used by the game and allows the "player" to keep track of ALL previous plays by the enemy.
<br/><br/>

https://github.com/OskarHofmann/RPS-AI/blob/b71df7f334ae6540c250ad190aa7a380f534cf30/main.py#L6-L9
sets the "player" against 4 hardcoded AIs defined in [RPS_game.py](RPS_game.py). A "player" randomly selecting an output wins ~50 % of the time. To pass the test, "player" should win >= 600 out of 1000 games against all 4 AIs. To test this, run
https://github.com/OskarHofmann/RPS-AI/blob/b71df7f334ae6540c250ad190aa7a380f534cf30/main.py#L20
<br/>

Via
https://github.com/OskarHofmann/RPS-AI/blob/b71df7f334ae6540c250ad190aa7a380f534cf30/main.py#L12
one can play against the "player".



