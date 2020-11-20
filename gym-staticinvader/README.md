# My repo

https://github.com/VivianoRiccardo/OpenAI-custom-static-environment-policy-iteration

# Tutorial for the openai gym environment creation

https://github.com/openai/gym/blob/master/docs/creating-environments.md

# Setup

Run from this readme folder

```
pip install -e gym-staticinvader
```

# Run into gym

```
gym.make('gym_staticinvader:staticinvader-v0')
```

# State

6 X 4 Grid:

4 := enemy invaders (cyka)

3 := our space ship 

1 := laser beam

0 := empty square

7 := wall

In the initial state we have the first 2 rows filled with 2 enemies per row
the column where they stay is randomly chosen. Then we have in the middle row
the 2 random walls and at the center of the last row our spaceship.

# Static environment

Only the actions of the agent can change the environment!

# Action space

1 := left

2 := right

3 := shoot 

# Policy iteration algorithm

The game has been handled as a markov decision process with a transition function
that maps the next state s' with probability P(s' | s,a) = 1 to simplify the computations, so in our case
i decided to have only one possible outcome with probability 1 given a state s and an action a,
however the code has been scripted to handle a distribution probability among different states

# Execution

Leave the policy iteration algorithm run (about 5 minutes to complete) and you will see
the agent playing the game according to the optimal policy he found during the policy iteration algorithm,
When there are no 4 (invaders) the agent achieved its goal!



