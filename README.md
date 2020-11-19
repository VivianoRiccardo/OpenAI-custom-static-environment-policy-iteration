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

7 X 5 Grid:

4 := enemy invaders (cyka)

3 := our space ship 

1 := laser beam

0 := empty square

7 := wall

In the initial state we have the first 3 rows filled with 3 enemies per row
the column where they stay is randomly chosen. Then we have in the middle row
the 3 random walls and at the center of the last row our spaceship.

# Static environment

Only the actions of the agent can change the environment!

# Action space

1 := left

2 := right

3 := shoot 

# Policy iteration algorithm

The game has been handled as a markov decision process with a transition function
that maps the next state s' with probability P(s' | s,a) = 1, so in our case
i decided to have only 1 possible outcome given a state s and an action a



