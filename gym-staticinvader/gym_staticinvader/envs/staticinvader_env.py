import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

# 6X4 grid
# We consider a transition function
# 4 = invaders, 1 = shot, 3 = shuttle, 7 = walls (identifiers)
# 1 left, 2 right, 3 shot (discrete actions)

class StaticinvaderEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    init_state = None
    current_state = None
    action_space = {}
    
    def __init__(self):
        self.set_init_space()
        self.current_state = np.copy(self.init_state)
        self.action_space['left'] = 1
        self.action_space['right'] = 2
        self.action_space['shoot'] = 3
        
    
    # set self._init_state:
    # There can be different initial states for each static invaders env, just to test
    # the policy iteration algorithm with different static enviroment settings. What doesn't change among intial states:
    # the first 3 rows are always with 3 invaders per row displayed on random columns
    # in the middle row there are 3 walls displayed randomly
    # in the last row central column there is our space ship
    def set_init_space(self):
        self.init_state = np.zeros((6,4))
        idx1 = np.random.rand(2,4).argsort(1)[:,:2]
        self.init_state[np.arange(0,2)[:,None],idx1] = 4
        idx2 = np.random.rand(1,4).argsort(1)[:,:2]
        self.init_state[np.arange(3,4)[:,None],idx2] = 7
        self.init_state[5][2] = 3
    
    def get_actions(self):
        return action_space
  # give me a state and an action i will return a new state
    def transition(self,current_state, action):
        new_state = np.copy(current_state)
      
        #update according to the shots
        for i in range(5):
            for j in range(4):
                if i == 0 and new_state[i][j] == 1:
                    new_state[i][j] = 0
                elif new_state[i][j] == 4 or new_state[i][j] == 7:
                    if new_state[i+1][j] == 1:
                        new_state[i][j] = 0
                        new_state[i+1][j] = 0
                elif new_state[i][j] == 0:
                    if new_state[i+1][j] == 1:
                      new_state[i+1][j] = 0
                      new_state[i][j] = 1
        idx = -1
        for i in range(4):
            if new_state[5][i] == 3:
                idx = i
                break
                
        if action == 1:#go to left
            new_state[5][idx] = 0
            new_state[5][max(0,idx-1)] = 3
      
        elif action == 2:#go to right
            new_state[5][idx] = 0
            new_state[5][min(3,idx+1)] = 3
      
        elif action == 3:#shot
            new_state[4][idx] = 1
        
        else:
            return current_state

        return new_state
    
    #uniform sampling actions
    def sample_from_action_space(self):
        r = np.random.rand()
        if r < 0.33:
            return 'left'
        elif r < 0.67:
            return 'right'
        return 'shoot'
       
    #returns 1 if we are in terminal state, 0 otherwise
    def is_terminal_state(self,current_state):
        if 4 in current_state:
            return False
        return True
       
   # returns a list of all possible states given the old state and an action
    def get_all_possible_new_states(self,old_state, action):
        return [self.transition(old_state,action)]
   
   # get the probability to get the "new state" from "old state and action" (transition function) 
    def get_probability_of_new_state(self,new_state,old_state,action):
        if(new_state == self.transition(old_state,action)).all():
            return 1
        return 0
       
  # compare old state and new state and i will give you the reward (difference of invaders) in this sense is defined as R(s,s')
    def get_reward(self,old_state,new_state):
        unique, counts = np.unique(old_state, return_counts=True)
        d1 = dict(zip(unique, counts))
      
        unique, counts = np.unique(new_state, return_counts=True)
        d2 = dict(zip(unique, counts))
        
        if 4 in d1:
            if 4 in d2:
                return d1[4]-d2[4]
            else:
                return d1[4]
        return 0
    
    
    def get_reward_s(self,state):
        reward = 0
        for i in range(5):
            for j in range(4):
                if state[i][j] == 4:
                    if state[i+1][j] == 1:
                        reward+=1
        return reward
        
        
    def step(self, string_action):
        if string_action not in self.action_space:
            return np.copy(self.current_state), 0, self.is_terminal_state(self.current_state), {}
        action = self.action_space[string_action]
        old_state = self.current_state
        new_states = self.get_all_possible_new_states(old_state,action)
        l1 = []
        l2 = []
        for i in range(len(new_states)):
            l1.append(i)
            l2.append(self.get_probability_of_new_state(new_states[i],old_state,action))
        new_state = new_states[np.random.choice(l1,p=l2)]
        self.current_state = new_state
        reward = self.get_reward(old_state,self.current_state)
        done = self.is_terminal_state(self.current_state)
        d = {}
        d['probability of being in this state given the old state and the action taken'] =  self.get_probability_of_new_state(self.current_state,old_state,action)
        return np.copy(self.current_state), reward, done, d
        
    def reset(self):
        self.current_state = np.copy(self.init_state)
    def render(self, mode='human'):
        print(self.current_state)
        print("\n")
     
    def close(self):
        pass
