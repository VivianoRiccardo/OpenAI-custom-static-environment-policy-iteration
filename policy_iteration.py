import gym
import numpy as np
import time
import pickle

def save_d(filename,d):
    with open(filename, 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_d(filename,d):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)
    
def from_nparray_to_str(array):
    a = array.flatten()
    a_str = ','.join(str(x) for x in a)
    return a_str

def from_string_to_nparray(string):
    a = np.array([float(x) for x in string.split(',')])
    return np.reshape(a,(-1,4))
        

# returns a dictionary with keys: all possible states, value: a list with initialized utility function per state (0), initialized policy per state (a random action)
# not recursive because recursion exceeds
def get_states_recursive(d, env, state):
    if env.is_terminal_state(state):
        return d
    for i in env.action_space.keys():
        new_states = env.get_all_possible_new_states(state, env.action_space[i])
        for j in new_states:
            string = from_nparray_to_str(j)
            if string not in d:
                d[string] = [0,env.sample_from_action_space()]
                d = get_states_recursive(d,env,j)
                
    return d      
      
# returns a dictionary with keys: all possible states, value: a list with initialized utility function per state (0), initialized policy per state (a random action)
# not recursive because recursion exceeds
def get_states_iterative(d, env, state):
    if env.is_terminal_state(state):
        return d
    l = [state]
    num = 0
    while(len(l) > 0):
        num+=len(l)#print this number to debug
        l2 = []
        for s in l:
            if env.is_terminal_state(s):
                continue
            
            for i in env.action_space.keys():
                new_states = env.get_all_possible_new_states(s, env.action_space[i])
                

                for j in new_states:
                    string = from_nparray_to_str(j)
                    if string not in d:
                        l2.append(j)
                        d[string] = [0,env.sample_from_action_space()]
        l = l2
                
    return d        
        
def policy_evaluation(env, d, discount_factor):
    flag = True
    while(flag):
        flag = False
        for i in d.keys():
            state = from_string_to_nparray(i)
            if env.is_terminal_state(state):
                continue
            new_states = env.get_all_possible_new_states(state, env.action_space[d[i][1]])
            sum_over_probability = 0.
            for j in new_states:
                sum_over_probability += env.get_probability_of_new_state(j,state,env.action_space[d[i][1]])*d[from_nparray_to_str(j)][0]
            value = env.get_reward_s(state) + sum_over_probability*discount_factor
            if value != d[i][0]:
                flag = True
            d[i][0] = value

# returns a dictionary with keys: all possible states, value: a list with utility function per state , policy per state
def policy_iteration(env):
    d = {}
    
    init_state_str = from_nparray_to_str(env.init_state)
    d[init_state_str] = [0,env.sample_from_action_space()]
    d = get_states_iterative(d,env,env.init_state)
    discount_factor = 0.5
    
    for _ in range(4):
        print("policy evaluation")
        policy_evaluation(env,d,discount_factor)
        print("policy improvement")
        for i in d:
            
            state = from_string_to_nparray(i)
            if env.is_terminal_state(state):
                continue
            
            maximum = -1
            optimal_action = "None"
            
            for j in env.action_space.keys():
                new_states = env.get_all_possible_new_states(state, env.action_space[j])
                sum_over_probability = 0.
                for z in new_states:
                    sum_over_probability += env.get_probability_of_new_state(z,state,env.action_space[j])*d[from_nparray_to_str(z)][0]
                if sum_over_probability > maximum:
                    maximum = sum_over_probability
                    optimal_action = j
                    
            new_states = env.get_all_possible_new_states(state, env.action_space[d[i][1]])
            sum_over_probability = 0.
            for j in new_states:
                sum_over_probability += env.get_probability_of_new_state(j,state,env.action_space[d[i][1]])*d[from_nparray_to_str(j)][0]
            
            if maximum > sum_over_probability:
                d[i][1] = optimal_action
                
    return d

def first_quest():
    env = gym.make('gym_staticinvader:staticinvader-v0')
    np.random.seed()
    for _ in range(5):
        print("old state")
        env.render()#old_state
        action = env.sample_from_action_space()
        new_state, reward, terminal, info = env.step(action)
        print("action: "+action)#action
        print("new state")
        print(new_state)#new state
        print("reward: "+str(reward))#reward
        print("is terminal: "+str(terminal))#is terminal
        print('Probability of being in this new state given that action: '+str(info['probability of being in this state given the old state and the action taken']))#probability distribution
             
def second_quest():
    env = gym.make('gym_staticinvader:staticinvader-v0')
    np.random.seed()
    d = policy_iteration(env)
    env.reset()
    while(not env.is_terminal_state(env.current_state)):
        env.render()
        env.step(d[from_nparray_to_str(env.current_state)][1])
        time.sleep(0.3)
    env.render()    
    print("The game ended")
            
if __name__ == "__main__":
    print("First quest:")
    first_quest()
    print("Second quest:")
    second_quest()
