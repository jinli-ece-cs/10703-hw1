#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input

import gym
import deeprl_hw1.lake_envs as lake_env
import deeprl_hw1.rl as rl
import time
import matplotlib.pylab as plt
import numpy as np
def run_random_policy(env):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    initial_state = env.reset()
    env.render()
    time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    while True:
        nextstate, reward, is_terminal, debug_info = env.step(
            env.action_space.sample())
        env.render()

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

        time.sleep(1)

    return total_reward, num_steps
def run_given_policy_get_reward(env,policy,gamma):
    initial_state = env.reset()
    #env.render()
    total_reward = 0
    is_terminal = False
    nextstate = initial_state
    step = 0
    while not is_terminal:
       
        nextstate, reward, is_terminal, debug_info = env.step(policy[nextstate].astype(int))
        total_reward += total_reward + reward*(gamma**step)
        #env.render()
        step += 1
    return total_reward

def print_env_info(env):
    print('Environment has %d states and %d actions.' % (env.nS, env.nA))


def print_model_info(env, state, action):
    transition_table_row = env.P[state][action]
    print(
        ('According to transition function, '
         'taking action %s(%d) in state %d leads to'
         ' %d possible outcomes') % (lake_env.action_names[action],
                                     action, state, len(transition_table_row)))
    for prob, nextstate, reward, is_terminal in transition_table_row:
        state_type = 'terminal' if is_terminal else 'non-terminal'
        print(
            '\tTransitioning to %s state %d with probability %f and reward %f'
            % (state_type, nextstate, prob, reward))


def main():
    
    # create the environment
    env = gym.make('Deterministic-8x8-FrozenLake-v0')
    n = np.sqrt(env.nS).astype(int)
    print_env_info(env)
    #print_model_info(env, 0, lake_env.DOWN)
    #print_model_info(env, 1, lake_env.DOWN)
    #print_model_info(env, 14, lake_env.RIGHT)
    

    
    #total_reward, num_steps = run_random_policy(env)
    gamma = 0.9
    final_reward = 0
    #for i in range (100):
    start = time.time()
    #policy, value_func, num_policy, num_eval = rl.policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3), tol=1e-3)
    value_func, num_value = rl.value_iteration_async_custom(env, gamma, max_iterations=int(1e3),tol=1e-3)
    duration = time.time()-start
    #print('Agent received total reward of: %f' % total_reward)
    #print('Agent took %d number of policy improvement iterations' % num_policy)
    print('Agent took %d number of value improvement iterations' % num_value)
    #print('Agent took %d number of eval iterations' % num_eval)
    print('Run time took %f milli seconds' % (duration*1000))
    
    
    #policy = rl.value_function_to_policy(env, gamma, value_func)
    #reward = run_given_policy_get_reward(env,policy,gamma)
    #final_reward += reward
    #print('iteration: %d' % i)
    #print('computed value for starting state: %f'% value_func[3])
    #print('simulated value for starting state: %f'% reward)
    #print('average simulated value for starting state: %f'% (final_reward/100.0))
    #rl.print_policy(policy, lake_env.action_names) 
'''
    value_print = value_func.reshape([n,n])
    value_print = np.matrix(value_print)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(value_print, interpolation='none', cmap=plt.cm.ocean)
    plt.colorbar()
    plt.show()
'''
if __name__ == '__main__':
    main()
