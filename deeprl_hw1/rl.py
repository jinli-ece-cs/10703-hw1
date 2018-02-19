# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import deeprl_hw1.lake_envs as lake_env
import random
''' env parameters
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    (*) dictionary dict of dicts of lists, where
    P[s][a] == [(probability, nextstate, reward, done), ...]
'''
def print_policy(policy, action_names):
    """Print the policy in human-readable format.

    Parameters
    ----------
    policy: np.ndarray
      Array of state to action number mappings
    action_names: dict
      Mapping of action numbers to characters representing the action.
    """
    str_policy = policy.astype('str')
    for action_num, action_name in action_names.items():
        np.place(str_policy, policy == action_num, action_name)
	#print(str_policy.size)
    n= np.sqrt(str_policy.size).astype(int)
    print(str_policy.reshape([n,n]))


def value_function_to_policy(env, gamma, value_function):
    """Output action numbers for each state in value_function.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to compute policy for. Must have nS, nA, and P as
      attributes.
    gamma: float
      Discount factor. Number in range [0, 1)
    value_function: np.ndarray
      Value of each state.

    Returns
    -------
    np.ndarray
      An array of integers. Each integer is the optimal action to take
      in that state according to the environment dynamics and the
      given value function.
    """    
    policy = np.zeros(env.nS)
    
    for s in range(env.nS):
        values = np.zeros(env.nA)
        for a in range(env.nA):
            for (probability, nextstate, reward, done) in env.P[s][a]:
                values[a] += probability*(reward+gamma*value_function[nextstate])
        policy[s]=np.argmax(values).astype(int)
    return policy


def evaluate_policy_sync(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a given policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value = np.zeros(env.nS)
    
    for it in range (max_iterations):
      delta = 0
      old_value = np.copy(value)
      value = np.zeros(env.nS)
      for s in range(env.nS):
        
        a = policy[s]
        for (probability, nextstate, reward, done) in env.P[s][a]:
          value[s] += probability*(reward+gamma*old_value[nextstate])
        delta = max(delta, abs(value[s]-old_value[s]))
	
      if delta<tol:
        break
    #print('it is %d' % it)
    return value, (it+1)


def evaluate_policy_async_ordered(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a given policy by asynchronous DP.  Updates states in
    their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value = np.zeros(env.nS)
    
    for it in range (max_iterations):
     
      old_value = np.copy(value)
      for s in range(env.nS):   
        value[s]=0     
        a = policy[s]
        for (probability, nextstate, reward, done) in env.P[s][a]:
          value[s] += probability*(reward+gamma*value[nextstate])
      
      if max(abs(value-old_value))<tol:
        break
    return value, (it+1)


def evaluate_policy_async_randperm(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluates the value of a policy.  Updates states by randomly sampling index
    order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    value = np.zeros(env.nS)
    
    for it in range (max_iterations):
      old_value = np.copy(value)
      rand_idx = np.arange(env.nS)
      np.random.shuffle(rand_idx)
      for s in rand_idx:           
        value[s]=0     
        a = policy[s]
        for (probability, nextstate, reward, done) in env.P[s][a]:
          value[s] += probability*(reward+gamma*value[nextstate])
        
	
      if max(abs(value-old_value))<tol:
        break
    return value, (it+1)
    


def evaluate_policy_async_custom(env, gamma, policy, max_iterations=int(1e3), tol=1e-3):
    """Performs policy evaluation.
    
    Evaluate the value of a policy. Updates states by a student-defined
    heuristic. 

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    policy: np.array
      The policy to evaluate. Maps states to actions.
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, int
      The value for the given policy and the number of iterations till
      the value function converged.
    """
    return np.zeros(env.nS), 0


def improve_policy(env, gamma, value_func, policy):
    """Performs policy improvement.
    
    Given a policy and value function, improves the policy.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    value_func: np.ndarray
      Value function for the given policy.
    policy: dict or np.array
      The policy to improve. Maps states to actions.

    Returns
    -------
    bool, np.ndarray
      Returns true if policy changed. Also returns the new policy.
    """
    changed = False
    for s in range(env.nS):
      old_action = policy[s]
      values = np.zeros(env.nA)
      for a in range(env.nA):
        for (probability, nextstate, reward, done) in env.P[s][a]:
          values[a] += probability*(reward+gamma*value_func[nextstate])
      policy[s]=np.argmax(values)
      if policy[s] != old_action:
        changed = True
    return changed, policy


def policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs policy iteration.

    See page 85 of the Sutton & Barto Second Edition book.

    You should use the improve_policy() and evaluate_policy_sync() methods to
    implement this method.
    
    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)
    num_iter = 0
    for s in range(max_iterations):
      value_func, num_iter_tmp = evaluate_policy_sync(env, gamma, policy, max_iterations, tol)
      num_iter += num_iter_tmp
      changed, policy=improve_policy(env, gamma, value_func, policy)
      if not changed:
        break
      
      
    return policy, value_func, s+1, num_iter


def policy_iteration_async_ordered(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_ordered methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)

    num_iter = 0
    for s in range(max_iterations):
      value_func, num_iter_tmp = evaluate_policy_async_ordered(env, gamma, policy, max_iterations, tol)
      num_iter += num_iter_tmp
      changed, policy=improve_policy(env, gamma, value_func, policy)
      if not changed:
        break
      
      
    return policy, value_func, s+1, num_iter
   


def policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                    tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_randperm methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)

    num_iter = 0
    for s in range(max_iterations):
      value_func, num_iter_tmp = evaluate_policy_async_randperm(env, gamma, policy, max_iterations, tol)
      num_iter += num_iter_tmp
      changed, policy=improve_policy(env, gamma, value_func, policy)
      if not changed:
        break
      
      
    return policy, value_func, s+1, num_iter
    


def policy_iteration_async_custom(env, gamma, max_iterations=int(1e3),
                                  tol=1e-3):
    """Runs policy iteration.

    You should use the improve_policy and evaluate_policy_async_custom methods
    to implement this method.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    (np.ndarray, np.ndarray, int, int)
       Returns optimal policy, value function, number of policy
       improvement iterations, and number of value iterations.
    """
    policy = np.zeros(env.nS, dtype='int')
    value_func = np.zeros(env.nS)

    return policy, value_func, 0, 0


def value_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)
    n = np.sqrt(env.nS).astype(int)
    #print(n)
    if n == 4:
        for i in range (n):
            for j in range(n):
                if lake_env.MAPS["4x4"][i][j] == 'S' or lake_env.MAPS["4x4"][i][j] == 'F':
                    value_func[i*n+j] = random.random()
    if n == 8:
        for i in range (n):
            for j in range(n):
                if lake_env.MAPS["8x8"][i][j] == 'S' or lake_env.MAPS["8x8"][i][j] == 'F':
                    value_func[i*n+j] = random.random()
    #print(value_func )
	
	
    for it in range(max_iterations):
		delta = 0
		old_value = value_func.copy()
		
		for s in range(env.nS):
			values = np.zeros(env.nA)
			for a in range(env.nA):
				for (probability, nextstate, reward, done) in env.P[s][a]:
					values[a] += probability*(reward+gamma*old_value[nextstate])		
			value_func[s] = max(values)
			delta = max(delta, abs(value_func[s]-old_value[s]))
			#print(delta)
		if (delta < tol):
			#print(delta)
			break
    return value_func, (it+1)


def value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states in their 1-N order.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)
    n = np.sqrt(env.nS).astype(int)
    if n == 4:
        for i in range (n):
            for j in range(n):
                if lake_env.MAPS["4x4"][i][j] == 'S' or lake_env.MAPS["4x4"][i][j] == 'F':
                    value_func[i*n+j] = random.random()
    if n == 8:
        for i in range (n):
            for j in range(n):
                if lake_env.MAPS["8x8"][i][j] == 'S' or lake_env.MAPS["8x8"][i][j] == 'F':
                    value_func[i*n+j] = random.random()
    
    for it in range(max_iterations):
		
		old_value = value_func.copy()
		
		for s in range(env.nS):
			values = np.zeros(env.nA)
			for a in range(env.nA):
				for (probability, nextstate, reward, done) in env.P[s][a]:
					values[a] += probability*(reward+gamma*value_func[nextstate])		
			value_func[s] = max(values)
			
			
		if ( max(abs(value_func-old_value))< tol):
			break
    return value_func, (it+1)


def value_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
                                   tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by randomly sampling index order permutations.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)
    n = np.sqrt(env.nS).astype(int)
    if n == 4:
        for i in range (n):
            for j in range(n):
                if lake_env.MAPS["4x4"][i][j] == 'S' or lake_env.MAPS["4x4"][i][j] == 'F':
                    value_func[i*n+j] = random.random()
    if n == 8:
        for i in range (n):
            for j in range(n):
                if lake_env.MAPS["8x8"][i][j] == 'S' or lake_env.MAPS["8x8"][i][j] == 'F':
                    value_func[i*n+j] = random.random()
    
    for it in range(max_iterations):
        old_value = value_func.copy()
        rand_idx = np.arange(env.nS)
        np.random.shuffle(rand_idx)
        for s in rand_idx:
			values = np.zeros(env.nA)
			for a in range(env.nA):
				for (probability, nextstate, reward, done) in env.P[s][a]:
					values[a] += probability*(reward+gamma*value_func[nextstate])		
			value_func[s] = max(values)
			
			
        if ( max(abs(value_func-old_value))< tol):
            break
    return value_func, (it+1)



def value_iteration_async_custom(env, gamma, max_iterations=int(1e3), tol=1e-3):
    """Runs value iteration for a given gamma and environment.
    Updates states by student-defined heuristic.

    Parameters
    ----------
    env: gym.core.Environment
      The environment to compute value iteration for. Must have nS,
      nA, and P as attributes.
    gamma: float
      Discount factor, must be in range [0, 1)
    max_iterations: int
      The maximum number of iterations to run before stopping.
    tol: float
      Determines when value function has converged.

    Returns
    -------
    np.ndarray, iteration
      The value function and the number of iterations it took to converge.
    """
    value_func = np.zeros(env.nS)
    n = np.sqrt(env.nS).astype(int)
    if n == 4:
        for i in range (n):
            for j in range(n):
                if lake_env.MAPS["4x4"][i][j] == 'S' or lake_env.MAPS["4x4"][i][j] == 'F':
                    value_func[i*n+j] = random.random()
    if n == 8:
        for i in range (n):
            for j in range(n):
                if lake_env.MAPS["8x8"][i][j] == 'S' or lake_env.MAPS["8x8"][i][j] == 'F':
                    value_func[i*n+j] = random.random()
    
    
    dist = np.zeros(env.nS)
    goal_state = 3
    for i in range(n):
        for j in range(n):
            dist[i*n+j] = np.sqrt((i-0)**2 + (j-3)**2)
    sorted_states = np.append([np.arange(env.nS)], [dist],axis=0)
    #print(sorted_states[1,:].argsort())
    sorted_states = sorted_states[:,sorted_states[1,:].argsort()]
    print(sorted_states)
    for it in range(max_iterations):
        old_value = value_func.copy()
        
        for s in sorted_states[0,:].astype(int):
            print(s)
            values = np.zeros(env.nA)
            for a in range(env.nA):
                for (probability, nextstate, reward, done) in env.P[s][a]:
                    values[a] += probability*(reward+gamma*value_func[nextstate])		
            value_func[s] = max(values)			
        if ( max(abs(value_func-old_value))< tol):
            break
    return value_func, (it+1)


