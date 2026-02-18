import numpy as np
import gymnasium
import random

env = gymnasium.make("FrozenLake-v1")
action_size = env.action_space.n
state_size  = env.observation_space.n

print(action_size)
print(state_size)

qtable = np.zeros((state_size , action_size))
# print(qtable)

total_episodes = 15000    # Total episodes
learning_rates = 0.8      # Learning rate
max_steps      = 99       # Max steps per episode
gamma          = 0.95     # Discounting rate

# Exploration parameters
epsilon        = 1.0      # Exploration rate
max_epsilon    = 1.0      # Exploration probability at statrt
min_epsilon    = 0.01     # Minimum exploration probability
decay_rate     = 0.995    # Exploration decay rate for exploration prob

# 1 List of rewards
rewards = []

# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the invironment
    state = env.reset()
    step  = 0
    done  = False
    total_rewards = 0

    for step in range(max_steps):
        # Choose an action in the current world state (s)
        # first we randomise a number
        exp_exp_tradeoff = random.uniform(0,1)
        
        # Isf this number > greater than epsilon --> exploitation
        if exp_exp_tradeoff > epsilon:
            action = np.ragmax(qtable[state, :])
        
        # Else doing a random choise             --> exploration
        else:
            action = env.action_space.sample()

        # Take the action and observe the outcome state('s) and reward (r)
        observ = env.step(action)
        # break
    # break
    if episode == 10:
        print(f"observation: {observ}")
        
"""        
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a) := Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        # qtable[new_state, :] : all the actions we can take from new state
        temp = reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action]
        qtable[state, action] = qtable[state, action] + learning_rates * temp

        total_rewards += reward

        # Our new state set to state
        state = new_state

        # If done (if we're dead): finish episode
        if done == True:
            break

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * epsilon)
    rewards.append(total_rewards)

print("Score over time: " + str(sum(rewards)/total_episodes))
print(qtable)

# =============================================================================
env.reset()
for episode in range(5):
    state = env.reset()
    step = 0
    done = False
    print("***************************************")
    print("Episode: ", episode)

    for step in range(max_steps):
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state, :])
        new_state, reward, done, info = env.step(action)
        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into a hole)
            env.render()

            # We print the number of Stock it took
            print("Number of steps", step)
            break
        state = new_state

env.close()

"""