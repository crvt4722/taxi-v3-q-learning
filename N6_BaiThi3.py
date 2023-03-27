# Run in Pycharm
# pip install stable_baselines3
# pip install pygame
# pip install gym

import gym
import numpy
import random

R = 0
G = 1
Y = 2
B = 3

def strategy(taxi_row, taxi_col, passenger_index, destination_index):
    result = []
    env = gym.make("Taxi-v3").env

    # Make a new matrix filled with zeros.
    q_table = numpy.zeros([env.observation_space.n, env.action_space.n])

    training_episodes = 20000  # Amount of times to run environment while training.

    alpha = 0.1  # Learning Rate
    gamma = 0.6  # Discount Rate
    epsilon = 0.1  # Chance of selecting a random action instead of maximising reward.

    # Trainning
    for i in range(training_episodes):
        state = env.reset()  # Reset returns observation state and other info. We only need the state.
        done = False
        penalties, reward, = 0, 0

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Pick a new action for this state.
            else:
                action = numpy.argmax(q_table[state])  # Pick the action which has previously given the highest reward.

            next_state, reward, done, info = env.step(action)

            old_value = q_table[state, action]  # Retrieve old value from the q-table.
            next_max = numpy.max(q_table[next_state])

            # Update q-value for current state.
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:  # Checks if agent attempted to do an illegal action.
                penalties += 1

            state = next_state

        if i % 1000 == 0:  # Output number of completed episodes every 100 episodes.
            print(f"Episode: {i}")

    print("Training finished.\n")


    state = env.encode(taxi_row,taxi_col,passenger_index,destination_index)
    env.s = state

    epochs, penalties, rewards = 0, 0, 0
    done = False

    while not done:
        action = numpy.argmax(q_table[state])
        state, reward, done, info = env.step(action)
        rewards += reward

        if reward == -10:
            penalties += 1

        epochs += 1
        env.render()
        result.append(action)

    print('\n\nDONE!!!')
    print(f"Timestep: {epochs}")
    print(f"Reward: {rewards}")

    return result

# Try this function.
# print(strategy(3,1,R,Y))

'''
DONE!!!
Timestep: 10
Reward: 11
[1, 1, 3, 1, 4, 0, 0, 0, 0, 5]
'''