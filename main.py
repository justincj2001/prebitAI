import pandas as pd
import numpy as np
import pickle

# Load data from CSV
data = pd.read_csv('bitcoin_data.csv')
prices = data['Close'].values

# Define action space
num_actions = 2  # buy (1) or sell (0)

# Initialize Q-table
num_states = len(prices) - 1
Q = np.zeros((num_states, num_actions))

epsilon = 0.1
alpha = 0.1
gamma = 0.9
num_episodes = 10000
start_episode = 0  # Initialize starting episode number

# Load previously saved Q-table and episode number if available
try:
    with open('q_table.pkl', 'rb') as f:
        saved_data = pickle.load(f)
        if len(saved_data) == 2:
            Q, start_episode = saved_data
            print("Loaded Q-table successfully.")
            print("Resuming training from episode", start_episode)
        else:
            raise ValueError("Invalid data format in the file.")
except FileNotFoundError:
    print("No previous Q-table found. Starting from scratch.")

# Define state as the difference between today's and yesterday's closing price
states = np.diff(prices)

def get_reward(action, state):
    if action == 1:  # buy
        reward = state if state > 0 else -state
    elif action == 0:  # sell
        reward = -state if state > 0 else state
    else:
        reward = 0
    
    # Clip reward to ensure it stays within [-10, 10]
    return np.clip(reward, -10, 10)

def calculate_accuracy(Q):
    correct_predictions = 0
    for state, state_values in enumerate(states):
        action = np.argmax(Q[state])
        predicted_value = prices[state] - states[state] if action == 1 else prices[state] + states[state]
        real_value = prices[state+1]
        if (predicted_value > prices[state] and real_value > prices[state]) or (predicted_value < prices[state] and real_value < prices[state]):
            correct_predictions += 1
    return correct_predictions / num_states

for episode in range(start_episode, num_episodes):
    state = 0
    total_reward = 0
    
    while state < num_states - 1:  # Ensure the next state doesn't exceed the maximum index
        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(Q[state])
        
        next_state = state + 1
        reward = get_reward(action, states[state])
        
        Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]))
        
        total_reward += reward
        state = next_state
        
        predicted_value = prices[state] - states[state] if action == 1 else prices[state] + states[state]
        real_value = prices[state+1]
        # print(f"Episode {episode+1}, State: {states[state]}, Predicted Value: {predicted_value}, Real Value: {real_value}, Action: {'Buy' if action == 1 else 'Sell'}, Reward: {reward}, Total Reward: {total_reward}")
    
    accuracy = calculate_accuracy(Q)
    print(f"Episode {episode+1} Accuracy: {accuracy*100:.2f}%")

    # Save Q-table and episode number after each episode
    with open('q_table.pkl', 'wb') as f:
        pickle.dump((Q, episode+1), f)
