import numpy as np
import pandas as pd
import pickle

# Function to load the trained Q-table from file
def load_q_table(file_path):
    try:
        with open(file_path, 'rb') as f:
            Q, _ = pickle.load(f)
        return Q
    except FileNotFoundError:
        print(f"Error: Q-table file '{file_path}' not found.")
        return None
    except Exception as e:
        print("Error loading Q-table:", e)
        return None

# Function to get action using the trained Q-table
def get_action(Q, state):
    try:
        action = np.argmax(Q[state])
        return action
    except IndexError:
        print(f"Error: State {state} is out of range for the Q-table.")
        return None

# Function to calculate state representation for prediction data
def calculate_state(prices_30_days, Q_shape):
    # Calculate the sum of the prices for the past 30 days
    state = sum(prices_30_days)
    
    # Normalize the state value to fit within the range of the Q-table indices
    normalized_state = (state / max_state_value) * (Q_shape[0] - 1)
    
    # Ensure the normalized state value is within the range of the Q-table indices
    return int(np.clip(normalized_state, 0, Q_shape[0] - 1))


# Function for predicting tomorrow's price
def predict_tomorrows_price(Q, prices_30_days):
    if Q is None:
        print("Error: Q-table is not loaded.")
        return None

    state = calculate_state(prices_30_days, Q.shape)
    action = get_action(Q, state)

    if action is not None:
        today_price = prices_30_days[3]  # Today's price is the last price in the list
        print(prices_30_days[3])
        predicted_tomorrow_price = today_price + (1 if action == 1 else -1)
        return predicted_tomorrow_price
    else:
        return None

if __name__ == "__main__":
    # Define the file paths
    q_table_file_path = 'q_table.pkl'
    csv_file_path = 'bitcoin_data.csv'

    # Load the trained Q-table
    Q = load_q_table(q_table_file_path)

    if Q is not None:
        # Load data from CSV file
        try:
            data = pd.read_csv(csv_file_path)
            close_prices_30_days = data['Close'].iloc[:30].tolist()
        except FileNotFoundError:
            print(f"Error: CSV file '{csv_file_path}' not found.")
            close_prices_30_days = []
        except Exception as e:
            print("Error loading data from CSV:", e)
            close_prices_30_days = []

        # Calculate max_state_value
        max_state_value = 1000000  # Placeholder value, replace it with the maximum observed state value

        # Predict tomorrow's price
        predicted_tomorrow_price = predict_tomorrows_price(Q, close_prices_30_days)

        if predicted_tomorrow_price is not None:
            print("Predicted Tomorrow's Price:", predicted_tomorrow_price)
        else:
            print("Failed to predict tomorrow's price. Check your data.")
    else:
        print("Failed to load Q-table. Please check the file path and ensure the Q-table is valid.")
