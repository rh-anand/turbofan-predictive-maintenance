import pandas as pd
import os

# Define base path assuming your script is in 'src' and data is in 'data/raw'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # This gets the project root directory
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
TRAIN_FILE = os.path.join(DATA_DIR, 'train_FD001.txt') # Adjust filename as needed
# Add paths for test file and RUL file if you plan to use them later

index_cols = ['unit number', 'time, in cycles']
setting_cols = [f'operational setting {num}' for num in range(1, 4)]
sensor_cols = [f'sensor measurement {num}' for num in range(1, 22)]
column_names = index_cols + setting_cols + sensor_cols

try:
    df_train = pd.read_csv(TRAIN_FILE, sep = ' ', header = None, names = column_names, index_col = False)
    print(f'Successfully loaded data from {TRAIN_FILE}')
   
   # Drop extra space-generated columns if they exist (often happens with space delimiters)
    df_train.dropna(axis=1, how='all', inplace=True)

    # Initial Inspection (Print to terminal)
    print("\n--- Training Data ---")
    print("First 5 rows:")
    print(df_train.head())
    print("\nShape:")
    print(df_train.shape)
    print("\nInfo:")
    df_train.info()
    print("\nBasic Statistics:")
    print(df_train.describe())
    
except FileNotFoundError:
    print(f"Error: Training file not found at {TRAIN_FILE}")
    print("Please ensure the data is placed correctly in the data/raw directory.")
except Exception as e:
    print(f"An error occurred: {e}")