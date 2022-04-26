import joblib
import pandas as pd
import time
import argparse

# Loading model
model = joblib.load('model.joblib')

# Setting up the CLI
parser = argparse.ArgumentParser(description='Predict heart attack')

parser.add_argument('file_path', type=str,
                    help='Path to file where user records are stored')

args = parser.parse_args()

path_to_records = args.file_path

# Loading records
user_data = pd.read_csv(path_to_records)
column_names = user_data.columns

# Predictions
for _, row in user_data.iterrows():
    x = pd.DataFrame([row])

    print(x.to_string(index=False), '\n')

    pred = model.predict(x)[0]
    if pred == 1:
        print("You are prone to heart attack...Please take your drugs!!!")
    else:
        print("No risk of heart attack")
    print('-'*100, '\n\n')

    # Pause (to simulate readings every 3 secondes)
    time.sleep(3)
