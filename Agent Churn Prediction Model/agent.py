import pandas as pd

# Load the data
file_path = 'Updated_Agent_data_for_modeling.csv'  # Update with your file path
data = pd.read_csv(file_path)

# Refined conditions for an ideal churn ratio using AND (intersection)
def calculate_churn_ideal(row):
    churn = (
        (row['Age'] >= 30) & (row['Age'] <= 55) &
        (row['Tenure'] >= 1) & (row['Tenure'] <= 20) &
        (row['Average Training Hours per Year'] >= 2) & (row['Average Training Hours per Year'] <= 25) &
        (row['Average Premium per Policy'] >= 1000) & (row['Average Premium per Policy'] <= 15000) &
        (row['Activity Rate'] >= 5)
    )
    return int(not churn)  # Assuming churn is True if conditions are not met

# Apply the ideal churn calculation to the dataset
data['Churn_Ideal'] = data.apply(calculate_churn_ideal, axis=1)

# Calculate the ideal churn ratio
ideal_churn_ratio = data['Churn_Ideal'].mean()

print("Ideal Churn Ratio:", ideal_churn_ratio)