import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Update the file path to the full path where your CSV file is located
file_path = 'C:/Users/Administrator/Desktop/Internship work/Agent Churn Prediction Model/Agent_data_file.csv'
agent_data = pd.read_csv(file_path)

# Split the data into features and target
X = agent_data.drop(columns=['Agent ID', 'Churn'])
y = agent_data['Churn']

# Identify numerical and categorical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Preprocessing pipeline for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing pipeline for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Initialize models
log_reg = LogisticRegression(random_state=42, max_iter=200)
gb = GradientBoostingClassifier(random_state=42)
svm = SVC(random_state=42)

# Train the models
log_reg.fit(X_train, y_train)
gb.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Evaluate the models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, cm

# Store evaluation results
models = {
    'Logistic Regression': log_reg,
    'Gradient Boosting': gb,
    'SVM': svm
}

results = {}
for model_name, model in models.items():
    results[model_name] = evaluate_model(model, X_test, y_test)

# Print the results
for model_name, result in results.items():
    print(f"{model_name} Results:")
    print(f"Accuracy: {result[0]}")
    print(f"Precision: {result[1]}")
    print(f"Recall: {result[2]}")
    print(f"F1 Score: {result[3]}")
    print(f"Confusion Matrix: \n{result[4]}\n")

# Generate predictions on the test dataset
predictions_test = gb.predict(X_test)

# Save predictions on the test dataset to a CSV file
predictions_test_df = pd.DataFrame({'Agent ID': agent_data.loc[y_test.index, 'Agent ID'], 'Predicted Churn': predictions_test})
test_predictions_path = 'C:/Users/Administrator/Desktop/Internship work/Agent Churn Prediction Model/Agent_Churn_Predictions.csv'
predictions_test_df.to_csv(test_predictions_path, index=False)

# Generate predictions and probabilities on the entire dataset
X_all = preprocessor.transform(X)
predictions_all = gb.predict(X_all)
probabilities_all = gb.predict_proba(X_all)[:, 1]  # Probability of the positive class

# Add the churn label and probability to the original dataframe
agent_data['Churn Label'] = predictions_all
agent_data['Churn Probability'] = probabilities_all

# Save the results to a CSV file
all_predictions_path = 'C:/Users/Administrator/Desktop/Internship work/Agent Churn Prediction Model/Agent_Churn_Predictions_with_Probabilities.csv'
agent_data.to_csv(all_predictions_path, index=False)

# Display the head of the dataframe to confirm
print(agent_data.head())