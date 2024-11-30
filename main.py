import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load Dataset
file_path = "D:\\Arnav\\ML\\Projects\\Disease Detection\\Data_file.xlsx"  # Path to the Excel file
data = pd.read_excel(file_path)

# Data Cleaning
data.dropna(inplace=True)  # Remove rows with missing values
data = data.drop(['date', 'country', 'id', 'occupation'], axis=1)  # Drop unnecessary columns

# Feature Engineering
data['BMI'] = data['weight'] / (data['height'] / 100) ** 2
data = data.drop(['height', 'weight'], axis=1)

# Encode Categorical Variables
data['gender'] = data['gender'].map({'male': 1, 'female': 0})

# Target Variable and Features
X = data.drop('disease', axis=1)
y = data['disease']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Evaluate Model
y_pred = rf_model.predict(X_test)
print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Recall: {recall_score(y_test, y_pred):.2f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")

# Save Scaler and Model
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(rf_model, 'models/final_model.pkl')
print("Model and scaler saved successfully!")

# Function to Get User Input
def get_user_input():
    print("Enter the following details:")
    age = int(input("Age (in days): "))
    active = int(input("Physical Activity (1 for active, 0 for not active): "))
    alco = int(input("Alcohol intake (1 for yes, 0 for no): "))
    ap_hi = int(input("Systolic blood pressure (e.g., 120): "))
    ap_lo = int(input("Diastolic blood pressure (e.g., 80): "))
    cholesterol = int(input("Cholesterol level (1 for normal, 2 for above normal, 3 for well above normal): "))
    gender = int(input("Gender (1 for male, 0 for female): "))
    gluc = int(input("Glucose level (1 for normal, 2 for above normal, 3 for well above normal): "))
    smoke = int(input("Smoking status (1 for smoker, 0 for non-smoker): "))
    height = float(input("Height (in cm, e.g., 175): "))
    weight = float(input("Weight (in kg, e.g., 70): "))
    bmi = weight / (height / 100) ** 2
    return {
        "age": age,
        "active": active,
        "alco": alco,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gender": gender,
        "gluc": gluc,
        "smoke": smoke,
        "BMI": bmi
    }

# Function to Preprocess Input
def preprocess_input(data):
    input_features = [
        data['age'], data['active'], data['alco'], data['ap_hi'], data['ap_lo'],
        data['cholesterol'], data['gender'], data['gluc'], data['smoke'], data['BMI']
    ]
    input_array = np.array(input_features).reshape(1, -1)
    return scaler.transform(input_array)

# Function to Predict Heart Disease
def predict_heart_disease():
    user_data = get_user_input()
    preprocessed_data = preprocess_input(user_data)
    prediction = rf_model.predict(preprocessed_data)[0]
    probability = rf_model.predict_proba(preprocessed_data)[0][1]
    result = "Heart Disease" if prediction == 1 else "No Heart Disease"
    print("\n--- Prediction Result ---")
    print(f"Prediction: {result}")
    print(f"Probability of Heart Disease: {probability:.2f}")

if __name__ == "__main__":
    print("Welcome to the Heart Disease Prediction Program")
    print("1. Train Model")
    print("2. Predict Heart Disease")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        print("\nTraining completed. The model is saved.")
    elif choice == "2":
        predict_heart_disease()
    else:
        print("Invalid choice. Exiting.")
