# Heart Disease Detection

This project involves the development of a machine learning model to predict the likelihood of heart disease using patient health data.  
The model uses features such as **age, blood pressure, cholesterol levels, glucose levels, BMI, lifestyle habits (smoking, alcohol, physical activity), and gender** to assess the risk.

## Key Features
- **Data Preprocessing**: 
  - Handling missing values
  - Categorical encoding (gender)
  - Feature engineering (BMI calculation from height & weight)
  - Standardization of features
- **Model Training**:
  - Trained using **Random Forest Classifier**
  - Achieved an accuracy of ~**85%** on the test dataset
- **CLI Tool**:
  - Allows users to input health parameters directly
  - Provides **probability-based prediction** (`Heart Disease` or `No Heart Disease`)
- **Model Saving**:
  - Trained model and scaler saved using **Joblib** for future use and deployment

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Joblib
