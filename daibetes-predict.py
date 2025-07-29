import pandas as pd

# Load the dataset from CSV
df = pd.read_csv(r'C:/Users/Vanshal/Desktop/Machine_learning/Diabetes-Predictor/diabetes_prediction_dataset.csv')

# (Optional) Quick look at the first few rows and columns
print(df.head())
print(df.columns)

# Define feature columns and ensure this order is used for inference
features = ['gender', 'age', 'hypertension', 'heart_disease', 
            'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
target = 'diabetes'

# Keep only these columns
df = df[features + [target]]

# Encode 'gender' as numeric (Male=1, Female=0)
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

# Encode 'smoking_history' as numeric categories
# (convert to lowercase to handle case differences)
df['smoking_history'] = df['smoking_history'].str.lower()
smoking_map = {'never': 0, 'ever': 1, 'former': 2, 
               'not current': 3, 'current': 4, 'no info': 5}
df['smoking_history'] = df['smoking_history'].map(smoking_map)

# Verify encoding (optional)
print(df[['gender', 'smoking_history']].head())

# Drop rows with any missing values (NaN) in these features or the target
df = df.dropna()

from imblearn.over_sampling import SMOTE
from collections import Counter

# Separate features and target
X = df[features]
y = df[target]
print("Before SMOTE:", Counter(y))

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("After SMOTE:", Counter(y_res))

import lightgbm as lgb

# Initialize LightGBM classifier (using default hyperparameters)
model = lgb.LGBMClassifier(random_state=42)

# Train on the resampled dataset
model.fit(X_res, y_res)

# (Optional) Check training accuracy on resampled data
accuracy = model.score(X_res, y_res)
print(f"Training accuracy on SMOTE data: {accuracy:.3f}")


