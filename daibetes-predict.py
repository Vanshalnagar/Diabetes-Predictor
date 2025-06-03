import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle

# Load dataset
df = pd.read_csv(r'C:/Users/Vanshal/Desktop/Machine_learning/Diabetes-Predictor/diabetes_prediction_dataset.csv')

# One-hot encode categorical variables
df = pd.get_dummies(df)

# Define features and target
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

print("Before SMOTE, training set class distribution:")
print(y_train.value_counts())

# Apply SMOTE to training data only
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE, training set class distribution:")
print(y_train.value_counts())

# Train Random Forest (optional)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions on train and test (optional output)
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest - Training Precision:", precision_score(y_train, y_train_pred_rf))
print("Random Forest - Test Precision:", precision_score(y_test, y_test_pred_rf))

# Train LightGBM
lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(X_train, y_train)

# Default threshold predictions
y_pred = lgb_model.predict(X_test)
print("\nLightGBM Precision:", precision_score(y_test, y_pred))
print("LightGBM Recall:", recall_score(y_test, y_pred))
print("LightGBM F1 Score:", f1_score(y_test, y_pred))

# Adjust prediction threshold to 0.4
y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
y_pred_adjusted = (y_pred_proba >= 0.4).astype(int)

print("\nLightGBM With Adjusted Threshold (0.4):")
print("Precision:", precision_score(y_test, y_pred_adjusted))
print("Recall:", recall_score(y_test, y_pred_adjusted))
print("F1 Score:", f1_score(y_test, y_pred_adjusted))

# Save LightGBM model to disk
with open('diabetes_lgb_model.pkl', 'wb') as f:
    pickle.dump(lgb_model, f)

print("\n✅ Model training complete and saved as diabetes_lgb_model.pkl")
