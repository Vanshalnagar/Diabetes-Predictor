import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Import SMOTE for balancing the training data
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv(r'C:/Users/Vanshal/Desktop/Machine_learning/Diabetes-Predictor/diabetes_prediction_dataset.csv')

# Convert categorical columns to numeric using one-hot encoding
df = pd.get_dummies(df)

# Define features (x) and target (y)
y = df['diabetes']
x = df.drop('diabetes', axis=1)

# Split the data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Check initial class distribution in training set before SMOTE
print("Before SMOTE, training set class distribution:")
print(y_train.value_counts())

# Initialize SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE only on the training data to create synthetic samples of the minority class
x_train, y_train = smote.fit_resample(x_train, y_train)

# Check class distribution after applying SMOTE to ensure balancing
print("\nAfter SMOTE, training set class distribution:")
print(y_train.value_counts())

# Train the Random Forest model on the balanced training data
lr = RandomForestClassifier()
lr.fit(x_train, y_train)

# Make predictions on training and test sets
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

# Output true training labels and predicted labels (optional)
print("\nTrue labels (y_train):")
print(y_train)
print("\nPredicted labels (y_lr_train_pred):")
print(y_lr_train_pred)

# Check balance of target variable in entire dataset (before splitting)
print("\nClass distribution in 'diabetes' (entire dataset):")
print(y.value_counts())

# Show class distribution in percentage (entire dataset)
print("\nClass distribution in % (entire dataset):")
print(y.value_counts(normalize=True) * 100)



import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize LightGBM classifier
lgb_model = lgb.LGBMClassifier(random_state=42)

# Train the model
lgb_model.fit(x_train, y_train)

# Predict with default threshold
y_pred_lgb = lgb_model.predict(x_test)

# Evaluate with default threshold
print("LightGBM Precision:", precision_score(y_test, y_pred_lgb))
print("LightGBM Recall:", recall_score(y_test, y_pred_lgb))
print("LightGBM F1 Score:", f1_score(y_test, y_pred_lgb))

# Predict probabilities for threshold adjustment
y_pred_proba = lgb_model.predict_proba(x_test)[:, 1]
y_pred_adjusted = (y_pred_proba >= 0.4).astype(int)  # Adjusting threshold from 0.5 to 0.4

# Evaluate with adjusted threshold
print("\nWith Adjusted Threshold (0.4):")
print("Adjusted Precision:", precision_score(y_test, y_pred_adjusted))
print("Adjusted Recall:", recall_score(y_test, y_pred_adjusted))
print("Adjusted F1 Score:", f1_score(y_test, y_pred_adjusted))

print("Model training complete.")


import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score

# Initialize XGBoost classifier
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Train on SMOTE-balanced data
xgb_model.fit(x_train, y_train)

# Predict
y_pred_xgb = xgb_model.predict(x_test)

# Evaluate with default threshold
print("\n🔹 XGBoost:")
print("Precision:", precision_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print("F1 Score:", f1_score(y_test, y_pred_xgb))

# Adjust threshold
y_proba_xgb = xgb_model.predict_proba(x_test)[:, 1]
y_pred_adjusted_xgb = (y_proba_xgb >= 0.4).astype(int)

# Evaluate with adjusted threshold
print("\n🔹 XGBoost (Adjusted Threshold 0.4):")
print("Adjusted Precision:", precision_score(y_test, y_pred_adjusted_xgb))
print("Adjusted Recall:", recall_score(y_test, y_pred_adjusted_xgb))
print("Adjusted F1 Score:", f1_score(y_test, y_pred_adjusted_xgb))


from catboost import CatBoostClassifier

# Initialize CatBoost classifier
cat_model = CatBoostClassifier(verbose=0, random_state=42)

# Train on SMOTE-balanced data
cat_model.fit(x_train, y_train)

# Predict
y_pred_cat = cat_model.predict(x_test)

# Evaluate with default threshold
print("\n🔹 CatBoost:")
print("Precision:", precision_score(y_test, y_pred_cat))
print("Recall:", recall_score(y_test, y_pred_cat))
print("F1 Score:", f1_score(y_test, y_pred_cat))

# Adjust threshold
y_proba_cat = cat_model.predict_proba(x_test)[:, 1]
y_pred_adjusted_cat = (y_proba_cat >= 0.4).astype(int)

# Evaluate with adjusted threshold
print("\n🔹 CatBoost (Adjusted Threshold 0.4):")
print("Adjusted Precision:", precision_score(y_test, y_pred_adjusted_cat))
print("Adjusted Recall:", recall_score(y_test, y_pred_adjusted_cat))
print("Adjusted F1 Score:", f1_score(y_test, y_pred_adjusted_cat))
