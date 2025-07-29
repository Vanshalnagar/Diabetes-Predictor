import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv(r'C:/Users/Vanshal/Desktop/Machine_learning/Diabetes-Predictor/Covid Dataset.csv')

for column in df.columns[:-1]:
    df[column] = df[column].map({'Yes': 1, 'No': 0})

df['COVID-19'] = df['COVID-19'].map({'Yes': 1, 'No': 0})

print("Missing values:\n", df.isnull().sum())

X = df.drop('COVID-19', axis=1)
y = df['COVID-19']

selector = SelectKBest(chi2, k=10)
selector.fit(X, y)
cols = selector.get_support(indices=True)
selected_features = X.columns[cols]
print("Selected Features:", selected_features)

X = X[selected_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.4f}")



print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

cv_scores = cross_val_score(model, X, y, cv=5)
print(f"\nCross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("\nBest Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Model Accuracy after Tuning: {final_accuracy:.4f}")

