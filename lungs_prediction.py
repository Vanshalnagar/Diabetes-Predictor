import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

df = pd.read_csv(r'C:/Users/Vanshal/Desktop/Machine_learning/Diabetes-Predictor/lung_cancer_prediction.csv')

features = ['Age', 'Gender', 'Smoking_Status', 'Second_Hand_Smoke',
            'Air_Pollution_Exposure', 'Occupation_Exposure', 'Rural_or_Urban',
            'Socioeconomic_Status', 'Healthcare_Access', 'Treatment_Access',
            'Clinical_Trial_Access', 'Language_Barrier', 'Delay_in_Diagnosis',
            'Family_History', 'Indoor_Smoke_Exposure', 'Tobacco_Marketing_Exposure']
target = 'Final_Prediction'

df = df[features + [target]]

binary_mappings = {
    'Gender': {'Male': 1, 'Female': 0},
    'Second_Hand_Smoke': {'Yes': 1, 'No': 0},
    'Occupation_Exposure': {'Yes': 1, 'No': 0},
    'Rural_or_Urban': {'Urban': 1, 'Rural': 0},
    'Language_Barrier': {'Yes': 1, 'No': 0},
    'Delay_in_Diagnosis': {'Yes': 1, 'No': 0},
    'Family_History': {'Yes': 1, 'No': 0},
    'Indoor_Smoke_Exposure': {'Yes': 1, 'No': 0},
    'Tobacco_Marketing_Exposure': {'Yes': 1, 'No': 0},
    'Final_Prediction': {'Yes': 1, 'No': 0}
}
for col, mapping in binary_mappings.items():
    df[col] = df[col].map(mapping)

multi_cats = ['Smoking_Status', 'Air_Pollution_Exposure',
              'Socioeconomic_Status', 'Healthcare_Access',
              'Treatment_Access', 'Clinical_Trial_Access']
for col in multi_cats:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

df = df.dropna()

X = df[features]
y = df[target]

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

model = RandomForestClassifier(random_state=42)
model.fit(X_res, y_res)
y_pred = model.predict(X_res)
acc = accuracy_score(y_res, y_pred)

print("[RandomForest] Training accuracy on SMOTE data:", round(acc, 3))



