# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Loading the dataset
data = pd.read_csv("creditcard.csv")

# Displaying basic statistics of the dataset
print(data.describe())

# Preprocessing the dataset
# Scaling the data using StandardScaler
scaler = StandardScaler()
data['Amount_Scaled'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)

# Splitting the data into training and testing sets
X = data.drop('Class', axis=1).values
y = data['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balancing the classes using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Training the machine learning models
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train_smote, y_train_smote)

# Random Forest
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train_smote, y_train_smote)

# XGBoost
xgb = XGBClassifier(n_estimators=100, random_state=42)
xgb.fit(X_train_smote, y_train_smote)

# Evaluating the performance of the models
models = [('Logistic Regression', logreg), ('Random Forest', rfc), ('XGBoost', xgb)]
for name, model in models:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f'{name}:')
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(f'Precision: {precision_score(y_test, y_pred):.4f}')
    print(f'Recall: {recall_score(y_test, y_pred):.4f}')
    print(f'F1-score: {f1_score(y_test, y_pred):.4f}')
    print(f'AUC-ROC score: {roc_auc_score(y_test, y_prob):.4f}')
    print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}\n')

# Saving the trained model
import pickle
filename = 'fraud_detection_model.sav'
pickle.dump(xgb, open(filename, 'wb'))
