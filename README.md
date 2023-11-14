import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from catboost import CatBoostClassifier

# Load the dataset
df = pd.read_csv('c:\\Users\\User\\Desktop\\pyt\\heart.csv')

# Assume 'target' is the column to be predicted
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multiple models
rf_model = RandomForestClassifier()
gb_model = GradientBoostingClassifier()
cb_model = CatBoostClassifier()

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
cb_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)
gb_pred = gb_model.predict(X_test)
cb_pred = cb_model.predict(X_test)

# Ensemble predictions (simple majority vote)
ensemble_pred = np.round((rf_pred + gb_pred + cb_pred) / 3)

# Evaluate individual model performances
print('Random Forest Accuracy: ', accuracy_score(y_test, rf_pred))
print('Random Forest Confusion Matrix: \n', confusion_matrix(y_test, rf_pred))

print('Gradient Boosting Accuracy: ', accuracy_score(y_test, gb_pred))
print('Gradient Boosting Confusion Matrix: \n', confusion_matrix(y_test, gb_pred))

print('CatBoost Accuracy: ', accuracy_score(y_test, cb_pred))
print('CatBoost Confusion Matrix: \n', confusion_matrix(y_test, cb_pred))

# Evaluate ensemble performance
print('Ensemble Accuracy: ', accuracy_score(y_test, ensemble_pred))
print('Ensemble Confusion Matrix: \n', confusion_matrix(y_test, ensemble_pred))

# ROC curve for CatBoost (you can modify this for other models if needed)
y_score1 = cb_model.predict_proba(X_test)[:, 1]
false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y_test, y_score1)

print('roc_auc_score for Catboost: ', roc_auc_score(y_test, y_score1))

# Plot ROC curves
plt.subplots(1, figsize=(10, 10))
plt.title('Receiver Operating Characteristic - Catboost')
plt.plot(false_positive_rate1, true_positive_rate1)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
