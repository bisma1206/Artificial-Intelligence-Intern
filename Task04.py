# Task 4 – Intermediate AI Task

# Objective: Implement Decision Tree and Random Forest classifiers on Iris dataset

# 1. Import Libraries

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Load Dataset

iris = load_iris()
X = iris.data       # Features
y = iris.target     # Target labels

# Convert into DataFrame for visualization 
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
print("\nSample Data:")
print(df.head())

# 3. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Decision Tree Classifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# Evaluate Decision Tree

dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_cm = confusion_matrix(y_test, y_pred_dt)

print("\nDecision Tree Accuracy:", dt_accuracy)
print("Decision Tree Confusion Matrix:\n", dt_cm)

# 5. Random Forest Classifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluate Random Forest

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_cm = confusion_matrix(y_test, y_pred_rf)

print("\nRandom Forest Accuracy:", rf_accuracy)
print("Random Forest Confusion Matrix:\n", rf_cm)

# 6. Visualization (Confusion Matrix for Random Forest)

plt.figure(figsize=(6,4))
sns.heatmap(rf_cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
