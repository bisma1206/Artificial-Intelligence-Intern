import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Features and Target
X = df[iris.feature_names]
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Output
print("Predicted Labels:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 1. Confusion Matrix Visualization

# cm = confusion_matrix(y_test, y_pred)

# plt.figure(figsize=(6,4))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#             xticklabels=iris.target_names,
#             yticklabels=iris.target_names)
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix - Iris Logistic Regression")
# plt.show()


# 2. Scatter Plot of Two Features (Petal Length vs Petal Width)

plt.figure(figsize=(6,4))
plt.scatter(X_test['petal length (cm)'], X_test['petal width (cm)'], 
            c=y_pred, cmap='viridis', edgecolor='k', s=80)
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Logistic Regression Predictions")
plt.show()
