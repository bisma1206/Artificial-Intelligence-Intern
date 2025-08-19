# 2. Example 1: Linear Regression (Salary vs Experience)

# This predicts salary based on years of experience.

# Task 3 - Basic ML Implementation

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample dataset (Salary vs Experience)
data = {
    'YearsExperience': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 65000, 70000, 80000, 85000, 90000, 100000, 110000]
}

df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df[['YearsExperience']]
y = df['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Output results
print("Predicted salaries:", y_pred)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Visualization
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience (Linear Regression)")
plt.legend()
plt.show()
