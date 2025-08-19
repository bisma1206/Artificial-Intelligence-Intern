**Basic Machine Learning Implementation**

## 📌 Objective
Implement a **basic Machine Learning model in Python** using  scikit-learn  and pandas.

---

## 🚀 Requirements
- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn (optional, for visualization)

**Install dependencies**

pip install pandas scikit-learn matplotlib seaborn

## 🔹 Example 1: Linear Regression (Salary vs Experience)

### 📖 Description
This model predicts **Salary** based on **Years of Experience**.  
It uses LinearRegression  from **scikit-learn**.

### 🧾 Dataset (Sample)
| YearsExperience | Salary |
|-----------------|--------|
| 1               | 45000  |
| 2               | 50000  |
| 3               | 60000  |
| ...             | ...    |

### 🖥️ Steps
1. Load dataset into pandas DataFrame.  
2. Split into training & testing sets.  
3. Train a **Linear Regression model**.  
4. Make predictions.  
5. Visualize results with a regression line.  

### 📊 Output
- Predicted salaries for test data.  
- Mean Squared Error (MSE).  
- Scatter plot (actual data) + regression line.  

---

## 🔹 Example 2: Logistic Regression (Iris Dataset)

### 📖 Description
This model classifies **Iris flowers** into three species:
- Setosa  
- Versicolor  
- Virginica  

based on **sepal & petal measurements**.

### 🧾 Dataset (Iris)
- **Samples**: 150 (50 per species)  
- **Features**: sepal length, sepal width, petal length, petal width  
- **Target (labels)**:  
  - 0 = Setosa  
  - 1 = Versicolor  
  - 2 = Virginica  

### 🖥️ Steps
1. Load the Iris dataset using scikit-learn.  
2. Split into training & testing sets.  
3. Train a **Logistic Regression model**.  
4. Predict species of flowers.  
5. Evaluate with accuracy & confusion matrix.  

### 📊 Output
- Predicted labels for test set.  
- Accuracy score (~95%).  
- Confusion matrix heatmap.  
- Scatter plot (Petal length vs Petal width).  

---

## 🏆 Results
- **Linear Regression**: Learns the relationship between years of experience and salary.  
- **Logistic Regression**: Classifies Iris flowers with high accuracy.  

---

## ✨ Key Learnings
- Load and preprocess datasets in pandas.  
- Split data into training and testing sets.  
- Train and evaluate ML models with scikit-learn.  
- Visualize predictions using matplotlib and seaborn.  
