# Machine-Learning-Practice-Activity-
Customer Churn Analysis and Prediction using Random Forest Machine Learning Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
df = pd.read_csv("Data_for_ML.csv")
2
 
3
df.head()
print(df.shape)
2
 
3
df.info()
df.isnull().sum()
sns.countplot(x='churn', data=df)
2
plt.title('Customer Churn Distribution')
3
plt.show()
sns.boxplot(x='churn',
2
y='tenure_months',
3
data=df)
4
 
5
plt.title('Tenure Months vs Churn')
6
plt.show()
sns.boxplot(x='churn',
            y='monthly_charges',
            data=df)

plt.title('Monthly Charges vs Churn')
plt.show()
sns.countplot(x='contract_type',
              hue='churn',
              data=df)

plt.xticks(rotation=45)
plt.title('Contract Type vs Churn')
plt.show()
categorical_columns = [
    'contract_type',
    'tech_support',
    'internet_service',
    'payment_method',
    'paperless_billing',
    'has_partner'
]

encoder = LabelEncoder()

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])
``X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues')

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

importance.head(10)
plt.figure(figsize=(10,6))

sns.barplot(
    data=importance.head(10),
    x='Importance',
    y='Feature'
)

plt.title('Top 10 Important 
# Customer Churn Prediction Using Machine Learning

## Project Overview

This project develops a supervised machine learning model to predict customer churn in a telecommunications company. The objective is to identify customers who are likely to leave the service and support customer retention strategies.

---

## Business Problem

Customer churn is a major challenge for subscription-based businesses. Losing customers can reduce revenue and increase acquisition costs. By predicting churn, companies can proactively engage at-risk customers and improve retention.

---

## Dataset Features

The dataset contains customer demographic, contract, billing, and service-related information, including:

- Tenure Months
- Contract Type
- Monthly Charges
- Total Charges
- Internet Service
- Technical Support
- Payment Method
- Paperless Billing
- Partner Status
- Support Calls Last Year

Target Variable:

- Churn (0 = No Churn, 1 = Churn)

---

## Machine Learning Method

Model Used:

- Random Forest Classifier

Machine Learning Type:

- Supervised Learning
- Classification

---

## Tools and Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- GitHub

---

## Project Workflow

1. Data Loading
2. Data Understanding
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing
5. Feature Encoding
6. Train-Test Split
7. Model Development
8. Model Evaluation
9. Feature Importance Analysis
10. Business Recommendations

---

## Key Insights

The analysis identified several important factors influencing customer churn:

- Customer tenure
- Contract type
- Monthly charges
- Support call frequency
- Technical support availability

Customers with shorter tenure, higher monthly charges, and frequent support calls are generally more likely to churn.

---

## Business Recommendation

- Improve onboarding programs for new customers.
- Provide incentives for longer-term contracts.
- Reduce service issues causing support calls.
- Offer proactive retention campaigns for high-risk customers.

---

## Author

Edwin Hindom

Machine Learning Customer Churn Analysis Project
