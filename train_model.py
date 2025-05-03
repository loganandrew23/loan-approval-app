import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Sample data
data = {
    'credit_score': [720, 690, 610, 580, 750, 660, 640, 710, 690, 680],
    'income': [60000, 55000, 40000, 30000, 80000, 50000, 42000, 70000, 62000, 49000],
    'debt': [15000, 12000, 18000, 25000, 10000, 14000, 20000, 13000, 16000, 17000],
    'employment_years': [5, 4, 2, 1, 6, 3, 2, 5, 4, 3],
    'approved': [1, 1, 0, 0, 1, 1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

# Features and labels
X = df[['credit_score', 'income', 'debt', 'employment_years']]
y = df['approved']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'loan_model.pkl')
print("✅ Model saved as 'loan_model.pkl'")
