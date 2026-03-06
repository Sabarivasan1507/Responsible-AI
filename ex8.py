import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

data = {
    'age': [25, 45, 32, 50, 22, 38, 41, 29, 55, 33],
    'sex': ['Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'income': [50000, 85000, 35000, 120000, 28000, 70000, 60000, 45000, 95000, 52000],
    'zip_code': [90210, 10001, 90210, 30301, 10001, 30301, 90210, 10001, 30301, 90210],
    'loan_status': [1, 1, 0, 1, 0, 1, 1, 0, 1, 0]
}
df_initial = pd.DataFrame(data)
df_initial.to_csv('credit_data.csv', index=False)


df = pd.read_csv('credit_data.csv')

protected_attribute = 'sex'
outcome = 'loan_status'

correlation = df.groupby([protected_attribute, 'zip_code']).size().unstack(fill_value=0)
print("--- Proxy Variable Check (Correlation) ---")
print(correlation)

features = ['age', 'income', 'zip_code']
X = df[features]
y = df[outcome]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

test_results = pd.DataFrame({
    'actual': y_test, 
    'predicted': y_pred, 
    protected_attribute: df.loc[X_test.index, protected_attribute] 
})

approval_rates = test_results.groupby(protected_attribute)['predicted'].mean() * 100
print(f"\n--- Approval Rates (%) ---\n{approval_rates}")

def false_negative_rate(group):
    actual_pos = group['actual'] == 1
    pred_neg = group['predicted'] == 0
    
    fn = group[actual_pos & pred_neg].shape[0]
    total_actual_pos = group[actual_pos].shape[0]
    
    return (fn / total_actual_pos) if total_actual_pos > 0 else 0

fnr_rates = test_results.groupby(protected_attribute).apply(false_negative_rate) * 100
print(f"\n--- False Negative Rates (FNR) (% or Error) ---\n{fnr_rates}")