import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from fairlearn.metrics import MetricFrame, demographic_parity_difference


adult = fetch_openml(name="adult", version=2, as_frame=True)
df = adult.frame

X = df.drop("class", axis=1)
y = df["class"]

X = pd.get_dummies(X)


sensitive_features = df[['age']].copy()


sensitive_features['age_group'] = sensitive_features['age'].apply(
    lambda x: 'old' if x >= 40 else 'young'
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)


y_pred = model.predict(X)
mf = MetricFrame(
    metrics={"accuracy": accuracy_score},
    y_true=y,
    y_pred=y_pred,
    sensitive_features=sensitive_features['age_group']
)

dp = demographic_parity_difference(
    y_true=y,
    y_pred=y_pred,
    sensitive_features=sensitive_features['age_group']
)

print("Fairness Assessment Results:")
print(mf.overall)

print("\nMetrics by group:")
print(mf.by_group)

print("\nDemographic Parity Difference:", dp)