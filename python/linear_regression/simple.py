import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/Salary_dataset.csv')
print("dataset shape:", df.shape)
print(f"""Columns:
{df.columns}""")
print(df.sample(3))

print("Training Scikit learn model:")
X_train = df['YearsExperience'].values.reshape(-1, 1)
print(X_train.shape)
y_train = df['Salary'].values

lr = LinearRegression()
lr.fit(X=X_train, y=y_train)
print("Scikir learn prediction")
X_test = np.array([10]).reshape(1, -1)
print(X_test.shape)
print(lr.coef_, lr.intercept_)
print(lr.predict(X=X_test))

print("*"*100)
print("Custom model")
print("*"*100)

class SimpleLR:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        assert X_train.ndim == 2, "X_train must be 2D"
        
        x = X_train.reshape(-1)
        y = y_train.reshape(-1)

        num = ((x - x.mean()) * (y - y.mean())).sum()
        den = ((x - x.mean()) ** 2).sum()

        self.m = num / den
        self.b = y.mean() - self.m * x.mean()

    def predict(self, X_test: np.ndarray):
        assert X_test.ndim == 2, "X_test must be 2D"
        return (self.m * X_test + self.b).reshape(-1)

    
lr = SimpleLR()
lr.fit(X_train=X_train, y_train=y_train)
print(lr.predict(X_test))
    

