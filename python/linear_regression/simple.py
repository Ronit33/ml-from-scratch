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
y_train = df['Salary']

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

class SimpleLR():
    def __init__(self):
        self.m = 0
        self.b = 0
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        
        assert len(X_train.shape) == 2, "Shape of X_train should be 2D"

        num = ((X_train - X_train.mean()).reshape(-1) * (y_train - y_train.mean())).sum()
        den = ((X_train - X_train.mean())**2).sum()
        
        self.m = num / den
        self.b = y_train.mean() - (self.m * X_train.mean())      
        
        print(self.m, self.b)
    
    def predict(self, X_test: np.ndarray):
        assert len(X_test.shape) == 2, "Shape of X_test should be 2D"
        
        return (self.m * X_test) + self.b
    
lr = SimpleLR()
lr.fit(X_train=X_train, y_train=y_train)
print(lr.predict(X_test))
    

