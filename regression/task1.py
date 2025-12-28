import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

file_name = "regression_insurance.csv"
data = pd.read_csv(file_name)
X = data.drop(columns=['charges'])
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_f = ['age', 'bmi', 'children']
cate_f = ['sex', 'smoker', 'region']

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_f])
X_test_num = scaler.transform(X_test[num_f])

encoder = OneHotEncoder(drop='first', sparse_output=False)
X_train_cate = encoder.fit_transform(X_train[cate_f])
X_test_cate = encoder.transform(X_test[cate_f])

X_train_p = np.concatenate([X_train_num, X_train_cate], axis=1)
X_test_p = np.concatenate([X_test_num, X_test_cate], axis=1)
names = encoder.get_feature_names_out(cate_f)
feature_names = num_f + list(names)

model = LinearRegression()
model.fit(X_train_p, y_train)
for name, coef in zip(feature_names, model.coef_):
    print(f"{name}: {coef:.3f}")
print(f"Intercept: {model.intercept_:.3f}")

y_train_p = model.predict(X_train_p)
y_test_p = model.predict(X_test_p)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_p))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_p))
print(f"Train RMSE: {train_rmse:.3f}")
print(f"Test RMSE: {test_rmse:.3f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_p, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Charges', fontsize=12)
plt.ylabel('Predicted Charges', fontsize=12)
plt.title('Linear Regression: Predicted vs Actual Charges (Test Set)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()