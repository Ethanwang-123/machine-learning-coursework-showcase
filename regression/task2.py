import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

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
X_train_t = torch.FloatTensor(X_train_p)
y_train_t = torch.FloatTensor(y_train.values).reshape(-1, 1)
X_test_t = torch.FloatTensor(X_test_p)
y_test_t = torch.FloatTensor(y_test.values).reshape(-1, 1)

class InsuranceNN(nn.Module):
    def __init__(self, input_size):
        super(InsuranceNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

input_size = X_train_p.shape[1]
model = InsuranceNN(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 500
batch_size = 32
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train_t.size()[0])
    
    for i in range(0, X_train_t.size()[0], batch_size):
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train_t[indices], y_train_t[indices]

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    y_train_pred = model(X_train_t).numpy()
    y_test_pred = model(X_test_t).numpy()

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f"Train RMSE: {train_rmse:.3f}")
print(f"Test RMSE: {test_rmse:.3f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Charges', fontsize=12)
plt.ylabel('Predicted Charges', fontsize=12)
plt.title('Neural Network: Predicted vs Actual Charges (Test Set)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()