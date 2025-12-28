import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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

with pm.Model() as bayesian_model:
    intercept = pm.Normal('intercept', mu=0, sigma=20)

    coefficients = pm.Normal('coefficients', mu=0, sigma=10, shape=X_train_p.shape[1])

    sigma = pm.HalfNormal('sigma', sigma=10)

    mu = intercept + pm.math.dot(X_train_p, coefficients)

    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=y_train.values)
    
    trace = pm.sample(1000, tune=500, chains=2, return_inferencedata=True, 
                     random_seed=42, progressbar=True)

posterior_means = az.summary(trace, var_names=['intercept', 'coefficients', 'sigma'])
intercept_mean = posterior_means.loc['intercept', 'mean']
print(f"Intercept: {intercept_mean:.3f}")

for i, name in enumerate(feature_names):
    coefficients_mean = posterior_means.loc[f'coefficients[{i}]', 'mean']
    print(f"{name}: {coefficients_mean:.3f}")

sigma_mean = posterior_means.loc['sigma', 'mean']
print(f"sigma: {sigma_mean:.3f}")