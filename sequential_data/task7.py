import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

X_train_reduced = np.load('X_train_reduced.npy')
X_test_reduced = np.load('X_test_reduced.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reduced)
X_test_scaled = scaler.transform(X_test_reduced)

subset_size = 10000
indices = np.random.choice(len(X_train_scaled), subset_size, replace=False)
X_train_subset = X_train_scaled[indices]
y_train_subset = y_train[indices]

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf', 'poly', 'linear']
}

svm_classifier = SVC(random_state=42, cache_size=1000)

random_search = RandomizedSearchCV(
    svm_classifier,
    param_grid,
    n_iter=20,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train_subset, y_train_subset)

print(f"Best parameters found on sampled data: {random_search.best_params_}")
print(f"Cross-validation accuracy on sampled data: {random_search.best_score_:.4f}")

best_svm = SVC(**random_search.best_params_, random_state=42)
best_svm.fit(X_train_scaled, y_train)

train_accuracy_svm = best_svm.score(X_train_scaled, y_train)
test_accuracy_svm = best_svm.score(X_test_scaled, y_test)
print(f"Train accuracy after training on full data: {train_accuracy_svm:.4f}")
print(f"Test accuracy after training on full data: {test_accuracy_svm:.4f}")

y_pred_svm = best_svm.predict(X_test_scaled)
np.save('y_pred_svm.npy', y_pred_svm)

y_pred_dt = np.load('y_pred_dt.npy')
y_pred_rf = np.load('y_pred_rf.npy')

test_accuracy_dt = np.mean(y_pred_dt == y_test)
test_accuracy_rf = np.mean(y_pred_rf == y_test)

print(f"Decision tree: {test_accuracy_dt:.4f}")
print(f"Random forest: {test_accuracy_rf:.4f}")
print(f"SVM: {test_accuracy_svm:.4f}")