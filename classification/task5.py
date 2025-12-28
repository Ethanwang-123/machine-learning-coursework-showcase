import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd

X_train_reduced = np.load('X_train_reduced.npy')
X_test_reduced = np.load('X_test_reduced.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

param_grid = {
    'max_depth': [5, 10],
    'min_samples_split': [2, 5],
    #'min_samples_leaf': [1, 4],
    'criterion': ['gini', 'entropy']
}

dt_classifier = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(
    dt_classifier, 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_reduced, y_train)

results = pd.DataFrame(grid_search.cv_results_)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

best_dt = grid_search.best_estimator_

train_accuracy = best_dt.score(X_train_reduced, y_train)
test_accuracy = best_dt.score(X_test_reduced, y_test)
print(f"Decision tree train accuracy: {train_accuracy:.4f}")
print(f"Decision tree test accuracy: {test_accuracy:.4f}")

y_pred_dt = best_dt.predict(X_test_reduced)
np.save('y_pred_dt.npy', y_pred_dt)