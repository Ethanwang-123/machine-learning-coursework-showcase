import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

X_train_reduced = np.load('X_train_reduced.npy')
X_test_reduced = np.load('X_test_reduced.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

subset_size = 10000
indices = np.random.choice(len(X_train_reduced), subset_size, replace=False)
X_train_subset = X_train_reduced[indices]
y_train_subset = y_train[indices]

param_distributions = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    #'max_features': ['sqrt', 'log2']
    # min_samples_leaf
}

rf_classifier = RandomForestClassifier(random_state=42, n_jobs=-1)


random_search = RandomizedSearchCV(
    rf_classifier,
    param_distributions,
    n_iter=20,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train_subset, y_train_subset)

print(f"Best parameters (on subset): {random_search.best_params_}")
print(f"Best cross-validation accuracy (on subset): {random_search.best_score_:.4f}")

best_params = random_search.best_params_
best_rf = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
best_rf.fit(X_train_reduced, y_train)

train_accuracy_rf = best_rf.score(X_train_reduced, y_train)
test_accuracy_rf = best_rf.score(X_test_reduced, y_test)

print(f"Random forest train accuracy (full data): {train_accuracy_rf:.4f}")
print(f"Random forest test accuracy (full data):  {test_accuracy_rf:.4f}")

y_pred_rf = best_rf.predict(X_test_reduced)
np.save('y_pred_rf.npy', y_pred_rf)

X_test_original = np.load('X_test_original.npy')
y_test = np.load('y_test.npy')
y_pred_rf = np.load('y_pred_rf.npy')

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

misclassified_indices = np.where(y_pred_rf != y_test)[0]

np.random.seed(42)
selected_indices = np.random.choice(misclassified_indices, 5, replace=False)

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, idx in enumerate(selected_indices):
    img = X_test_original[idx].reshape(3, 32, 32).transpose(1, 2, 0)
    
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(f'True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred_rf[idx]]}', fontsize=10)

plt.tight_layout()
plt.savefig('rf_misclassified_samples.png', dpi=150, bbox_inches='tight')
plt.show()