import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.decomposition import PCA

# Load data
# Adjust the path if needed

df = pd.read_csv('data/train.csv')

# Select features and target (adjust columns as needed)
X = df.iloc[:, [2, 4, 5, 6, 7, 8, 9, 10, 11]].values  # Example: select relevant columns
y = df.iloc[:, 1].values  # Survived

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)

# Optionally scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model definition and training
model = LogisticRegression(max_iter=1000)
history = model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot error reduction (loss curve, if available)
if hasattr(model, 'n_iter_'):
    # Sklearn's LogisticRegression does not expose loss curve, so we plot log loss per epoch manually
    losses = []
    for i in range(1, model.n_iter_[0]+1):
        model_partial = LogisticRegression(max_iter=i, warm_start=True, solver='lbfgs')
        model_partial.fit(X_train, y_train)
        y_pred_proba_partial = model_partial.predict_proba(X_train)[:, 1]
        losses.append(log_loss(y_train, y_pred_proba_partial))
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(losses)+1), losses)
    plt.xlabel('Iteration')
    plt.ylabel('Log Loss')
    plt.title('Training Log Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.show()

# Visualize decision boundary (PCA to 2D)
pca = PCA(n_components=2)
X_vis = pca.fit_transform(X_scaled)
model_vis = LogisticRegression(max_iter=1000)
model_vis.fit(X_vis, y)

# Create mesh grid
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap='RdYlBu', edgecolor='k', s=30)
plt.title('Decision Boundary (PCA-reduced)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True, alpha=0.3)
plt.show()