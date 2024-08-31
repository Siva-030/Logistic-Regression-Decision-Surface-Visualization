import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Title of the app
st.title('Logistic Regression Decision Surface Visualization')

# Sidebar for user inputs
st.sidebar.header('Hyperparameters')

# Choose the dataset
dataset_name = st.sidebar.selectbox('Select Dataset', ('Make Moons', 'Make Circles', 'Make Classification'))

# Choose logistic regression hyperparameters
C = st.sidebar.slider('Regularization Strength (C)', 0.01, 10.0, 1.0)
max_iter = st.sidebar.slider('Max Iterations', 100, 1000, 200)
tol = st.sidebar.slider('Tolerance (tol)', 0.0001, 0.01, 0.0001, step=0.0001)
dual = st.sidebar.selectbox('Dual', (False, True))
fit_intercept = st.sidebar.selectbox('Fit Intercept', (True, False))
intercept_scaling = st.sidebar.slider('Intercept Scaling', 0.1, 10.0, 1.0)
solver = st.sidebar.selectbox('Solver', ('lbfgs', 'liblinear', 'saga', 'newton-cg', 'sag'))
class_weight = st.sidebar.selectbox('Class Weight', (None, 'balanced'))
penalty = st.sidebar.selectbox('Penalty', ('l2', 'l1', 'elasticnet', 'none'))
warm_start = st.sidebar.selectbox('Warm Start', (False, True))
l1_ratio = st.sidebar.slider('L1 Ratio (only for elasticnet)', 0.0, 1.0, 0.5) if penalty == 'elasticnet' else None

# Adjust the penalty parameter
penalty = None if penalty == 'none' else penalty

# Generate the dataset
def get_dataset(name):
    if name == 'Make Moons':
        return make_moons(noise=0.3, random_state=0)
    elif name == 'Make Circles':
        return make_circles(noise=0.2, factor=0.5, random_state=1)
    else:
        return make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=0, n_clusters_per_class=1)

X, y = get_dataset(dataset_name)

# Split and scale the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression(
    C=C, max_iter=max_iter, tol=tol, dual=dual, fit_intercept=fit_intercept,
    intercept_scaling=intercept_scaling, solver=solver, class_weight=class_weight,
    penalty=penalty, warm_start=warm_start, l1_ratio=l1_ratio
)
model.fit(X_train, y_train)

# Plot decision surface
def plot_decision_surface(X, y, model):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Surface')
    st.pyplot(plt.gcf())

# Display the decision surface
plot_decision_surface(X_train, y_train, model)

# Display model accuracy
accuracy = model.score(X_test, y_test)
st.write(f"Model Accuracy: {accuracy:.2f}")

