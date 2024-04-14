import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
file_path_data = "TCGAdata.txt"
file_path_labels = "TCGAlabels.txt"
df_data = pd.read_csv(file_path_data, delim_whitespace=True, header=0, quotechar='"')
df_labels = pd.read_csv(
    file_path_labels, delim_whitespace=True, header=0, quotechar='"'
)
df = pd.merge(df_data, df_labels, left_index=True, right_index=True, how="left")

# Split into train and test set
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]  # Labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Scale test data using the same scaler

# Prepare PCA and cross-validation framework
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
    "KNN": KNeighborsClassifier(n_neighbors=3),
}

# Determine optimal number of components using CV and evaluate performance
for name, clf in classifiers.items():
    best_score = 0
    best_components = 0
    best_training_error = 0
    for components in range(10, 21):
        pca = PCA(n_components=components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        cv_scores = cross_val_score(clf, X_train_pca, y_train, cv=kf, n_jobs=-1)
        mean_cv_score = np.mean(cv_scores)
        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_components = components
            # Evaluate on the training set
            clf.fit(X_train_pca, y_train)
            training_accuracy = accuracy_score(y_train, clf.predict(X_train_pca))
            best_training_error = 1 - training_accuracy

    # Train using the best components and evaluate on the test set
    pca = PCA(n_components=best_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    clf.fit(X_train_pca, y_train)
    test_accuracy = accuracy_score(y_test, clf.predict(X_test_pca))
    test_error = 1 - test_accuracy
    cross_val_error = 1 - best_score

    print(f"{name}:")
    print(f"  Optimal number of PCA components: {best_components}")
    print(f"  Training Error: {best_training_error:.2f}")
    print(f"  Cross-Validation Error: {cross_val_error:.2f}")
    print(f"  Test Error: {test_error:.2f}\n")
