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

# Dictionary to store the best number of components for each classifier
best_components = {name: (0, 0) for name in classifiers}

# Determine optimal number of components using CV
for name, clf in classifiers.items():
    best_score = 0
    for components in range(10, 21):
        pca = PCA(n_components=components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        cv_scores = cross_val_score(clf, X_train_pca, y_train, cv=kf, n_jobs=-1)
        mean_cv_score = np.mean(cv_scores)
        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_components[name] = (components, best_score)

# Display the best number of components for each classifier
for name, (components, score) in best_components.items():
    print(
        f"Best number of components for {name}: {components} with CV score of {score}"
    )


# Train and evaluate models using the selected PCA components
def plot_predictions(X, y_true, y_pred, title):
    correct = y_true == y_pred
    incorrect = ~correct  # Logical NOT to find incorrect predictions

    plt.figure(figsize=(8, 6))

    # Plot correctly predicted instances in green
    plt.scatter(
        X[correct, 0],
        X[correct, 1],
        alpha=0.7,
        c="green",
        edgecolors="w",
        label="Correct",
        marker="o",
    )

    # Plot incorrectly predicted instances in red
    plt.scatter(
        X[incorrect, 0],
        X[incorrect, 1],
        alpha=0.7,
        c="red",
        edgecolors="w",
        label="Incorrect",
        marker="x",
    )

    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.show()


for name, (components, _) in best_components.items():
    pca = PCA(n_components=components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    clf = classifiers[name]
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(
        f"Test accuracy for {name} using {components} PCA components: {test_accuracy}"
    )

    # Plot correctly predicted instances
    plot_predictions(X_test_pca, y_test, y_pred, f"{name} Correct Predictions")
