import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold

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

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Create a VarianceThreshold feature selector
sel = VarianceThreshold()
X_train_scaled = sel.fit_transform(X_train_scaled)
X_test_scaled = sel.transform(X_test_scaled)

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
    "KNN": KNeighborsClassifier(n_neighbors=3),
}

# Number of features to evaluate
max_features = min(100, X_train_scaled.shape[1])  # Limit max features to 100 or total features count
feature_range = range(1, max_features + 1, 5)  # Evaluate every 5 features

# Prepare to track best performance
best_accuracy = 0
best_k = 0
performance_history = []

# Evaluate feature count impact
for k in feature_range:
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    # Calculate average cross-validation score
    average_scores = {}
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_train_selected, y_train, cv=5)
        average_scores[name] = np.mean(scores)

    # Average the scores across all classifiers
    average_score = np.mean(list(average_scores.values()))
    performance_history.append((k, average_score))

    # Check if this is the best performance we've seen so far
    if average_score > best_accuracy:
        best_accuracy = average_score
        best_k = k

# Plotting the performance
ks, scores = zip(*performance_history)
plt.figure(figsize=(10, 5))
plt.plot(ks, scores, marker='o')
plt.title('Model Accuracy by Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Cross-validated Accuracy')
plt.grid(True)
plt.show()

print(f"Best number of features: {best_k} with accuracy: {best_accuracy:.2f}")

# Evaluate the best configuration on the test set
selector = SelectKBest(score_func=f_classif, k=best_k)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

for name, clf in classifiers.items():
    clf.fit(X_train_selected, y_train)
    y_pred = clf.predict(X_test_selected)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} with {best_k} features Test Accuracy: {test_accuracy:.2f}")
