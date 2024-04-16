import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
import math


def shuffle(arr, percentage: int):
    """

    :param arr: The array of booleans to shuffle
    :param percentage: Out of 100
    :return: The new array
    """
    number_of_elemets_to_change = math.floor(percentage / len(arr) * 100)
    array_indexes_to_change = []

    # Make a list of the indexes to change
    for i in range(number_of_elemets_to_change):
        array_indexes_to_change.append(np.random.randint(0, len(arr) - 1))

    for index in array_indexes_to_change:
        element = arr[index]
        arr[index] = not element

    return arr


# Parameters
test_size = 0.2

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1),
    "KNN": KNeighborsClassifier(n_neighbors=20),
}


file_path_data = "TCGAdata.txt"
file_path_labels = "TCGAlabels.txt"
df_data = pd.read_csv(file_path_data, delim_whitespace=True, header=0, quotechar='"')
df_labels = pd.read_csv(
    file_path_labels, delim_whitespace=True, header=0, quotechar='"'
)
df = pd.merge(df_data, df_labels, left_index=True, right_index=True, how="left")

X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]  # Labels
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

randomized_y_train_dict = {}

mislabeled_classifier_accuracy = {
    "Logistic Regression": {},
    "Random Forest": {},
    "KNN": {},
}


for percentage in range(1, 101):
    y_train_list = y_train.to_numpy()
    y_train_mislabeled = shuffle(y_train_list, percentage / 100)
    randomized_y_train_dict[percentage] = y_train_mislabeled


for name, clf in classifiers.items():
    for percentage in range(1, 101):
        y_train_mislabeled = randomized_y_train_dict[percentage]

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=20)),
                ("classifier", clf),
            ]
        )

        pipeline.fit(X_train, y_train_mislabeled)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mislabeled_classifier_accuracy[name][percentage] = accuracy
        print(f"{name} with {percentage}% mislabeled data: {accuracy}")

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, (name, accuracies) in enumerate(mislabeled_classifier_accuracy.items()):
    axs[i].plot(list(accuracies.keys()), list(accuracies.values()), label=name)
    axs[i].set_xlabel("Percentage")
    axs[i].set_ylabel("Accuracy")
    axs[i].set_title(name)

plt.tight_layout()
plt.show()
