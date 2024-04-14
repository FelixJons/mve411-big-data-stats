from scipy.stats import normaltest
import pandas as pd

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

# Normality test results
normality_results = {}

# Iterating over each column in the DataFrame
for column in X.columns:
    # Dropping missing values as normaltest cannot handle them
    data = X[column].dropna()
    stat, p_value = normaltest(data)
    normality_results[column] = {'Statistic': stat, 'p-value': p_value}

# Optionally, display the results in a sorted manner to see features with the lowest p-values first
results_df = pd.DataFrame(normality_results).T  # Transpose to get features as rows
results_df = results_df.sort_values(by='p-value')
print(results_df.head())  # Print results for the first few features

# Optionally, you can check how many features have p-values indicating non-normal distribution
alpha = 0.05  # significance level
non_normal_features = results_df[results_df['p-value'] < alpha]
print(f"Number of non-normal features: {len(non_normal_features)} out of {len(X.columns)} features")
