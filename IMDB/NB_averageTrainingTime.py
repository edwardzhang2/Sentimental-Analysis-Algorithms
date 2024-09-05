import pandas as pd

# Load the CSV data into a pandas DataFrame
df = pd.read_csv('NaiveBayes_results.csv')

# Group the data by 'Max Features', calculate the mean 'Accuracy' and 'F1 Score' for each group
grouped_accuracy = df.groupby('Smoothing')['Accuracy'].mean()
grouped_f1 = df.groupby('Smoothing')['F1 Score'].mean()

# Convert the Series to a DataFrame and reset the index
avg_accuracy = pd.DataFrame(grouped_accuracy).reset_index()
avg_f1 = pd.DataFrame(grouped_f1).reset_index()

# Merge the DataFrames on 'Smoothing'
avg_metrics = pd.merge(avg_accuracy, avg_f1, on='Smoothing')

# Write the DataFrame to a new CSV file
avg_metrics.to_csv('average_metrics.csv', index=False)
