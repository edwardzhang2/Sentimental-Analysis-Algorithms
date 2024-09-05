import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('LSTM_results.csv')

# Group the data by 'Embedding Dim' and select the row with the max 'Accuracy' in each group
max_accuracy_data = data.iloc[data.groupby('Embedding Dim')['Accuracy'].idxmax()]

plt.figure(figsize=(10, 8))
for rnn_unit in max_accuracy_data['RNN Units'].unique():
    subset = max_accuracy_data[max_accuracy_data['RNN Units'] == rnn_unit]
    plt.scatter(subset['Epochs'], subset['Accuracy'], label=f'RNN Units={rnn_unit}')

    # Fit a trend line to the data and plot it
    z = np.polyfit(subset['Epochs'], subset['Accuracy'], 1)
    p = np.poly1d(z)
    plt.plot(subset['Epochs'], p(subset['Epochs']), linestyle='--')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Max Accuracy by Epochs and RNN Units with Trend Lines for each Embedding Dim')
plt.show()
