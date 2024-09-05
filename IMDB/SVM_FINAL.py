import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('IMDB Dataset.csv')

# Preprocessing
encoder = LabelEncoder()
data['sentiment'] = encoder.fit_transform(data['sentiment'])

# Split the dataset into train and test sets
train_reviews, test_reviews, train_labels, test_labels = train_test_split(
    data['review'], data['sentiment'], test_size=0.2, random_state=42)

# Transform the text data to vectors
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
train_vectors = vectorizer.fit_transform(train_reviews)
test_vectors = vectorizer.transform(test_reviews)

# Set parameters for grid search
Cs = []
for i in range(201, 250):
    Cs.append(i/1000)
penalties = ['l1', 'l2']
losses = ['squared_hinge']
duals = [False]

param_combinations = [(C, penalty, loss, dual) for C in Cs for penalty in penalties for loss in losses for dual in duals]

# Define a DataFrame to store the results
results_df = pd.DataFrame(columns=['C', 'Penalty', 'Loss', 'Precision', 'Recall', 'F1 Score', 'Training Time', 'Prediction Time'])

# Loop over all combinations
for params in param_combinations:
    # Unpack parameters
    C, penalty, loss, dual = params
    # Handle combinations that are not compatible
    if (penalty == 'l1' and loss == 'hinge') or (dual and loss == 'squared_hinge' and penalty == 'l1') or (penalty == 'l2' and loss == 'hinge' and not dual):
        print(f"Skipping combination: C={C}, penalty={penalty}, loss={loss}, dual={dual}")
        continue
    print(f"Testing combination: C={C}, penalty={penalty}, loss={loss}, dual={dual}")
    # Create and train the classifier
    svc = LinearSVC(C=C, penalty=penalty, loss=loss, dual=dual, max_iter=10000)
    start_training = time.time()
    svc.fit(train_vectors, train_labels)
    end_training = time.time()
    training_time = end_training - start_training
    # Predict on the test set
    start_prediction = time.time()
    predictions = svc.predict(test_vectors)
    end_prediction = time.time()
    prediction_time = end_prediction - start_prediction
    # Calculate performance metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    # Append the results to the DataFrame
    new_row = {'C': C, 'Penalty': penalty, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'Training Time': training_time, 'Prediction Time': prediction_time}
    new_df = pd.DataFrame(new_row, index=[0])
    results_df = pd.concat([results_df, new_df], ignore_index=True)

# Save the results to a csv file
results_df.to_csv('SVM_results2.csv', index=False)
