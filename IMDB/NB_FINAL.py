import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import re
import time
import math
from collections import defaultdict

# Load the dataset
data = pd.read_csv('IMDB Dataset.csv')

# Preprocessing functions
def remove_tags(string):
    """Function to remove HTML tags and URLs from a given string"""
    string = re.sub('<.*?>', '', string)  # Remove HTML tags
    string = re.sub(r'http\S+|www\S+|https\S+', '', string, flags=re.MULTILINE)  # Remove URLs
    string = re.sub(r'\W', ' ', string)  # Remove non-alphanumeric characters
    return string.lower()

def lemmatize_text(text):
    """Function to lemmatize the given text"""
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])

def group_by_label(x, y, labels):
    """Group the data by its label"""
    return {l: x[np.where(y == l)] for l in labels}

def laplace_smoothing(n_label_items, vocab, word_counts, word, text_label, smoothing):
    """Calculate the probability with Laplace smoothing"""
    a = word_counts[text_label][word] + smoothing
    b = n_label_items[text_label] + len(vocab) * smoothing
    return math.log(a / b)

def fit(x, y, labels):
    """Calculate prior probabilities for the given labels"""
    n = len(x)
    grouped_data = group_by_label(x, y, labels)
    n_label_items = {l: len(data) for l, data in grouped_data.items()}
    log_label_priors = {l: math.log(n_label_items[l] / n) for l in labels}
    return n_label_items, log_label_priors

def predict(n_label_items, vocab, word_counts, log_label_priors, labels, x, smoothing):
    """Predict using the Naive Bayes model"""
    result = []
    for text in x:
        label_scores = {l: log_label_priors[l] for l in labels}
        words = set(vec.build_analyzer()(text))
        for word in words:
            if word not in vocab: continue
            for l in labels:
                log_w_given_l = laplace_smoothing(n_label_items, vocab, word_counts, word, l, smoothing)
                label_scores[l] += log_w_given_l
        result.append(max(label_scores, key=label_scores.get))
    return result

# Data preprocessing
data['review'] = data['review'].apply(remove_tags) 
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
data['review'] = data['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
data['review'] = data['review'].apply(lemmatize_text)

# Split the data and encode labels
reviews = data['review'].values
labels = data['sentiment'].values
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

# Hyperparameter optimization
smoothing_values = np.arange(0.1, 2.0, 0.1)  
max_features_values = range(1000, 3000, 100)
best_accuracy = 0

# Fixed test size for the experiments
test_size = 0.2

# DataFrame to store results
results_df = pd.DataFrame(columns=['Max Features', 'Smoothing', 'Accuracy', 'Precision', 'F1 Score', 'Train Time', 'Predict Time'])

# Hyperparameter search loop
for max_features in max_features_values:
    for smoothing in smoothing_values:
        
        # Split the data
        train_sentences, test_sentences, train_labels, test_labels = train_test_split(reviews, encoded_labels, test_size=test_size, stratify = encoded_labels)

        # Vectorize the text data
        vec = CountVectorizer(max_features=max_features)
        X = vec.fit_transform(train_sentences).toarray()
        vocab = vec.get_feature_names_out()

        # Count word occurrences by label
        word_counts = {l: defaultdict(int) for l in range(2)}
        for i in range(X.shape[0]):
            l = train_labels[i]
            for j, word in enumerate(vocab):
                word_counts[l][word] += X[i][j]

        # Train and predict
        start_train = time.time()
        labels = [0,1]
        n_label_items, log_label_priors = fit(train_sentences, train_labels, labels)
        end_train = time.time()

        start_predict = time.time()
        predictions = predict(n_label_items, vocab, word_counts, log_label_priors, labels, test_sentences, smoothing)
        end_predict = time.time()

        # Measure performance
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        
        train_time = end_train - start_train
        predict_time = end_predict - start_predict
        
        # Append to results dataframe
        results_df = results_df.append({
            'Max Features': max_features, 
            'Smoothing': smoothing, 
            'Accuracy': accuracy, 
            'Precision': precision, 
            'F1 Score': f1, 
            'Train Time': train_time, 
            'Predict Time': predict_time
        }, ignore_index=True)
        
        # Print progress
        print(f"Max features: {max_features}, Smoothing: {smoothing}, Accuracy: {accuracy:.4f}")
        
        # Track the best accuracy and associated hyperparameters
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_max_features = max_features
            best_smoothing = smoothing

print(f"\nBest max features: {best_max_features}")
print(f"Best smoothing: {best_smoothing}")
print(f"Best accuracy: {best_accuracy:.4f}")

# Save results to CSV
results_df.to_csv('NaiveBayes_results.csv', index=False)

