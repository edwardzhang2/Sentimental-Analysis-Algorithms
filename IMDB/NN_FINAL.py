import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers as keras_layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = pd.read_csv('IMDB Dataset.csv')
reviews = data['review']
sentiments = data['sentiment']

# Preprocessing
encoder = LabelEncoder()
sentiments = encoder.fit_transform(sentiments)

# Tokenizing the text - converting the words, letters into counts or vectors
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

# Padding sequences with zeros to make all sequences of same length
maxlen = 100
X = pad_sequences(sequences, padding='post', maxlen=maxlen)

# Converting labels to binary format
y = to_categorical(sentiments)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameters
neurons = [16, 32, 64]
layers = [1, 2, 3]

# Loop over hyperparameters
for n in neurons:
    for l in layers:
        # Define model
        model = Sequential()
        model.add(keras_layers.Embedding(input_dim=vocab_size, output_dim=50, input_length=maxlen))
        for _ in range(l):
            model.add(keras_layers.Dense(n, activation='relu'))
        model.add(keras_layers.Dense(2, activation='softmax'))

        # Compile model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Fit model
        model.fit(X_train, y_train, epochs=10, verbose=False, validation_data=(X_test, y_test), batch_size=10)

        # Evaluate model
        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print(f"Training Accuracy: {accuracy:.4f}")
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print(f"Testing Accuracy:  {accuracy:.4f}")
        print(f"Number of neurons: {n}, Number of layers: {l}")
