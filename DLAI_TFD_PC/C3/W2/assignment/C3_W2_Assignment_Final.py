import io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

import unittests

with open("data/bbc-text.csv", 'r') as csvfile:
    print(f"First line (header) looks like this:\n\n{csvfile.readline()}")
    print(f"The second line (first data point) looks like this:\n\n{csvfile.readline()}")

VOCAB_SIZE = 1000
EMBEDDING_DIM = 16
MAX_LENGTH = 120
TRAINING_SPLIT = 0.8

data_dir = "data/bbc-text.csv"
data = np.loadtxt(data_dir, delimiter=',', skiprows=1, dtype='str', comments=None)
print(f"Shape of the data: {data.shape}")
print(f"{data[0]}\n{data[1]}")

# Test the function
print(f"There are {len(data)} sentence-label pairs in the dataset.\n")
print(f"First sentence has {len((data[0, 1]).split())} words.\n")
print(f"The first 5 labels are {data[:5, 0]}")

# GRADED FUNCTIONS: train_val_datasets
def train_val_datasets(data):
    '''
    Splits data into traning and validations sets

    Args:
        data (np.array): array with two columns, first one is the label, the second is the text

    Returns:
        (tf.data.Dataset, tf.data.Dataset): tuple containing the train and validation datasets
    '''

    ### START CODE HERE ###
    # Compute the number of sentences that will be used for training (should be an integer)
    train_size = int(len(data) * TRAINING_SPLIT)
    # Slice the dataset to get only the texts. Remember that texts are on the second column
    texts = data[:, 1]
    # Slice the dataset to get only the labels. Remember that labels are on the first column
    labels = data[:, 0]

    # Split the sentences and labels into train/validation splits
    train_texts = texts[:train_size]
    validation_texts = texts[train_size:]
    train_labels = labels[:train_size]
    validation_labels = labels[train_size:]
    # create the train and validation datasets from the splits
    train_dataset = tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_texts, validation_labels))
    ### END CODE HERE ###

    return train_dataset, validation_dataset

# Create the datasets
train_dataset, validation_dataset = train_val_datasets(data)

print(f"There are {train_dataset.cardinality()} sentence-label pairs for training.\n")
print(f"There are {validation_dataset.cardinality()} sentence-label pairs for validation.\n")

# Test your code!
unittests.test_train_val_datasets(train_val_datasets)

## Vectorization - Sequences and padding
def standardize_func(sentence):
    """
    Removes a list of stopwords

    Args:
        sentence (tf.string): sentence to remove the stopwords from

    Returns:
        sentence (tf.string): lowercase sentence without the stopwords
    """
    # List of stopwords
    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
                 "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
                 "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
                 "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
                 "into", "is", "it", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on",
                 "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
                 "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
                 "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
                 "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "why",
                 "with", "would", "you", "your", "yours", "yourself", "yourselves", "'m", "'d", "'ll", "'re", "'ve",
                 "'s", "'d"]

    # Sentence converted to lowercase-only
    sentence = tf.strings.lower(sentence)

    # Remove stopwords
    for word in stopwords:
        if word[0] == "'":
            sentence = tf.strings.regex_replace(sentence, rf"{word}\b", "")
        else:
            sentence = tf.strings.regex_replace(sentence, rf"\b{word}\b", "")

    # Remove punctuation
    sentence = tf.strings.regex_replace(sentence, r'[!"#$%&()\*\+,-\./:;<=>?@\[\\\]^_`{|}~\']', "")

    return sentence

test_sentence = "Hello! We're just about to see this function in action =)"
standardized_sentence = standardize_func(test_sentence)
print(f"Original sentence is:\n{test_sentence}\n\nAfter standardizing:\n{standardized_sentence}")

### Exercise 2: fit_vectorizer

# GRADED FUNCTION: fit_tvectorizer
def fit_vectorizer(train_sentences, standardize_func):
    '''
    Defines and adapts the text vectorizer

    Args:
        train_sentences (tf.data.Dataset): sentences from the train dataset to fit the TextVectorization layer
        standardize_func (FunctionType): function to remove stopwords and punctuation, and lowercase texts.
    Returns:
        TextVectorization: adapted instance of TextVectorization layer
    '''
    ### START CODE HERE ###
    # Instantiate the Tokenizer class, passing in the correct values for num_words and oov_token
    vectorizer = tf.keras.layers.TextVectorization(
        standardize=standardize_func,
        max_tokens=VOCAB_SIZE,
        output_sequence_length=MAX_LENGTH
    )
    # Fit the tokenizer to the training sentences
    vectorizer.adapt(train_sentences)
    ### END CODE HERE ###

    return vectorizer


# %%
# Create the vectorizer
text_only_dataset = train_dataset.map(lambda text, label: text)
vectorizer = fit_vectorizer(text_only_dataset, standardize_func)
vocab_size = vectorizer.vocabulary_size()

print(f"Vocabulary contains {vocab_size} words\n")

# Test your code!
unittests.test_fit_vectorizer(fit_vectorizer, standardize_func)
# %% md
### Exercise 3: fit_label_encoder
def fit_label_encoder(train_labels, validation_labels):
    """Creates an instance of a StringLookup, and trains it on all labels

    Args:
        train_labels (tf.data.Dataset): dataset of train labels
        validation_labels (tf.data.Dataset): dataset of validation labels

    Returns:
        tf.keras.layers.StringLookup: adapted encoder for train and validation labels
    """
    ### START CODE HERE ###

    # Join the two label datasets
    labels = train_labels.concatenate(validation_labels)
    # Instantiate the StringLookup layer without OOV tokens and without a mask token
    label_encoder = tf.keras.layers.StringLookup(
        mask_token=None,  # No mask token
        oov_token=None,    # No OOV token
        num_oov_indices = 0
    )
    # Fit the StringLookup layer on the labels
    label_encoder.adapt(labels)
    # Remove the None token by setting the vocabulary to start from index 1
    label_encoder.vocabulary = label_encoder.get_vocabulary()[1:]  # Skip the first token (None)

    ### END CODE HERE ###

    return label_encoder

# Create the label encoder
train_labels_only = train_dataset.map(lambda text, label: label)
validation_labels_only = validation_dataset.map(lambda text, label: label)

label_encoder = fit_label_encoder(train_labels_only, validation_labels_only)

print(f'Unique labels: {label_encoder.get_vocabulary()}')

# Test your code!
unittests.test_fit_label_encoder(fit_label_encoder)

### Exercise 4: preprocess_dataset

def preprocess_dataset(dataset, text_vectorizer, label_encoder):
    """Apply the preprocessing to a dataset

    Args:
        dataset (tf.data.Dataset): dataset to preprocess
        text_vectorizer (tf.keras.layers.TextVectorization ): text vectorizer
        label_encoder (tf.keras.layers.StringLookup): label encoder

    Returns:
        tf.data.Dataset: transformed dataset
    """

    ### START CODE HERE ###
    # Convert the Dataset sentences to sequences, and encode the text labels
    dataset = dataset.map(lambda text, label: (text_vectorizer(text), label_encoder(label)))
    dataset = dataset.batch(32)  # Set a batch size of 32
    ### END CODE HERE ###

    return dataset


# %%
# Preprocess your dataset
train_proc_dataset = preprocess_dataset(train_dataset, vectorizer, label_encoder)
validation_proc_dataset = preprocess_dataset(validation_dataset, vectorizer, label_encoder)

print(f"Number of batches in the train dataset: {train_proc_dataset.cardinality()}")
print(f"Number of batches in the validation dataset: {validation_proc_dataset.cardinality()}")

train_batch = next(train_proc_dataset.as_numpy_iterator())
validation_batch = next(validation_proc_dataset.as_numpy_iterator())

print(f"Shape of the train batch: {train_batch[0].shape}")
print(f"Shape of the validation batch: {validation_batch[0].shape}")

# Test your code!
unittests.test_preprocess_dataset(preprocess_dataset, vectorizer, label_encoder)

## Selecting the model for text classification
### Exercise 5: create_model
# GRADED FUNCTION: create_model
def create_model():
    """
    Creates a text classifier model
    Returns:
      tf.keras Model: the text classifier model
    """

    ### START CODE HERE ###

    # Define your model
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(MAX_LENGTH,)),
        tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')  # <-- fixed to 5 classes
    ])

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    ### END CODE HERE ###

    return model

# Get the untrained model
model = create_model()

# Check the parameter count against a reference solution
unittests.parameter_count(model)
# %%
example_batch = train_proc_dataset.take(1)

try:
    model.evaluate(example_batch, verbose=False)
except:
    print(
        "Your model is not compatible with the dataset you defined earlier. Check that the loss function and last layer are compatible with one another.")
else:
    predictions = model.predict(example_batch, verbose=False)
    print(f"predictions have shape: {predictions.shape}")

# Test your code!
unittests.test_create_model(create_model)
# %%
history = model.fit(train_proc_dataset, epochs=30, validation_data=validation_proc_dataset)

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history[f'val_{metric}'])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, f'val_{metric}'])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")


with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

embedding = model.layers[0]

with open('./metadata.tsv', "w") as f:
    for word in vectorizer.get_vocabulary():
        f.write("{}\n".format(word))
weights = tf.Variable(embedding.get_weights()[0][1:])

with open('./weights.tsv', 'w') as f:
    for w in weights:
        f.write('\t'.join([str(x) for x in w.numpy()]) + "\n")