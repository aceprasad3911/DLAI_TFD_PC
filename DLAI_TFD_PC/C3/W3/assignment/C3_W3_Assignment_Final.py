import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import unittests

EMBEDDING_DIM = 100
MAX_LENGTH = 32
TRAINING_SPLIT = 0.9
BATCH_SIZE = 128

data_path = "./data/training_cleaned.csv"
df = pd.read_csv(data_path, header=None)
df.head()

# Standardize labels so they have 0 for negative and 1 for positive
labels = df[0].apply(lambda x: 0 if x == 0 else 1).to_numpy()

# Since the original dataset does not provide headers you need to index the columns by their index
sentences = df[5].to_numpy()

# Create the dataset
dataset = tf.data.Dataset.from_tensor_slices((sentences, labels))

# Get the first 5 elements of the dataset
examples = list(dataset.take(5))

print(f"dataset contains {len(dataset)} examples\n")

print(f"Text of second example look like this: {examples[1][0].numpy().decode('utf-8')}\n")
print(f"Labels of first 5 examples look like this: {[x[1].numpy() for x in examples]}")

## Exercise 1: train_val_datasets
# GRADED FUNCTION: train_val_datasets


def train_val_datasets(dataset):
    """
    Splits the dataset into training and validation sets, after shuffling it.

    Args:
        dataset (tf.data.Dataset): Tensorflow dataset with elements as (sentence, label)

    Returns:
        (tf.data.Dataset, tf.data.Dataset): tuple containing the train and validation datasets
    """
    ### START CODE HERE ###

    dataset = dataset.shuffle(buffer_size=len(dataset), seed=42)

    # Compute the number of sentences that will be used for training (should be an integer)
    train_size = int(TRAINING_SPLIT * len(dataset))

    # Split the sentences and labels into train/validation splits
    train_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size)

    # Turn the dataset into a batched dataset with NUM_BATCHES per batch
    train_dataset = train_dataset.batch(BATCH_SIZE)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)


    ### END CODE HERE ###

    return train_dataset, validation_dataset

# Create the train and validation datasets
train_dataset, validation_dataset = train_val_datasets(dataset)

print(
    f"There are {len(train_dataset)} batches for a total of {NUM_BATCHES * len(train_dataset)} elements for training.\n")
print(
    f"There are {len(validation_dataset)} batches for a total of {NUM_BATCHES * len(validation_dataset)} elements for validation.\n")

# Test your code!
unittests.test_train_val_datasets(train_val_datasets)

## Exercise 2: fit_vectorizer
# GRADED FUNCTION: fit_vectorizer

def fit_vectorizer(dataset):
    """
    Adapts the TextVectorization layer on the training sentences

    Args:
        dataset (tf.data.Dataset): Tensorflow dataset with training sentences.

    Returns:
        tf.keras.layers.TextVectorization: an instance of the TextVectorization class adapted to the training sentences.
    """

    ### START CODE HERE ###

    # Instantiate the TextVectorization class, defining the necessary arguments alongside their corresponding values
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=20000,                 # maximum vocab size
        output_sequence_length=MAX_LENGTH,
        output_mode="int"
    )

    # Fit the tokenizer to the training sentences
    vectorizer.adapt(dataset)

    ### END CODE HERE ###

    return vectorizer



# Get only the texts out of the dataset
text_only_dataset = train_dataset.map(lambda text, label: text)

# Adapt the vectorizer to the training sentences
vectorizer = fit_vectorizer(text_only_dataset)

# Check size of vocabulary
vocab_size = vectorizer.vocabulary_size()

print(f"Vocabulary contains {vocab_size} words\n")

# Test your code!
unittests.test_fit_vectorizer(fit_vectorizer)

# Apply vectorization to train and val datasets
train_dataset_vectorized = train_dataset.map(lambda x, y: (vectorizer(x), y))
validation_dataset_vectorized = validation_dataset.map(lambda x, y: (vectorizer(x), y))

# Define path to file containing the embeddings
glove_file = './data/glove.6B.100d.txt'

# Initialize an empty embeddings index dictionary
glove_embeddings = {}

# Read file and fill glove_embeddings with its contents
with open(glove_file) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        glove_embeddings[word] = coefs

test_word = 'dog'

test_vector = glove_embeddings[test_word]

print(f"Vector representation of word {test_word} looks like this:\n\n{test_vector}")
print(f"Each word vector has shape: {test_vector.shape}")

# Create a word index dictionary
word_index = {x: i for i, x in enumerate(vectorizer.get_vocabulary())}

print(f"The word dog is encoded as: {word_index['dog']}")

# Initialize an empty numpy array with the appropriate size
embeddings_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

# Iterate all of the words in the vocabulary and if the vector representation for
# each word exists within GloVe's representations, save it in the embeddings_matrix array
for word, i in word_index.items():
    embedding_vector = glove_embeddings.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector

test_word = 'dog'

test_word_id = word_index[test_word]

test_vector_dog = glove_embeddings[test_word]

test_embedding_dog = embeddings_matrix[test_word_id]

both_equal = np.allclose(test_vector_dog, test_embedding_dog)

print(
    f"word: {test_word}, index: {test_word_id}\n\nEmbedding is equal to column {test_word_id} in the embeddings_matrix: {both_equal}")

## Exercise 3: create_model
# GRADED FUNCTION: create_model

def create_model(vocab_size, pretrained_embeddings):
    """
    Creates a binary sentiment classifier model

    Args:
        vocab_size (int): Number of words in the vocabulary.
        pretrained_embeddings (np.ndarray): Array containing pre-trained embeddings.

    Returns:
        (tf.keras Model): the sentiment classifier model
    """
    ### START CODE HERE ###

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(MAX_LENGTH,)),  # Input expects padded sequences
        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=EMBEDDING_DIM,
            weights=[pretrained_embeddings],
            input_length=MAX_LENGTH,
            trainable=False  # freeze embeddings to avoid overfitting
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")  # binary classification
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    ### END CODE HERE ###

    return model

# Create your untrained model
model = create_model(vocab_size, embeddings_matrix)

# Check parameter count against a reference solution
unittests.parameter_count(model)

# Take an example batch of data
example_batch = train_dataset_vectorized.take(1)

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
# Train the model and save the training history
history = model.fit(
    train_dataset_vectorized,
    epochs=20,
    validation_data=validation_dataset_vectorized
)

# Get training and validation accuracies
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Training and validation performance')

for i, (data, label) in enumerate(zip([(acc, val_acc), (loss, val_loss)], ["Accuracy", "Loss"])):
    ax[i].plot(epochs, data[0], 'r', label="Training " + label)
    ax[i].plot(epochs, data[1], 'b', label="Validation " + label)
    ax[i].legend()
    ax[i].set_xlabel('epochs')

# Test your code!
unittests.test_history(history)

with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)