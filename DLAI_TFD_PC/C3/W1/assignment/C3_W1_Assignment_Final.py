import csv
import pandas as pd
import numpy as np
import tensorflow as tf

import unittests

with open("./data/bbc-text.csv", 'r') as csvfile:
    print(f"First line (header) looks like this:\n\n{csvfile.readline()}")
    print(f"Each data point looks like this:\n\n{csvfile.readline()}")


# GRADED FUNCTION: parse_data_from_file

def parse_data_from_file(filename):
    """
    Extracts sentences and labels from a CSV file

    Args:
        filename (str): path to the CSV file

    Returns:
        (list[str], list[str]): tuple containing lists of sentences and labels
    """
    sentences = []
    labels = []

    ### START CODE HERE ###
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            labels.append(row[0])  # Assuming the label is in the first column
            sentences.append(row[1])  # Assuming the sentence is in the second column
    ### END CODE HERE ###

    return sentences, labels

# Get sentences and labels as python lists
sentences, labels = parse_data_from_file("./data/bbc-text.csv")

print(f"There are {len(sentences)} sentences in the dataset.\n")
print(f"First sentence has {len(sentences[0].split())} words.\n")
print(f"There are {len(labels)} labels in the dataset.\n")
print(f"The first 5 labels are {labels[:5]}\n\n")

# Test your code!
unittests.test_parse_data_from_file(parse_data_from_file)

# List of stopwords
STOPWORDS = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]


# GRADED FUNCTION: standardize_func

def standardize_func(sentence):
    """Standardizes sentences by converting to lower-case and removing stopwords.

    Args:
        sentence (str): Original sentence.

    Returns:
        str: Standardized sentence in lower-case and without stopwords.
    """

    ### START CODE HERE ###
    # Convert the sentence to lower-case
    # Convert the sentence to lower-case
    sentence = sentence.lower()

    # Split the sentence into words
    words = sentence.split()

    # Remove stopwords while retaining punctuation
    filtered_words = [word for word in words if word not in STOPWORDS or word[-1] in ['!', '?', '=', ')', '(', '.']]

    # Join the filtered words back into a sentence
    sentence = ' '.join(filtered_words)
    ### END CODE HERE ###

    return sentence



test_sentence = "Hello! We're just about to see this function in action =)"
standardized_sentence = standardize_func(test_sentence)
print(f"Original sentence is:\n{test_sentence}\n\nAfter standardizing:\n{standardized_sentence}")

standard_sentences = [standardize_func(sentence) for sentence in sentences]

print("\n\n--- Apply the standardization to the dataset ---\n")
print(f"There are {len(standard_sentences)} sentences in the dataset.\n")
print(f"First sentence has {len(sentences[0].split())} words originally.\n")
print(f"First sentence has {len(standard_sentences[0].split())} words (after removing stopwords).\n")

# Test your code!
unittests.test_standardize_func(standardize_func)


# GRADED FUNCTION: fit_vectorizer

def fit_vectorizer(sentences):
    """
    Instantiates the TextVectorization layer and adapts it to the sentences.

    Args:
        sentences (list[str]): lower-cased sentences without stopwords

    Returns:
        tf.keras.layers.TextVectorization: an instance of the TextVectorization layer adapted to the texts.
    """

    ### START CODE HERE ###

    # Instantiate the TextVectorization layer
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=33088,
        output_mode='int',
        output_sequence_length=100,
        standardize=None  # since you've already standardized data
    )

    # Adapt the vectorizer to the sentences
    vectorizer.adapt(sentences)

    ### END CODE HERE ###

    return vectorizer

# Create the vectorizer adapted to the standardized sentences
vectorizer = fit_vectorizer(standard_sentences)

# Get the vocabulary
vocabulary = vectorizer.get_vocabulary()

print(f"Vocabulary contains {len(vocabulary)} words\n")
print("[UNK] token included in vocabulary" if "[UNK]" in vocabulary else "[UNK] token NOT included in vocabulary")

# Test your code!
unittests.test_fit_vectorizer(fit_vectorizer)

# Vectorize and pad sentences
padded_sequences = vectorizer(standard_sentences)

# Show the output
print(f"First padded sequence looks like this: \n\n{padded_sequences[0]}\n")
print(f"Tensor of all sequences has shape: {padded_sequences.shape}\n")
print(f"This means there are {padded_sequences.shape[0]} sequences in total and each one has a size of {padded_sequences.shape[1]}")


# GRADED FUNCTION: fit_label_encoder

def fit_label_encoder(labels):
    """
    Tokenizes the labels

    Args:
        labels (list[str]): labels to tokenize

    Returns:
        tf.keras.layers.StringLookup: adapted encoder for labels
    """
    ### START CODE HERE ###

    # Get just the unique labels
    unique_labels = list(set(labels))

    # Create the encoder with our specific vocabulary
    label_encoder = tf.keras.layers.StringLookup(
        vocabulary=unique_labels,
        mask_token=None,
        oov_token=None,
        num_oov_indices=0  # Critical to ensure no additional tokens
    )

    ### END CODE HERE ###

    return label_encoder

# Create the encoder adapted to the labels
label_encoder = fit_label_encoder(labels)

# Get the vocabulary
vocabulary = label_encoder.get_vocabulary()

# Encode labels
label_sequences = label_encoder(labels)

print(f"Vocabulary of labels looks like this: {vocabulary}\n")
print(f"First ten labels: {labels[:10]}\n")
print(f"First ten label sequences: {label_sequences[:10]}\n")

# Test your code!
unittests.test_fit_label_encoder(fit_label_encoder)



