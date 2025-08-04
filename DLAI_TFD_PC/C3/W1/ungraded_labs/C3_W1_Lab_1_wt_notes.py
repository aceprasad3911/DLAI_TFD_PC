import tensorflow as tf

# Sample inputs
sentences = [
    'i love my dog',
    'I, love my cat'
    ]

# Initialize the layer
vectorize_layer = tf.keras.layers.TextVectorization() # Encodes words with value, helps create vectors out of sentences

# Build the vocabulary
vectorize_layer.adapt(sentences) # Adapt method takes in data and generates vocabulary out of the words found in the sentences

# Get the vocabulary list. Ignore special tokens for now.
vocabulary = vectorize_layer.get_vocabulary(include_special_tokens=False) # Get vocab Method returns list of all the words in the vocab

# Print the token index
for index, word in enumerate(vocabulary):
  print(index, word)

# Add another input
sentences = [
    'i love my dog',
    'I, love my cat',
    'You love my dog!'
]

# Initialize the layer
vectorize_layer = tf.keras.layers.TextVectorization()

# Build the vocabulary
vectorize_layer.adapt(sentences)

# Get the vocabulary list. Ignore special tokens for now.
vocabulary = vectorize_layer.get_vocabulary(include_special_tokens=False) # True by default, returns extra '' and '[UNK]' when removed from parameter

# Print the token index
for index, word in enumerate(vocabulary):
  print(index, word)

# Get the vocabulary list.
vocabulary = vectorize_layer.get_vocabulary()

# Print the token index
for index, word in enumerate(vocabulary):
  print(index, word)