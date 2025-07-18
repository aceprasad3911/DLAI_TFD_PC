import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

fmnist = tf.keras.datasets.fashion_mnist
# fashion_mnist is available as a dataset with api call in tensorflow

(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()
# load 4 lists via load_data method to object fashion_mnist
# good to have training content (60,000) + test content that the model has not seen (10,000)
# labels are kept as numbers to avoid language bias

index = 0

# Set number of characters per row when printing
np.set_printoptions(linewidth=320)

# Print the label and image
print(f'fLABEL: {training_labels[index]}')
print(f'\nIMAGE PIXEL ARRAY:\n\n{training_images[index]}\n\n')

# Visualize the image using the default colormap (viridis)
plt.imshow(training_images[index])
plt.colorbar()
plt.show()

# Normalize the pixel values of the train and test images
training_images  = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28,28)), # images are 28x28 pixels
    # 3 layers to this model as seen below
    tf.keras.layers.Flatten(), # Flatten takes square and turns it into a linear array
    tf.keras.layers.Dense (128, activation=tf.nn.relu), # Hidden/Middle layer has 128 neurons, turns 784 values representing image into label value
    # Relu means if x > 0 -> return x, else -> return 0
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # 10 neurons in layer because 10 classes of clothing in fashion_mnist
    # Softmax takes list of values abd scales (e.g units: 10 -> 1/10 -> 0.1 probability)
])

# Expanded explanation of SoftMax function
# Declare sample inputs and convert to a tensor
inputs = np.array([[1.0, 3.0, 4.0, 2.0, 1.6, 6.2]])
inputs = tf.convert_to_tensor(inputs)
print(f'input to softmax function: {inputs.numpy()}')

# Feed the inputs to a softmax activation function
outputs = tf.keras.activations.softmax(inputs)
print(f'output of softmax function: {outputs.numpy()}')

# Get the sum of all values after the softmax
sum = tf.reduce_sum(outputs)
print(f'sum of outputs: {sum}')

# Get the index with highest value
prediction = np.argmax(outputs)
print(f'index with highest probability: {prediction}')

# loss function measures accuracy of guess; data then passed onto optimizer for the next guess, with each guess becoming more accurate until convergence
model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

# Evaluate the model on unseen data
model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
# Output 10 probabilities predicting the value being classified is the indexed value
# For context, there are 10 labels that all test images fall into
print(classifications[0])

