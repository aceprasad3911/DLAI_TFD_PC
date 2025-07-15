import tensorflow as tf
import numpy as np

model = tf.keras.Sequential({ # Successive layers are defined in sequence, hence Sequential
    tf.keras.Input(shape=(1,)), # Industry practise to map out the shape of input x
    tf.keras.layers.Dense(units=1) # Dense used to define layer of connected neurons, only one dense, only one layer, only unit -> single neuron
})
model.compile(optimizer='sgd', loss='mean_squared_error') # loss function measures accuracy of guess; data then passed onto optimizer for the next guess, with each guess becoming more accurate until convergence
# sgd = stochastic gradient descent

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float) # represent known data of x and y values in arrays
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500) # training takes place in fit command, asking model to figure out how to fit x values to y values, using loss and optimizer in compile method

model.predict(np.array([10.0])) # predicts outcome -> supposed to be 19 since relation is y = 2x - 1
                                # the outcome will be very close to 19 but not = 19
                                # because trained with very little data (only 6 (x, y) points
                                # no guarantee of relationship assumption, just very high probability of accurate guesswork
                                # neural networks only work in probability. not certainty

# Model Prediction output:
print(f"model predicted: {model.predict(np.array([10.0]), verbose=0).item():.5f}")
# model predicted: 18.98361

# Navigate to each course's directory page to install workbook necessary systems

# E.g.)
# cd C1
# pip install -r requirements.txt