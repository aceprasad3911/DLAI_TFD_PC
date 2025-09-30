import csv
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import unittests

DATA_PATH = './data/daily-min-temperatures.csv'

with open(DATA_PATH, 'r') as csvfile:
    print(f"Header looks like this:\n\n{csvfile.readline()}")
    print(f"First data point looks like this:\n\n{csvfile.readline()}")
    print(f"Second data point looks like this:\n\n{csvfile.readline()}")

def plot_series(time, series, format="-", start=0, end=None):
    """Plot the series"""
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)

### Exercise 1: parse_data_from_file
# GRADED FUNCTION: parse_data

def parse_data_from_file(filename):
    """Parse data from csv file

    Args:
        filename (str): complete path to file (path + filename)

    Returns:
        (np.ndarray, np.ndarray): arrays of timestamps and values of the time series
    """
    ### START CODE HERE
    # Load the temperatures from column 1 (skip header row)
    temperatures = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=1)
    # Generate integer time steps from 0 up to len(temperatures)-1
    times = np.arange(len(temperatures))
    ### END CODE HERE

    return times, temperatures

TIME, SERIES = parse_data_from_file(DATA_PATH)

# Plot the series!
plt.figure(figsize=(10, 6))
plot_series(TIME, SERIES)

# Test your code!
unittests.test_parse_data_from_file(parse_data_from_file)

# Save all global variables
SPLIT_TIME = 2500
WINDOW_SIZE = 64
BATCH_SIZE = 256
SHUFFLE_BUFFER_SIZE = 1000

## Processing the data
def train_val_split(time, series):
    """ Splits time series into train and validations sets"""
    time_train = time[:SPLIT_TIME]
    series_train = series[:SPLIT_TIME]
    time_valid = time[SPLIT_TIME:]
    series_valid = series[SPLIT_TIME:]

    return time_train, series_train, time_valid, series_valid

# Split the dataset
time_train, series_train, time_valid, series_valid = train_val_split(TIME, SERIES)

def windowed_dataset(series, window_size):
    """Creates windowed dataset"""
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(BATCH_SIZE).prefetch(1)
    return dataset

# Apply the transformation to the training set
train_dataset = windowed_dataset(series_train, window_size=WINDOW_SIZE)

### Exercise 2: create_uncompiled_model
# GRADED FUNCTION: create_uncompiled_model
def create_uncompiled_model():
    """Define uncompiled model

    Returns:
        tf.keras.Model: uncompiled model
    """
    ### START CODE HERE ###

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size, 1)),
        # A small 1D-conv block for sequence representation
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='causal', activation='relu'),
        tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='causal', activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)  # regression output
    ])

    ### END CODE HERE ###
    return model

# Get your uncompiled model
uncompiled_model = create_uncompiled_model()

# Check the parameter count against a reference solution
unittests.parameter_count(uncompiled_model)

example_batch = train_dataset.take(1)

try:
    predictions = uncompiled_model.predict(example_batch, verbose=False)
except:
    print(
        "Your model is not compatible with the dataset you defined earlier. Check that the loss function and last layer are compatible with one another.")
else:
    print("Your current architecture is compatible with the windowed dataset! :)")
    print(f"predictions have shape: {predictions.shape}")

# Test your code!
unittests.test_create_uncompiled_model(create_uncompiled_model)

uncompiled_model.summary()

## Adjusting the learning rate - (Optional Exercise)
def adjust_learning_rate(dataset):
    """Fit model using different learning rates

    Args:
        dataset (tf.data.Dataset): train dataset

    Returns:
        tf.keras.callbacks.History: callback history
    """

    model = create_uncompiled_model()

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10 ** (epoch / 20))

    ### START CODE HERE ###

    # Select your optimizer
    optimizer = None

    # Compile the model passing in the appropriate loss
    model.compile(loss=None,
                  optimizer=optimizer,
                  metrics=["mae"])

    ### END CODE HERE ###

    history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])

    return history

# Run the training with dynamic LR
lr_history = adjust_learning_rate(train_dataset)

plt.semilogx(lr_history.history["learning_rate"], lr_history.history["loss"])

### Exercise 3: create_model
# GRADED FUNCTION: create_model

def create_model():
    """Creates and compiles the model

    Returns:
        tf.keras.Model: compiled model
    """

    model = create_uncompiled_model()

    ### START CODE HERE ###

    model.compile(loss=None,
                  optimizer=None,
                  metrics=["mae"])

    ### END CODE HERE ###

    return model

# Save an instance of the model
model = create_model()

# Test your code!
unittests.test_create_model(create_model)

# Train it
history = model.fit(train_dataset, epochs=50)

# Plot the training loss for each epoch

loss = history.history['loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')
plt.title('Training loss')
plt.legend(loc=0)
plt.show()

## Evaluating the forecast
def compute_metrics(true_series, forecast):
    """Computes MSE and MAE metrics for the forecast"""
    mse = tf.keras.losses.MSE(true_series, forecast)
    mae = tf.keras.losses.MAE(true_series, forecast)
    return mse, mae


## Faster model forecasts
def model_forecast(model, series, window_size):
    """Generates a forecast using your trained model"""
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast

# Compute the forecast for the validation dataset. Remember you need the last WINDOW SIZE values to make the first prediction
rnn_forecast = model_forecast(model, SERIES[SPLIT_TIME - WINDOW_SIZE:-1], WINDOW_SIZE).squeeze()

# Plot the forecast
plt.figure(figsize=(10, 6))
plot_series(time_valid, series_valid)
plot_series(time_valid, rnn_forecast)

mse, mae = compute_metrics(series_valid, rnn_forecast)

print(f"mse: {mse:.2f}, mae: {mae:.2f} for forecast")

# Save metrics into a dictionary
metrics_dict = {
    "mse": float(mse),
    "mae": float(mae)
}

# Save your metrics in a binary file
with open('metrics.pkl', 'wb') as f:
    pickle.dump(metrics_dict, f)
