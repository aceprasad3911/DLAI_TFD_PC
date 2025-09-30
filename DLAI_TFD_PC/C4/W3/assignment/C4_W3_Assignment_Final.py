import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

import unittests

# Generating the data
def plot_series(time, series, format="-", start=0, end=None):
    """Plot the series"""
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(False)


def trend(time, slope=0):
    """A trend over time"""
    return slope * time


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.1,
                    np.cos(season_time * 6 * np.pi),
                    2 / np.exp(9 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    """Adds noise to the series"""
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

def generate_time_series():
    """ Creates timestamps and values of the time series """

    # The time dimension or the x-coordinate of the time series
    time = np.arange(4 * 365 + 1, dtype="float32")

    # Initial series is just a straight line with a y-intercept
    y_intercept = 10
    slope = 0.005
    series = trend(time, slope) + y_intercept

    # Adding seasonality
    amplitude = 50
    series += seasonality(time, period=365, amplitude=amplitude)

    # Adding some noise
    noise_level = 3
    series += noise(time, noise_level, seed=51)

    return time, series

## Defining some useful global variables

SPLIT_TIME = 1100
WINDOW_SIZE = 20
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

# Create the time series
TIME, SERIES = generate_time_series()

# Plot the generated series
plt.figure(figsize=(10, 6))
plot_series(TIME, SERIES)
plt.show()

## Processing the data

def train_val_split(time, series):
    """ Splits time series into train and validation sets"""
    time_train = time[:SPLIT_TIME]
    series_train = series[:SPLIT_TIME]
    time_valid = time[SPLIT_TIME:]
    series_valid = series[SPLIT_TIME:]

    return time_train, series_train, time_valid, series_valid

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

# Split the dataset
time_train, series_train, time_valid, series_valid = train_val_split(TIME, SERIES)
# Apply the transformation to the training set
dataset = windowed_dataset(series_train, WINDOW_SIZE)

## Defining the model architecture

### Exercise 1: create_uncompiled_model
# GRADED FUNCTION: create_uncompiled_model

def create_uncompiled_model():
    """Define uncompiled model

    Returns:
        tf.keras.Model: uncompiled model
    """
    ### START CODE HERE ###

    model = tf.keras.models.Sequential([
        tf.keras.Input((WINDOW_SIZE, 1)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 100.0)
    ])

    ### END CODE HERE ###

    return model

# Define your uncompiled model
uncompiled_model = create_uncompiled_model()

# Check the parameter count against a reference solution
unittests.parameter_count(uncompiled_model)

example_batch = dataset.take(1)

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
def adjust_learning_rate(model):
    """Fit model using different learning rates

    Args:
        model (tf.keras.Model): uncompiled model

    Returns:
        tf.keras.callbacks.History: callback history
    """

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-6 * 10 ** (epoch / 20))

    ### START CODE HERE ###

    # Compile the model passing in the appropriate loss
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9)

    # Compile the model with Huber loss (robust for time series forecasting)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])

    ### END CODE HERE ###

    history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])

    return history

# Run the training with dynamic LR
lr_history = adjust_learning_rate(uncompiled_model)


# Plot the loss for every LR
plt.semilogx(lr_history.history["learning_rate"], lr_history.history["loss"])
plt.axis([1e-6, 1, 0, 30])


### Exercise 2: create_model
# GRADED FUNCTION: create_model
def create_model():
    """Creates and compiles the model

    Returns:
        tf.keras.Model: compiled model
    """
    model = create_uncompiled_model()

    ### START CODE HERE ###

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=["mae"])

    ### END CODE HERE ###

    return model

# Create an instance of the model
model = create_model()

# Test your code!
unittests.test_create_model(create_model)

# Train it
history = model.fit(dataset, epochs=50)

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

def generate_forecast(model, series, window_size):
    """Generates a forecast using your trained model"""
    forecast = []
    for time in range(SPLIT_TIME, len(series)):
        pred = model.predict(series[time - window_size:time][np.newaxis])
        forecast.append(pred[0][0])
    return forecast

# Save the forecast
rnn_forecast = generate_forecast(model, SERIES, WINDOW_SIZE)

# Plot your forecast
plt.figure(figsize=(10, 6))

plot_series(time_valid, series_valid)
plot_series(time_valid, rnn_forecast)

mse, mae = compute_metrics(series_valid, rnn_forecast)

print(f"mse: {mse:.2f}, mae: {mae:.2f} for forecast")

# Save your mae in a pickle file
with open('forecast_mae.pkl', 'wb') as f:
    pickle.dump(mae.numpy(), f)