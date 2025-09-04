import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import unittests


## Generating the data

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

# Save all global variables
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

## Splitting the data

def train_val_split(time, series):
    time_train = time[:SPLIT_TIME]
    series_train = series[:SPLIT_TIME]
    time_valid = time[SPLIT_TIME:]
    series_valid = series[SPLIT_TIME:]

    return time_train, series_train, time_valid, series_valid


# Split the dataset
time_train, series_train, time_valid, series_valid = train_val_split(TIME, SERIES)

### Exercise 1: windowed_dataset
# GRADED FUNCTION: windowed_dataset


def windowed_dataset(series, window_size, shuffle=True):
    """Create a windowed dataset

    Args:
        series (np.ndarray): time series
        window_size (int): length of window to use for prediction
        shuffle (bool): (For testing purposes) Indicates whether to shuffle data before batching or not. Defaults to True

    Returns:
        td.data.Dataset: windowed dataset
    """

    ### START CODE HERE ###
    # Create dataset from the series.
    # HINT: use an appropriate method from the tf.data.Dataset object
    dataset = None

    # Slice the dataset into the appropriate windows
    dataset = None

    # Flatten the dataset
    dataset = None

    # Shuffle it
    if shuffle:  # For testing purposes
        dataset = None

        # Split it into the features and labels.
    dataset = None

    # Batch it
    dataset = None

    ### END CODE HERE ###

    return dataset

# Try out your function with windows size of 1 and no shuffling
test_dataset = windowed_dataset(series_train, window_size=10, shuffle=False)

# Get the first batch of the test dataset
batch_of_features, batch_of_labels = next((iter(test_dataset)))

print(f"batch_of_features has type: {type(batch_of_features)}\n")
print(f"batch_of_labels has type: {type(batch_of_labels)}\n")
print(f"batch_of_features has shape: {batch_of_features.shape}\n")
print(f"batch_of_labels has shape: {batch_of_labels.shape}\n")
print(
    f"First element in batch_of_features is equal to first 10 elements in the series: {np.allclose(batch_of_features.numpy()[0].flatten(), series_train[:10])}\n")
print(
    f"batch_of_labels is equal to the first 32 values after the window_lenght of 10): {np.allclose(batch_of_labels.numpy(), series_train[10:BATCH_SIZE + 10])}")

plt.plot(np.arange(10), batch_of_features[0].numpy(), label='features')
plt.plot(np.arange(9, 11), [batch_of_features[0].numpy()[-1], batch_of_labels[0].numpy()], label='label');
plt.legend()

# Apply the processing to the whole training series
train_dataset = windowed_dataset(series_train, WINDOW_SIZE)

# Test your code!
unittests.test_windowed_dataset(windowed_dataset)

## Defining the model architecture

### Exercise 2: create_model
# GRADED FUNCTION: create_model

def create_model(window_size):
    """Create model for predictions
    Args:
        window_size (int): length of window to use for prediction

    Returns:
        tf.keras.Model: model
    """
    ### START CODE HERE ###

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=None),

    ])

    model.compile(loss=None,
                  optimizer=None)

    ### END CODE HERE ###

    return model

# Get the untrained model
model = create_model(WINDOW_SIZE)

# Check the parameter count against a reference solution
unittests.parameter_count(model)
# %%
example_batch = train_dataset.take(1)

try:
    model.evaluate(example_batch, verbose=False)
except:
    print(
        "Your model is not compatible with the dataset you defined earlier. Check that the loss function and last layer are compatible with one another.")
else:
    predictions = model.predict(example_batch, verbose=False)
    print(f"predictions have shape: {predictions.shape}")

print(f'Model input shape: {model.input_shape}')
print(f'Model output shape: {model.output_shape}')

model.summary()

# Test your code!
unittests.test_create_model(create_model, windowed_dataset)

# Train it
history = model.fit(train_dataset, epochs=100)

# Plot the training loss for each epoch

loss = history.history['loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'r', label='Training loss')
plt.title('Training loss')
plt.legend(loc=0)
plt.show()

## Evaluating the forecast

def compute_metrics(true_series, forecast):
    mse = tf.keras.losses.MSE(true_series, forecast)
    mae = tf.keras.losses.MAE(true_series, forecast)
    return mse, mae

def generate_forecast(model, series, window_size):
    forecast = []
    for time in range(SPLIT_TIME, len(series)):
        pred = model.predict(series[time - window_size:time][np.newaxis], verbose=0)
        forecast.append(pred[0][0])
    return forecast

# Save the forecast
dnn_forecast = generate_forecast(model, SERIES, WINDOW_SIZE)

# Plot it
plt.figure(figsize=(10, 4))
plot_series(time_valid, series_valid)
plot_series(time_valid, dnn_forecast)

mse, mae = compute_metrics(series_valid, dnn_forecast)

print(f"mse: {mse:.2f}, mae: {mae:.2f} for forecast")

# ONLY RUN THIS CELL IF YOUR MSE ACHIEVED THE DESIRED MSE LEVEL
# Save your model
model.save('trained_model.keras')
