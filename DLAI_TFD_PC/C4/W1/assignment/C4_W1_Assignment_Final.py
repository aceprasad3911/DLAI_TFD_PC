import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import unittests

def trend(time, slope=0):
    """A trend over time"""
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern"""
    return np.where(season_time < 0.1,
                    np.cos(season_time * 7 * np.pi),
                    1 / np.exp(5 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level=1, seed=None):
    """Adds noise to the series"""
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

def plot_series(time, series, format="-", title="", label=None, start=0, end=None):
    """Plot the series"""
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)
    if label:
        plt.legend()
    plt.grid(True)

## Generate time series data

# The time dimension or the x-coordinate of the time series
TIME = np.arange(4 * 365 + 1, dtype="float32")

# Initial series is just a straight line with a y-intercept
y_intercept = 10
slope = 0.01
SERIES = trend(TIME, slope) + y_intercept

# Adding seasonality
amplitude = 40
SERIES += seasonality(TIME, period=365, amplitude=amplitude)

# Adding some noise
noise_level = 2
SERIES += noise(TIME, noise_level, seed=42)

# Plot the series
plt.figure(figsize=(10, 6))
plot_series(TIME, SERIES)
plt.show()

# Define time step to split the series
SPLIT_TIME = 1100

# Define the window size for forecasting later on
WINDOW_SIZE = 50


### Exercise 1: train_val_split
# GRADED FUNCTION: train_val_split
def train_val_split(time, series, split_time=1100):
    """Split time series into train and validation sets

    Args:
        time (np.ndarray): array with timestamps
        series (np.ndarray): array with values of the time series

    Returns:
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray): tuple containing timestamp and
                                                          series values for train and validation
    """
    ### START CODE HERE ###

    # Get train split
    time_train = time[:split_time]
    series_train = series[:split_time]

    # Get validation split
    time_valid = time[split_time:]
    series_valid = series[split_time:]
    ### END CODE HERE ###

    return time_train, series_train, time_valid, series_valid

# Get your train and validation splits
time_train, series_train, time_valid, series_valid = train_val_split(TIME, SERIES)

plt.figure(figsize=(10, 6))
plot_series(time_train, series_train, title="Training")

plt.figure(figsize=(10, 6))
plot_series(time_valid, series_valid, title="Validation")

# Test your code!
unittests.test_train_val_split(train_val_split)

### Exercise 2: compute_metrics
# GRADED FUNCTION: compute_metrics


def compute_metrics(true_series, forecast):
    """compute mean squared error and mean absolute error for predictions

    Args:
        true_series (np.ndarray): original (true) series
        forecast (np.ndarray): forecast series

    Returns:
        (np.float64, np.float64): MSE and MAE
    """
    ### START CODE HERE ###
    mse = tf.keras.metrics.mse(true_series, forecast).numpy()
    mae = tf.keras.metrics.mae(true_series, forecast).numpy()
    ### END CODE HERE ###

    return mse, mae

# Define some dummy series for testing
zeros = np.zeros(5)
ones = np.ones(5)

mse, mae = compute_metrics(zeros, ones)
print(f"mse: {mse}, mae: {mae} for series of zeros and prediction of ones\n")

mse, mae = compute_metrics(ones, ones)
print(f"mse: {mse}, mae: {mae} for series of ones and prediction of ones")

# Test your code!
unittests.test_compute_metrics(compute_metrics)

### Exercise 3: naive_forecast
# GRADED VARIABLE

### START CODE HERE ###
naive_forecast = SERIES[SPLIT_TIME - 1:-1]  # get naive forecast
### END CODE HERE ###

# Look into naive_forecast
print(f"validation series has shape: {series_valid.shape}\n")
print(f"naive forecast has shape: {naive_forecast.shape}\n")
print(f"comparable with validation series: {series_valid.shape == naive_forecast.shape}")

# Plot the validation data and the naive forecast
plt.figure(figsize=(10, 6))
plot_series(time_valid, series_valid, label="validation set")
plot_series(time_valid, naive_forecast, label="naive forecast")

plt.figure(figsize=(10, 6))
plot_series(time_valid, series_valid, start=330, end=361, label="validation set")
plot_series(time_valid, naive_forecast, start=330, end=361, label="naive forecast")

mse, mae = compute_metrics(series_valid, naive_forecast)

print(f"mse: {mse:.2f}, mae: {mae:.2f} for naive forecast")

# Test your code!
unittests.test_naive_forecast(naive_forecast)

### Exercise 4: moving_average_forecast
# GRADED FUNCTION: moving_average_forecast

def moving_average_forecast(series, window_size):
    """Forecasts the mean of the last few values.
        If window_size=1, then this is equivalent to naive forecast

    Args:
        series (np.ndarray): time series
        window_size (int): window size for the moving average forecast

    Returns:
        np.ndarray: time series forcast
    """

    forecast = []

    ### START CODE HERE ###
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    np_forecast = np.array(forecast)

    ### END CODE HERE ###

    return np_forecast

print(f"Whole SERIES has {len(SERIES)} elements so the moving average forecast should have {len(SERIES) - 50} elements")
# %%
# Try out your function
moving_avg = moving_average_forecast(SERIES, window_size=WINDOW_SIZE)
print(f"moving average forecast with whole SERIES has shape: {moving_avg.shape}\n")

# Slice it so it matches the validation period
moving_avg = moving_avg[1100 - WINDOW_SIZE:]
print(f"moving average forecast after slicing has shape: {moving_avg.shape}\n")
print(f"comparable with validation series: {series_valid.shape == moving_avg.shape}")

plt.figure(figsize=(10, 6))
plot_series(time_valid, series_valid)
plot_series(time_valid, moving_avg)

# Compute evaluation metrics
mse, mae = compute_metrics(series_valid, moving_avg)

print(f"mse: {mse:.2f}, mae: {mae:.2f} for moving average forecast")

### Exercise 5: diff_series
# GRADED VARIABLES

### START CODE HERE ###
# Differentiate the series. Use a differentiation step according to the series seasonality
# Differencing with lag equal to seasonality period (365)
diff_series = SERIES[365:] - SERIES[:-365]
diff_time = TIME[365:]
### END CODE HERE ###

print(f"Whole SERIES has {len(SERIES)} elements so the differencing should have {len(SERIES) - 365} elements\n")
print(f"diff series has shape: {diff_series.shape}\n")
print(f"x-coordinate of diff series has shape: {diff_time.shape}\n")

plt.figure(figsize=(10, 6))
plot_series(diff_time, diff_series)

unittests.test_diff_series(diff_series)

### Exercise 6: diff_moving_average
# GRADED VARIABLE

### START CODE HERE ###

# Apply moving average on differenced series
diff_moving_avg = moving_average_forecast(diff_series, WINDOW_SIZE)

# Align with validation period
diff_moving_avg = diff_moving_avg[SPLIT_TIME - 365 - WINDOW_SIZE:]

### END CODE HERE ###

print(f"moving average forecast with diff series after slicing has shape: {diff_moving_avg.shape}\n")
print(f"comparable with validation series: {series_valid.shape == diff_moving_avg.shape}")

plt.figure(figsize=(10, 6))
plot_series(time_valid, diff_series[1100 - 365:])
plot_series(time_valid, diff_moving_avg)

# Test your code!
unittests.test_diff_moving_avg(diff_moving_avg)

### Exercise 7: diff_moving_avg_plus_past
# GRADED VARIABLES
### START CODE HERE ###

# Slice the whole SERIES to get the past values.
# You want to get the value from the previous period for each forecasted value
# Past values (the series shifted back one seasonal period)
# Take the last 361 values to match validation length
past_series = SERIES[SPLIT_TIME - 365:][:- (len(SERIES) - (SPLIT_TIME + len(series_valid)))]
# or simply:
past_series = SERIES[SPLIT_TIME - 365: SPLIT_TIME - 365 + len(series_valid)]

# Add the past to the moving average of diff series
diff_moving_avg_plus_past = past_series + diff_moving_avg

### END CODE HERE ###

print(f"past series has shape: {past_series.shape}\n")
print(f"moving average forecast with diff series plus past has shape: {diff_moving_avg_plus_past.shape}\n")
print(f"comparable with validation series: {series_valid.shape == diff_moving_avg_plus_past.shape}")

plt.figure(figsize=(10, 6))
plot_series(time_valid, series_valid)
plot_series(time_valid, diff_moving_avg_plus_past)

# Compute evaluation metrics
mse, mae = compute_metrics(series_valid, diff_moving_avg_plus_past)

print(f"mse: {mse:.2f}, mae: {mae:.2f} for moving average plus past forecast")

# Test your code!
unittests.test_diff_moving_avg_plus_past(diff_moving_avg_plus_past)

### Exercise 8: smooth_past_series
# GRADED VARIABLE

### START CODE HERE ###

# Slice SERIES so the moving average output has exactly len(series_valid) elements
start = SPLIT_TIME - 365 - 5
end = len(SERIES) - (365 - 6)

# Compute the smoothed past series
smooth_past_series = moving_average_forecast(SERIES[start:end], window_size=11)
### END CODE HERE ###

print(f"smooth past series has shape: {smooth_past_series.shape}\n")

# Add the smoothed out past values to the moving avg of diff series
diff_moving_avg_plus_smooth_past = smooth_past_series + diff_moving_avg

print(f"moving average forecast with diff series plus past has shape: {diff_moving_avg_plus_smooth_past.shape}\n")
print(f"comparable with validation series: {series_valid.shape == diff_moving_avg_plus_smooth_past.shape}")

plt.figure(figsize=(10, 6))
plot_series(time_valid, series_valid)
plot_series(time_valid, diff_moving_avg_plus_smooth_past)

# Compute evaluation metrics
mse, mae = compute_metrics(series_valid, diff_moving_avg_plus_smooth_past)

print(f"mse: {mse:.2f}, mae: {mae:.2f} for moving average plus smooth past forecast")

# Test your code!
unittests.test_smooth_past_series(smooth_past_series)
