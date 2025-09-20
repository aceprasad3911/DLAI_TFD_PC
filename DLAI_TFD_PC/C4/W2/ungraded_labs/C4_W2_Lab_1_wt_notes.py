import tensorflow as tf

# Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
dataset = tf.data.Dataset.range(10)

# Preview the result
for val in dataset:
   print(val.numpy())

## Windowing the data

# Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
dataset = tf.data.Dataset.range(10)

# Window the data
dataset = dataset.window(size=5, shift=1)

# Print the result
for window_dataset in dataset:
  print(window_dataset)

# Print the result
for window_dataset in dataset:
  print([item.numpy() for item in window_dataset])

# Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
dataset = tf.data.Dataset.range(10)

# Window the data but only take those with the specified size
dataset = dataset.window(size=5, shift=1, drop_remainder=True) # size of window & size of shift each time
# drop_remainder=True truncates all data by dropping all remainders -> providing windows with only 5 items)

# Print the result
for window_dataset in dataset:
  print([item.numpy() for item in window_dataset])

# Output without drop_remainder=True:
# 0 1 2 3 4 (Window of 5 items per row)
# 1 2 3 4 5
# 2 3 4 5 6
# 3 4 5 6 7
# 4 5 6 7 8
# 5 6 7 8 9 (As you reach final value, space filled by empty spaces so shift ends at 9)
# 6 7 8 9
# 7 8 9
# 8 9
# 9

# Output with drop_remainder=True:
# 0 1 2 3 4 (Window of 5 items per row)
# 1 2 3 4 5
# 2 3 4 5 6
# 3 4 5 6 7
# 4 5 6 7 8
# 5 6 7 8 9




## Flatten the Windows

# Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
dataset = tf.data.Dataset.range(10)

# Window the data but only take those with the specified size
dataset = dataset.window(5, shift=1, drop_remainder=True)

# Flatten the windows by putting its elements in a single batch
dataset = dataset.flat_map(lambda window: window.batch(5))

# Print the results
for window in dataset:
  print(window.numpy()) # Call .numpy method for each item in dataset, prints output in numpy lists (better for ML usage)

# Output with print(window.numpy()):
# [0 1 2 3 4]
# [1 2 3 4 5]
# [2 3 4 5 6]
# [3 4 5 6 7]
# [4 5 6 7 8]
# [5 6 7 8 9]

## Group into features and labels

# Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
dataset = tf.data.Dataset.range(10)

# Window the data but only take those with the specified size
dataset = dataset.window(5, shift=1, drop_remainder=True)

# Flatten the windows by putting its elements in a single batch
dataset = dataset.flat_map(lambda window: window.batch(5))

# Create tuples with features (first four elements of the window) and labels (last element)
dataset = dataset.map(lambda window: (window[:-1], window[-1])) # :-1 = everything but the last item, -1 = the last one
# For each item in list, we map them to make all the values but the last one the feature, and last one being the label

# Output with dataset.map(lambda window: (window[:-1], window[-1])):
# [0 1 2 3] [4]
# [1 2 3 4] [5]
# [2 3 4 5] [6]
# [3 4 5 6] [7]
# [4 5 6 7] [8]
# [5 6 7 8] [9]

# Print the results
for x,y in dataset:
  print("x = ", x.numpy())
  print("y = ", y.numpy())
  print()

## Shuffle the data

# Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
dataset = tf.data.Dataset.range(10)

# Window the data but only take those with the specified size
dataset = dataset.window(5, shift=1, drop_remainder=True)

# Flatten the windows by putting its elements in a single batch
dataset = dataset.flat_map(lambda window: window.batch(5))

# Create tuples with features (first four elements of the window) and labels (last element)
dataset = dataset.map(lambda window: (window[:-1], window[-1]))

# Shuffle the windows
dataset = dataset.shuffle(buffer_size=10) # Feature & Label sets have been shuffled but sets are still paired accordingly

# Output with dataset.shuffle(buffer_size=10):
# [3 4 5 6] [7]
# [4 5 6 7] [8]
# [1 2 3 4] [5]
# [2 3 4 5] [6]
# [5 6 7 8] [9]
# [0 1 2 3] [4]

# Print the results
for x,y in dataset:
  print("x = ", x.numpy())
  print("y = ", y.numpy())
  print()

## Create batches for training

# Generate a tf dataset with 10 elements (i.e. numbers 0 to 9)
dataset = tf.data.Dataset.range(10)

# Window the data but only take those with the specified size
dataset = dataset.window(5, shift=1, drop_remainder=True)

# Flatten the windows by putting its elements in a single batch
dataset = dataset.flat_map(lambda window: window.batch(5))

# Create tuples with features (first four elements of the window) and labels (last element)
dataset = dataset.map(lambda window: (window[:-1], window[-1]))

# Shuffle the windows
dataset = dataset.shuffle(buffer_size=10)

# Create batches of windows (batches of 2 features, 2 labels)
dataset = dataset.batch(2)

# Output with dataset.batch(2).prefetch(1):
# x = [4 5 6 7] [1 2 3 4]
# y = [8] [5]
# x = [3 4 5 6] [2 3 4 5]
# y = [7] [6]
# x = [5 6 7 8] [0 1 2 3]
# y = [9] [4]

# Optimize the dataset for training
dataset = dataset.cache().prefetch(1)

# Print the results
for x,y in dataset:
  print("x = ", x.numpy())
  print("y = ", y.numpy())
  print()
