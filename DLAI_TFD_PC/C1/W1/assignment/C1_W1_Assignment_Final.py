import tensorflow as tf
import numpy as np
import unittests


# GRADED FUNCTION: create_training_data
def create_training_data():
    """Creates the data that will be used for training the model.

    Returns:
        (numpy.ndarray, numpy.ndarray): Arrays that contain info about the number of bedrooms and price in hundreds of thousands for 6 houses.
    """

    ### START CODE HERE ###

    # Define feature and target tensors with the values for houses with 1 up to 6 bedrooms.
    # For this exercise, please arrange the values in ascending order (i.e. 1, 2, 3, and so on).
    # Hint: Remember to explictly set the dtype as float when defining the numpy arrays
    n_bedrooms = None
    price_in_hundreds_of_thousands = None

    ### END CODE HERE ###

    return n_bedrooms, price_in_hundreds_of_thousands

features, targets = create_training_data()

print(f"Features have shape: {features.shape}")
print(f"Targets have shape: {targets.shape}")

# Expected Output:
# Features have shape: (6,)
# Targets have shape: (6,)

# Test your code!
unittests.test_create_training_data(create_training_data)


# GRADED FUNCTION: define_and_compile_model
def define_and_compile_model():
    """Returns the compiled (but untrained) model.

    Returns:
        tf.keras.Model: The model that will be trained to predict house prices.
    """

    ### START CODE HERE ###

    # Define your model
    model = tf.keras.Sequential([
        # Define the Input with the appropriate shape
        None,
        # Define the Dense layer
        None
    ])

    # Compile your model
    model.compile(optimizer=None, loss=None)

    ### END CODE HERE ###

    return model

untrained_model = define_and_compile_model()
untrained_model.summary()

# Test your code!
unittests.test_define_and_compile_model(define_and_compile_model)


# GRADED FUNCTION: train_model

def train_model():
    """Returns the trained model.

    Returns:
        tf.keras.Model: The trained model that will predict house prices.
    """

    ### START CODE HERE ###

    # Define feature and target tensors with the values for houses with 1 up to 6 bedrooms
    # Hint: Remember you already coded a function that does this!
    n_bedrooms, price_in_hundreds_of_thousands = None

    # Define a compiled (but untrained) model
    # Hint: Remember you already coded a function that does this!
    model = None

    # Train your model for 500 epochs by feeding the training data
    model.fit(None, None, epochs=None)

    ### END CODE HERE ###

    return model

# Get your trained model
trained_model = train_model()

new_n_bedrooms = np.array([7.0])
predicted_price = trained_model.predict(new_n_bedrooms, verbose=False).item()
print(f"Your model predicted a price of {predicted_price:.2f} hundreds of thousands of dollars for a {int(new_n_bedrooms.item())} bedrooms house")

# Test your code!
unittests.test_trained_model(trained_model)

