import os
import matplotlib.pyplot as plt
import tensorflow as tf

import unittests

TRAIN_DIR = './data/train/'
VALIDATION_DIR = './data/validation/'

# Directories for each class
horses_dir = os.path.join(TRAIN_DIR, 'horses')
humans_dir = os.path.join(TRAIN_DIR, 'humans')

# Load the first example of each one of the classes
sample_image_horse  = tf.keras.preprocessing.image.load_img(os.path.join(horses_dir, os.listdir(horses_dir)[0]))
sample_image_human  = tf.keras.preprocessing.image.load_img(os.path.join(humans_dir, os.listdir(humans_dir)[0]))

ax = plt.subplot(1,2,1)
ax.imshow(sample_image_horse)
ax.set_title('Sample horse image')

ax = plt.subplot(1,2,2)
ax.imshow(sample_image_human)
ax.set_title('Sample human image')
plt.show()

# Convert the image into its numpy array representation
sample_array = tf.keras.preprocessing.image.img_to_array(sample_image_horse)

print(f"Each image has shape: {sample_array.shape}")


# GRADED FUNCTION: train_val_datasets

def train_val_datasets():
    """Creates training and validation datasets

    Returns:
        (tf.data.Dataset, tf.data.Dataset): training and validation datasets
    """

    ### START CODE HERE ###

    training_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=TRAIN_DIR,
        batch_size=32,
        image_size=(150, 150),
        shuffle=True,
        seed=7
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=VALIDATION_DIR,
        batch_size=32,
        image_size=(150, 150),
        shuffle=True,
        seed=7
    )

    ### END CODE HERE ###

    return training_dataset, validation_dataset

# Test your generators
training_dataset, validation_dataset = train_val_datasets()

# Test your code!
unittests.test_train_val_datasets(train_val_datasets)

val_batches = int(validation_dataset.cardinality())
test_dataset, validation_dataset = tf.keras.utils.split_dataset(validation_dataset, val_batches//5)

print(f'Number of validation batches: {validation_dataset.cardinality()}')
print(f'Number of test batches: {test_dataset.cardinality()}')

# Define the path to the inception v3 weights
LOCAL_WEIGHTS_FILE = './model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


# GRADED FUNCTION: create_pre_trained_model

def create_pre_trained_model():
    """Creates the pretrained inception V3 model

    Returns:
        tf.keras.Model: pre-trained model
    """

    ### START CODE HERE ###

    pre_trained_model = tf.keras.applications.inception_v3.InceptionV3(
        include_top=False,
        input_shape=(15, 150, 3),
        weights='imagenet'
    )

    # Make all the layers in the pre-trained model non-trainable
    for layer in pre_trained_model.layers:
        layer.trainable = False

    ### END CODE HERE ###

    return pre_trained_model

# Create the pre-trained model
pre_trained_model = create_pre_trained_model()

# Count the total number of parameters and how many are trainable
num_total_params = pre_trained_model.count_params()
num_trainable_params = sum([w.shape.num_elements() for w in pre_trained_model.trainable_weights])

print(f"There are {num_total_params:,} total parameters in this model.")
print(f"There are {num_trainable_params:,} trainable parameters in this model.")

# Test your code!
unittests.test_create_pre_trained_model(create_pre_trained_model)

# Print the model summary
pre_trained_model.summary()

# Define a Callback class that stops training once accuracy reaches 99.9%
class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs['accuracy']>0.999:
            self.model.stop_training = True
            print("\nReached 99.9% accuracy so cancelling training!")


# GRADED FUNCTION: output_of_last_layer

def output_of_last_layer(pre_trained_model):
    """Fetches the output of the last desired layer of the pre-trained model

    Args:
        pre_trained_model (tf.keras.Model): pre-trained model

    Returns:
        tf.keras.KerasTensor: last desired layer of pretrained model
    """
    ### START CODE HERE ###

    last_desired_layer = pre_trained_model.get_layer('mixed7')
    last_output = last_desired_layer.output

    print('last layer output shape: ', last_output.shape)

    ### END CODE HERE ###

    return last_output

last_output = output_of_last_layer(pre_trained_model)

# Test your code!
unittests.test_output_of_last_layer(output_of_last_layer, pre_trained_model)

# Print the type of the pre-trained model
print(f"The pretrained model has type: {type(pre_trained_model)}")


# GRADED FUNCTION: create_final_model

def create_final_model(pre_trained_model, last_output):
    """Creates final model by adding layers on top of the pretrained model.

    Args:
        pre_trained_model (tf.keras.Model): pre-trained inceptionV3 model
        last_output (tf.keras.KerasTensor): last layer of the pretrained model

    Returns:
        Tensorflow model: final model
    """

    # Flatten the output layer of the pretrained model to 1 dimension
    x = tf.keras.layers.Flatten()(last_output)

    ### START CODE HERE ###

    # Add a fully connected layer with 1024 hidden units and ReLU activation
    x = tf.keras.layers.Dense(1024, activation='relu') (x)
    # Add a dropout rate of 0.2
    x = tf.keras.layers.Dropout(0.2) (x)
    # Add a final sigmoid layer for classification
    x = tf.keras.layers.Dense(1, activation='sigmoid') (x)

    # Create the complete model by using the Model class
    model = tf.keras.Model(inputs=pre_trained_model.input, outputs=x)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.00001),
        loss='binary_crossentropy',  # use a loss for binary classification
        metrics=['accuracy']
    )

    ### END CODE HERE ###

    return model

# Save your model in a variable
model = create_final_model(pre_trained_model, last_output)

# Inspect parameters
total_params = model.count_params()
num_trainable_params = sum([w.shape.num_elements() for w in model.trainable_weights])

print(f"There are {total_params:,} total parameters in this model.")
print(f"There are {num_trainable_params:,} trainable parameters in this model.")

# Test your code!
unittests.test_create_final_model(create_final_model, pre_trained_model, last_output)

# Define the preprocess function
def preprocess(image, label):
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    return image, label

# Apply the preprocessing to all datasets
training_dataset = training_dataset.map(preprocess)
validation_dataset = validation_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)

# Run this and see how many epochs it takes before the callback fires
history = model.fit(
    training_dataset,
    validation_data = validation_dataset,
    epochs = 100,
    verbose = 2,
    callbacks = [EarlyStoppingCallback()],
)

# Plot the training and validation accuracies for each epoch

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.show()

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test loss: {test_loss},\nTest accuracy: {test_accuracy}')









