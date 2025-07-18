import tensorflow as tf

# Can use callbacks to cancel training based on certain parameters being met
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Sends logs object containing information on training
        if logs['loss'] < 0.4:
            # Halts the training when the loss falls below 0.4
            # Stop if threshold is met
            print('\nLoss is low so cancelling training')
            self.model.stop_training = True
            # callback can be kept in the same file as other code just as a seperate class


fashion_mnist = tf.keras.datasets.fashion_mnist
# Load the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# Normalize the pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, callbacks=[MyCallback()]) # Here is where callback is implements during model training