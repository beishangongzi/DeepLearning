
model = tf.keras.Sequential([
                             tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(28, 28, 1)),
                             tf.keras.layers.Conv2D(6, 5, (1, 1), "same", activation="relu"),
                             tf.keras.layers.MaxPool2D((2, 2), (2, 2)),
                             tf.keras.layers.Conv2D(16, 5, (1,1), "valid", activation="relu"),
                             tf.keras.layers.MaxPool2D((2,2), (2, 2)),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(120),
                             tf.keras.layers.Dense(84),
                             tf.keras.layers.Dense(num_classes, activation="softmax")
                             
])

model.summary()
