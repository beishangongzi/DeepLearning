import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf

batch_size = 32
img_height = 224
img_width = 180
data_dir = "/content/drive/MyDrive/tf/VGG16/Dataset_Palace_Museum"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir,
                                                               validation_split=0.2,
                                                               subset="training",
                                                               seed=123,
                                                               image_size=(img_height, img_width),
                                                               batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split=0.2, subset="validation", seed=123,
                                                             image_size=(img_height, img_width),
                                                             batch_size=batch_size)


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


num_classes = 10

model = tf.keras.Sequential([
                             tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
                             tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
                             tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
                             tf.keras.layers.MaxPool2D(),
                             tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
                             tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
                             tf.keras.layers.MaxPool2D(),
                             tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
                             tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
                             tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
                             tf.keras.layers.MaxPool2D(),
                             tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu"),
                             tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu"),
                             tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu"),
                             tf.keras.layers.MaxPool2D(),
                             tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu"),
                             tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu"),
                             tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu"),
                             tf.keras.layers.MaxPool2D(),
                             tf.keras.layers.Flatten(),
                             tf.keras.layers.Dense(4096),
                             tf.keras.layers.Dense(4096),
                             tf.keras.layers.Dense(4096),
                             tf.keras.layers.Dense(num_classes, activation="softmax")
                             
])

model.compile(optimizer="adam",
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=["accuracy"])

model.fit(train_ds, validation_data=val_ds, epochs=15)
