import tensorflow as tf

batch_size = 32
img_height = 180
img_width = 180
data_dir = 'datasets/flower_photos'

def get_datasets():

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, batch_size=batch_size, seed=123, validation_split=0.2,
                                                                subset="training", image_size=(img_height, img_width))

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split=0.2, subset="validation", seed=123, image_size=(img_height, img_width),
                                                                batch_size=batch_size)

    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds
    
