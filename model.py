import tensorflow as tf
from tensorflow import keras
from keras.applications.mobilenet import MobileNet
from keras.layers import Conv2D, ReLU, MaxPooling2D, Dense, BatchNormalization, GlobalAveragePooling2D, Concatenate
import numpy as np
import matplotlib.pyplot as plt
from check_data import data_size
mobilenetv2 = MobileNet(weights='imagenet', include_top=False, input_shape=(512, 512, 3))


def _parse_function(tfrecord_serialized):
    features={'image': tf.io.FixedLenFeature([], tf.string),             
              'x': tf.io.FixedLenFeature([], tf.float32),
              'y': tf.io.FixedLenFeature([], tf.float32),            
             }
    parsed_features = tf.io.parse_single_example(tfrecord_serialized, features)
    
    image = tf.io.decode_raw(parsed_features['image'], tf.uint8)    
    image = tf.reshape(image, [512, 512])
    image = tf.cast(image, tf.float32)/255.
    image = tf.repeat(image[..., tf.newaxis], 3, -1)
    
    x = tf.cast(parsed_features['x'], tf.float32)
    y = tf.cast(parsed_features['y'], tf.float32)
    gt = tf.stack([x, y], -1)
    
    return image, gt


def create_model():
	model = keras.models.Sequential()
	model.add(mobilenetv2)
	model.add(GlobalAveragePooling2D())
	model.add(Dense(256))
	model.add(BatchNormalization())
	model.add(ReLU())
	model.add(Dense(64))
	model.add(BatchNormalization())
	model.add(ReLU())
	model.add(Dense(2, activation='sigmoid'))
	return model


N_TRAIN, N_VAL = data_size()
N_BATCH = 40
N_VAL_BATCH = 23
N_EPOCHS = 40
learning_rate = 0.0001

train_dataset = tf.data.TFRecordDataset('./RIDER Lung CT/tfrecord/train.tfr')
train_dataset = train_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=N_TRAIN).prefetch(tf.data.experimental.AUTOTUNE).batch(N_BATCH)

val_dataset = tf.data.TFRecordDataset('./RIDER Lung CT/tfrecord/val.tfr')
val_dataset = val_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(N_VAL_BATCH)

## Create model, compile & summary
model = create_model()
model.summary()

def loss_fn(y_true, y_pred):
	return keras.losses.MeanSquaredError()(y_true, y_pred)

## learning rate scheduling(필요시), model.compile
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                          decay_steps=N_TRAIN/N_BATCH*10,
                                                          decay_rate=0.5,
                                                          staircase=True)

model.compile(keras.optimizers.RMSprop(lr_schedule, momentum=0.9), loss=loss_fn)
model.fit(train_dataset, epochs=N_EPOCHS, validation_data=val_dataset)
model.save('model.h5')