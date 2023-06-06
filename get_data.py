import tensorflow as tf
from check_data import data_size
N_TRAIN, N_VAL = data_size()
N_BATCH = 40
N_EPOCHS = 40
N_VAL_BATCH = 23


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


def get_data(data_type):
    if data_type == 'train':
        train_dataset = tf.data.TFRecordDataset('./RIDER Lung CT/tfrecord/train.tfr')
        train_dataset = train_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.shuffle(buffer_size=N_TRAIN).prefetch(tf.data.experimental.AUTOTUNE).batch(N_BATCH)
        return train_dataset
    else:
        val_dataset = tf.data.TFRecordDataset('./RIDER Lung CT/tfrecord/val.tfr')
        val_dataset = val_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(N_VAL_BATCH)
        return val_dataset