import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def _parse_function(tfrecord_serialized):
    features={'image': tf.io.FixedLenFeature([], tf.string),             
              'x': tf.io.FixedLenFeature([], tf.float32),
              'y': tf.io.FixedLenFeature([], tf.float32),            
             }
    parsed_features = tf.io.parse_single_example(tfrecord_serialized, features)
    
    image = tf.io.decode_raw(parsed_features['image'], tf.uint8)    
    image = tf.reshape(image, [512, 512, 1])
    image = tf.cast(image, tf.float32)/255.
    
    x = tf.cast(parsed_features['x'], tf.float32)
    y = tf.cast(parsed_features['y'], tf.float32)
    gt = tf.stack([x, y], -1)
    
    return image, gt

N_TRAIN = 100
N_BATCH = 40

train_dataset = tf.data.TFRecordDataset('./RIDER Lung CT/tfrecord/train.tfr')
train_dataset = train_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=N_TRAIN).prefetch(tf.data.experimental.AUTOTUNE).batch(N_BATCH)


for image, gt in train_dataset.take(1):
    x = gt[:,0]
    y = gt[:,1]
    x_value = int(x[0].numpy()*512)
    y_value = int(y[0].numpy()*512)

    c = Circle((x_value, y_value), radius=3, fill=True, color='red')
    plt.axes().add_patch(c)
    plt.imshow(image[0], cmap='gray')
    plt.show()