import tensorflow as tf
from tensorflow import keras
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
WC = -1400
WW = 1600

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


writer_train = tf.io.TFRecordWriter('./RIDER Lung CT/tfrecord/train.tfr')
writer_val = tf.io.TFRecordWriter('./RIDER Lung CT/tfrecord/val.tfr')

with open('./RIDER Lung CT/train-data/ans.txt', 'r') as f:
    ans = eval(f.read())
for dir_name in os.listdir('./RIDER Lung CT/train-data/')[:-1]:
    # 임시
    if ans.get(dir_name) == None: continue
    for file_name in os.listdir(f'./RIDER Lung CT/train-data/{dir_name}/'):
        file_number = int(file_name[2:5])
        if ans[dir_name].get(file_number) == None: continue

        fn = f'./RIDER Lung CT/train-data/{dir_name}/{file_name}'
        dcm = pydicom.read_file(fn)
        img = o = dcm.pixel_array

        slope = int(dcm.RescaleSlope)
        b = int(dcm.RescaleIntercept)
        img = img * slope + b

        dcm.WindowCenter = WC
        dcm.WindowWidth = WW
        img = apply_modality_lut(img, dcm)
        img = apply_voi_lut(img, dcm)

        img_min = np.min(img)
        img_max = np.max(img)
        img = img - img_min
        img = img / (img_max-img_min)
        img *= 2**8-1
        img = img.astype(np.uint8)

        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(img.tobytes()),
            'x': _int64_feature(ans[dir_name][file_number][0]),
            'y': _int64_feature(ans[dir_name][file_number][1])
        }))
        writer_train.write(example.SerializeToString())
writer_train.close()


'''
fig = plt.figure()

ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(o, cmap='gray')
ax1.set_title('original')
ax1.axis('off')

ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(img, cmap='gray')
ax2.set_title('processed')
ax2.axis('off')

plt.show()
'''