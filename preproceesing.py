import tensorflow as tf
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
WC = -1400
WW = 1600

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def preprocessing(fn):
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
    img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    return img


def create_tfrecord(mode, writer):
    with open(f'./RIDER Lung CT/{mode}/ans.txt', 'r') as f:
        ans = eval(f.read())
    for dir_name in os.listdir(f'./RIDER Lung CT/{mode}/')[:-1]:
        # 임시
        if ans.get(dir_name) == None: continue
        for file_name in os.listdir(f'./RIDER Lung CT/{mode}/{dir_name}/'):
            file_number = int(file_name[2:5])
            if ans[dir_name].get(file_number) == None: continue

            fn = f'./RIDER Lung CT/{mode}/{dir_name}/{file_name}'
            img = preprocessing(fn)

            example = tf.train.Example(features=tf.train.Features(feature={
                'image': _bytes_feature(img.tobytes()),
                'x': _float_feature(ans[dir_name][file_number][0]/512.),
                'y': _float_feature(ans[dir_name][file_number][1]/512.)
            }))
            writer.write(example.SerializeToString())
    writer.close()


writer_train = tf.io.TFRecordWriter('./RIDER Lung CT/tfrecord/train.tfr')
writer_val = tf.io.TFRecordWriter('./RIDER Lung CT/tfrecord/val.tfr')

create_tfrecord('train-data', writer_train)
create_tfrecord('val-data', writer_val)
print('done')

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