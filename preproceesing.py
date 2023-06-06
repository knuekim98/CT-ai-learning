import tensorflow as tf
from tensorflow import keras
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

fn = './RIDER Lung CT/train-data/9012-2/1-111.dcm'
dcm = pydicom.read_file(fn)
img = o = dcm.pixel_array

slope = int(dcm.RescaleSlope)
b = int(dcm.RescaleIntercept)
img = img * slope + b

wc = -1400
ww = 1600
dcm.WindowCenter = wc
dcm.WindowWidth = ww
img = apply_modality_lut(img, dcm)
img = apply_voi_lut(img, dcm)

img_min = np.min(img)
img_max = np.max(img)
img = img - img_min
img = img / (img_max-img_min)
img *= 2**8-1
img = img.astype(np.uint8)

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