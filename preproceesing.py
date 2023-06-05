import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle

fn = './RIDER Lung CT/LD/RIDER-1225316081/01-30-2007-NA-NA-56138/101.000000-NA-90295/1-145.dcm'
dcm = pydicom.dcmread(fn)

img = o = dcm.pixel_array
min_img = np.min(img)
max_img = np.max(img)

img = img - min_img
img = img / (max_img-min_img)
img *= 2**8-1
img = img.astype(np.uint8)

img = np.expand_dims(img, axis=-1)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

with open('./RIDER Lung CT/LDP/1.p', 'wb') as f:
    pickle.dump(img, f)

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