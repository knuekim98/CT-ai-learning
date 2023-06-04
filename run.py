import pydicom
import matplotlib.pyplot as plt

fn = './RIDER Lung CT/LD/RIDER-1225316081/01-30-2007-NA-NA-56138/101.000000-NA-90295/1-001.dcm'
dcm = pydicom.dcmread(fn)

img = dcm.pixel_array
plt.imshow(img, cmap=plt.cm.bone)
plt.show()