import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from get_data import get_data


model = keras.models.load_model('./model/model.h5', compile=False)
#train_dataset = get_data('train')
val_dataset = get_data('val')

for val_data, val_gt in val_dataset:
    x = val_gt[:,0]
    y = val_gt[:,1]
    x_value = int(x[0]*224)
    y_value = int(y[0]*224)

    c = Circle((x_value, y_value), radius=2, fill=True, color='red', zorder=5)
    axe = plt.axes()
    axe.add_patch(c)

    p = model.predict(val_data)
    px = p[:,0]
    py = p[:,1]
    px_value = int(px[0]*224)
    py_value = int(py[0]*224)
    c2 = Circle((px_value, py_value), radius=2, fill=True, color='blue')
    axe.add_patch(c2)

    plt.imshow(val_data[0], cmap='gray')
    plt.show()
