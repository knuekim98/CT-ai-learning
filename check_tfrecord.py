import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from get_data import get_data


train_dataset = get_data('train')


for image, gt in train_dataset.take(1):
    x = gt[:,0]
    y = gt[:,1]
    x_value = int(x[0].numpy()*512)
    y_value = int(y[0].numpy()*512)

    c = Circle((x_value, y_value), radius=3, fill=True, color='red')
    plt.axes().add_patch(c)
    plt.imshow(image[0], cmap='gray')
    plt.show()