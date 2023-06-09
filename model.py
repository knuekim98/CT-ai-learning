import tensorflow as tf
from tensorflow import keras
from keras.applications import MobileNetV3Large
from keras.layers import Conv2D, ReLU, MaxPooling2D, Dense, BatchNormalization, GlobalAveragePooling2D, Concatenate
from check_data import data_size
from get_data import get_data
from const import N_EPOCHS, N_BATCH, N_VAL_BATCH
mobilenet = MobileNetV3Large(include_top=False, input_shape=(224, 224, 3))


def create_model():
	model = keras.models.Sequential()
	model.add(mobilenet)
	model.add(GlobalAveragePooling2D())
	model.add(Dense(256))
	model.add(BatchNormalization())
	model.add(ReLU())
	model.add(Dense(64))
	model.add(BatchNormalization())
	model.add(ReLU())
	model.add(Dense(2, activation='softmax'))
	return model


N_TRAIN, N_VAL = data_size()
learning_rate = 0.0001

train_dataset = get_data('train')
val_dataset = get_data('val')

## Create model, compile & summary
model = create_model()
model.summary()


## learning rate scheduling(필요시), model.compile
'''
lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate,
                                                          decay_steps=N_TRAIN/N_BATCH*10,
                                                          decay_rate=0.5,
                                                          staircase=True)
'''

#model.compile(keras.optimizers.RMSprop(lr_schedule, momentum=0.9), loss=loss_fn)
model.compile(optimizer='adam', loss='mse')
model.fit(train_dataset, epochs=N_EPOCHS, validation_data=val_dataset)
model.save('./model/model.h5')