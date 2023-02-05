from tensorflow.keras import Input, activations
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Activation, UnitNormalization, LayerNormalization, MaxPooling2D, BatchNormalization, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import os
import tensorflow as tf

norm = BatchNormalization
init = "glorot_normal"

img_width = 256
img_height= 256
n_class = 8
MODEL_FOLDER = "./models/common29/"
MODEL_NAME = "model.h5"

if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# Create model
img_inputs = Input(shape=(img_height, img_width, 3))
conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=init)(img_inputs)
act1 = Activation(activations.relu)(conv1)
batch1 = norm()(act1)
mPool1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(batch1)

conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=init)(mPool1)
act2 = Activation(activations.relu)(conv2)
batch2 = norm()(act2)
mPool2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(batch2)


conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=init)(mPool2)
act3 = Activation(activations.relu)(conv3)
batch3 = norm()(act3)
mPool3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(batch3)

conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=init)(mPool3)
act4 = Activation(activations.relu)(conv4)
batch4 = norm()(act4)
mPool4 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(batch4)

conv5 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=init)(mPool4)
act5 = Activation(activations.relu)(conv5)
batch5 = norm()(act5)

global_pool = GlobalAveragePooling2D()(batch5)
fc = Dense(n_class, activation='softmax',name='predictions')(global_pool)

model = Model(inputs=img_inputs, outputs=fc)

# Plot model
model.summary()
num_p = model.count_params()
plot_model(model, to_file=MODEL_FOLDER + 'modelStruct.png', show_shapes=True, show_layer_names=True)
model.save(MODEL_FOLDER + MODEL_NAME)

# Save param number
file  = open(MODEL_FOLDER + "modelParam.txt", "w")
file.write(f"Param number: {num_p}\n")
file.close()
