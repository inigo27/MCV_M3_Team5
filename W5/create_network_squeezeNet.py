from tensorflow.keras import Input, activations
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Activation, LayerNormalization, UnitNormalization, MaxPooling2D, BatchNormalization, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import os


img_width = 256
img_height = 256
n_class = 8
MODEL_FOLDER = "./models/squeeze444bru/"
MODEL_NAME = "model.h5"

norm = BatchNormalization
init = "random_uniform"

def fire(x, squeeze, expand, activation):
    y = Conv2D(filters=squeeze, kernel_size=(1, 1), padding='same', kernel_initializer=init)(x)
    y = Activation(activation)(y)
    y =  norm()(y)
    y1 = Conv2D(filters=expand//2, kernel_size=(1, 1), padding='same', kernel_initializer=init)(y)
    y1 = Activation(activation)(y1)
    y1 = norm()(y1)
    y3 = Conv2D(filters=expand//2, kernel_size=(3, 3), padding='same', kernel_initializer=init)(y)
    y3 = Activation(activation)(y3)
    y3 = norm()(y3)
    return Concatenate()([y1, y3])


if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# Create model
# First conv
img_inputs = Input(shape=(img_height, img_width, 3))
conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding="SAME", kernel_initializer=init)(img_inputs)
act1 = Activation(activations.relu)(conv1)
batch1 = norm()(act1)

# Max pooling
mPool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(batch1)

# Fire 1
y = fire(mPool, 16, 32, activations.relu)
mPool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(y)

# Fire 2
y = fire(mPool1, 16, 64, activations.relu)
mPool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(y)

# Fire 3
y = fire(mPool2, 32, 128, activations.relu)
mPool3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(y)

# Fire 4
y = fire(mPool3, 64, 256, activations.relu)
# mPool4 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(y)

# # Fire 5
# y = fire(mPool4, 128, 512, activations.relu)

global_pool = GlobalAveragePooling2D()(y)
fc = Dense(n_class, activation='softmax', name='predictions')(global_pool)

model = Model(inputs=img_inputs, outputs=fc)
# Plot model
model.summary()
num_p = model.count_params()
plot_model(model, to_file=MODEL_FOLDER + 'modelStruct.png',
           show_shapes=True, show_layer_names=True)
model.save(MODEL_FOLDER + MODEL_NAME)

# Save param number
file = open(MODEL_FOLDER + "modelParam.txt", "w")
file.write(f"Param number: {num_p}\n")
file.close()
