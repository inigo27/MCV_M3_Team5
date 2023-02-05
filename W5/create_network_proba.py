from tensorflow.keras import Input, activations
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Dense, Activation, MaxPooling2D, BatchNormalization, Concatenate
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import os
import tensorflow as tf

img_width = 256
img_height= 256
n_class = 8
MODEL_FOLDER = "./models/small22/"
MODEL_NAME = "model.h5"

if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# Create model
model = tf.keras.applications.MobileNetV3Small(
    input_shape=(img_height, img_width, 3),
    alpha=1.0,
    minimalistic=False,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    classes=8,
    pooling=None,
    dropout_rate=0.2,
    classifier_activation="softmax",
    include_preprocessing=True,
)
# Plot model
model.summary()
num_p = model.count_params()
plot_model(model, to_file=MODEL_FOLDER + 'modelStruct.png', show_shapes=True, show_layer_names=True)
model.save(MODEL_FOLDER + MODEL_NAME)

# Save param number
file  = open(MODEL_FOLDER + "modelParam.txt", "w")
file.write(f"Param number: {num_p}\n")
file.close()
