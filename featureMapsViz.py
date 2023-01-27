from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, InceptionResNetV2
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import optimizers
from matplotlib import pyplot
from numpy import expand_dims
import pydot
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


train_data_dir='./mcv/datasets/MIT_split/train'
#train_data_dir = "./MIT_split/train/"
#val_data_dir='/ghome/mcv/datasets/MIT_split/test'
#test_data_dir = "./MIT_split/test"
test_data_dir='./mcv/datasets/MIT_split/test'
#####
img_width = 299
img_height= 299
batch_size=32
number_of_epoch=20
validation_samples=807



img_path = './mcv/datasets/MIT_split/train/coast/arnat59.jpg'
img = image.load_img(img_path, target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x) 
# create the base pre-trained model
base_model = InceptionResNetV2(weights='imagenet')
#plot_model(base_model, to_file='modelInceptionResNetV2.png', show_shapes=True, show_layer_names=True)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('conv2d_1').output)
feature_maps = model.predict(x)

square = 2
index = 1
for _ in range(square):
 for _ in range(square):
        
		ax = plt.subplot(square, square, index)
		ax.set_xticks([])
		ax.set_yticks([])

		plt.imshow(feature_maps[0, :, :, index-1], cmap='viridis')
		index += 1
        
plt.show()
# Select one feature map
feature_map = feature_maps[0, :, :, 1]
#Remove the 3rd dimension
feature_map = np.squeeze(feature_map)

# Plot the feature map using imshow
plt.imshow(feature_map, cmap='viridis')
plt.show()









