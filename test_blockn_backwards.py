from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, InceptionResNetV2
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import optimizers
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




train_data_dir='/home/mcv/datasets/MIT_split/train'
#train_data_dir = "./MIT_split/train/"
#val_data_dir='/ghome/mcv/datasets/MIT_split/test'
#test_data_dir = "./MIT_split/test"
test_data_dir='/home/mcv/datasets/MIT_split/test'
MODEL_FNAME = '35_1_onwards_mlp.h5'
#####
img_width = 299
img_height= 299
batch_size=32
number_of_epoch=50
validation_samples=807


# create the base pre-trained model
base_model = InceptionResNetV2(weights='imagenet')
#plot_model(base_model, to_file='modelInceptionResNetV2.png', show_shapes=True, show_layer_names=True)

# get the layer with name
layer = base_model.get_layer('block35_1')

# create a new model  that cuts the first one at the layer named before
model = Model(inputs=base_model.input, outputs=layer.output)
x = model.output

#Add an averagePooling to make it shape conpatible
x = GlobalAveragePooling2D()(x)

# add a new dense layer with the output we're looking for 
x = Dense(8, activation='softmax',name='predict1')(x)

# create a new model
model = Model(inputs=model.input, outputs=x)



plot_model(model, to_file='modelInceptionResNetV2_at_35_1.png', show_shapes=True, show_layer_names=True)
for layer in base_model.layers:
    layer.trainable = True
    

opt = optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
for layer in model.layers:
    print(layer.name, layer.trainable)

#preprocessing_function=preprocess_input,
datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
 	preprocessing_function=preprocess_input,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None)

train_generator = datagen.flow_from_directory(train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = datagen.flow_from_directory(test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

# To save the best model
checkpointer = ModelCheckpoint(filepath=MODEL_FNAME, verbose=1, save_best_only=True, 
                               monitor='val_accuracy')

history=model.fit(train_generator,
        steps_per_epoch=(int(1881//batch_size)+1),
        epochs=number_of_epoch,
        validation_data=test_generator,
        validation_steps= (int(validation_samples//batch_size)+1), callbacks=[checkpointer])

model = load_model(MODEL_FNAME)

result = model.evaluate(test_generator)
print( result)
print(history.history.keys())
print(model.summary())
with open("Model_Summary.txt", "a") as f:

    print("count_params: ", file=f)
    print(model.count_params(),file=f)

# list all data in history

if True:
  # summarize history for accuracy
  fig1, ax1 = plt.subplots()
  ax1.plot(history.history['accuracy'])
  ax1.plot(history.history['val_accuracy'])
  ax1.set_title('model accuracy')
  ax1.set_ylabel('accuracy')
  ax1.set_xlabel('epoch')
  ax1.legend(['train', 'validation'], loc='upper left')
  fig1.savefig('accuracy.jpg')
  plt.close(fig1)
    # summarize history for loss
  fig1, ax1 = plt.subplots()
  ax1.plot(history.history['loss'])
  ax1.plot(history.history['val_loss'])
  ax1.set_title('model loss')
  ax1.set_ylabel('loss')
  ax1.set_xlabel('epoch')
  ax1.legend(['train', 'validation'], loc='upper left')
  fig1.savefig('loss.jpg')
