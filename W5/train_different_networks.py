from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

#train_data_dir='/ghome/mcv/datasets/MIT_split/train'
train_data_dir = "./MIT_small_train_1/train/"
#val_data_dir='/ghome/mcv/datasets/MIT_split/test'
test_data_dir = "./MIT_small_train_1/test"
#test_data_dir='/ghome/mcv/datasets/MIT_split/test'

img_width = 256
img_height= 256
batch_size= 16
number_of_epoch=100
validation_samples=2288
folder = "./models/squeeze444bru/"
MODEL_FNAME = folder + "best_model.h5"
initial_model = folder + "model.h5"

WIDTH_SHIFT = False
HEIGHT_SHIFT = False
HORIZONTAL_FLIP = False
VERTICAL_FLIP = False
ROTATION = False
BRIGHT = False
ZOOM = False
SHEAR = False

# create the base pre-trained model
model = load_model(initial_model)

opt = optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])

model.summary()

# Set data augmentation parameters
if WIDTH_SHIFT:
    width_shift_range = 0.1
else:
    width_shift_range = 0.
if HEIGHT_SHIFT:
    height_shift_range = 0.1
else:
    height_shift_range = 0.
if HORIZONTAL_FLIP:
    horizontal_flip = True
else:
    horizontal_flip = False
if VERTICAL_FLIP:
    vertical_flip = True
else:
    vertical_flip = False
if ROTATION:
    rotation_range = 10
else:
    rotation_range = 0
if BRIGHT:
    brightness_range = [0.8, 1.2]
else:
    brightness_range = [1., 1.]
if ZOOM:
    zoom_range = 0.1
else:
    zoom_range = 0.
if SHEAR:
    shear_range = 10
else:
    shear_range = 0

#preprocessing_function=preprocess_input,
datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
	rescale=1./255.,
    rotation_range=rotation_range,
    width_shift_range=width_shift_range,
    height_shift_range=height_shift_range,
    shear_range=shear_range,
    brightness_range = brightness_range,
    zoom_range=zoom_range,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=horizontal_flip,
    vertical_flip=vertical_flip)

datagenTest = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
	rescale=1./255.,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False)

train_generator = datagen.flow_from_directory(train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = datagenTest.flow_from_directory(test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')


# To save the best model
checkpointer = ModelCheckpoint(filepath=MODEL_FNAME, verbose=1, save_best_only=True, 
                               monitor='val_accuracy')

history=model.fit(train_generator,
        steps_per_epoch=(np.ceil(400/batch_size)),
        epochs=number_of_epoch,
        validation_data=test_generator,
        validation_steps= (np.ceil(validation_samples/batch_size)), callbacks=[checkpointer])

model = load_model(MODEL_FNAME)

result = model.evaluate(test_generator)
print( result)

# list all data in history
num_p = model.count_params()

file  = open(folder + "results.txt", "w")
file.write(f"Score: {result}\n")
file.write(f"Num param: {num_p}\n")
ratio = result[1]/(num_p/100000)
file.write(f"Ratio: {ratio}\n")
file.close()

# summarize history for accuracy
fig1, ax1 = plt.subplots()
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('model accuracy')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')
fig1.savefig(folder + 'accuracy.jpg')
plt.close(fig1)
  # summarize history for loss
fig1, ax1 = plt.subplots()
ax1.plot(history.history['loss'])
ax1.plot(history.history['val_loss'])
ax1.set_title('model loss')
ax1.set_ylabel('loss')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')
fig1.savefig(folder + 'loss.jpg')
