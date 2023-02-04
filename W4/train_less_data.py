from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#train_data_dir='/ghome/mcv/datasets/MIT_split/train'
train_data_dir = "../W3/MIT_split/train/"
#val_data_dir='/ghome/mcv/datasets/MIT_split/test'
test_data_dir = "../W3/MIT_split/test"
#test_data_dir='/ghome/mcv/datasets/MIT_split/test'

img_width = 299
img_height= 299
batch_size=32
number_of_epoch=50
validation_samples=2288
MODEL_FNAME = "best_model.h5"
initial_model = "./RemovedIntermediateLayers/36/initialRemoved.h5"

# create the base pre-trained model
model = load_model(initial_model)

opt = optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])


model.summary()

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
        steps_per_epoch=(int(400//batch_size)+1),
        epochs=number_of_epoch,
        validation_data=test_generator,
        validation_steps= (int(validation_samples//batch_size)+1), callbacks=[checkpointer])

model = load_model(MODEL_FNAME)

result = model.evaluate(test_generator)
print( result)
print(history.history.keys())


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
