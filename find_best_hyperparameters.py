from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax, softplus, softsign, relu, tanh, sigmoid, hard_sigmoid, linear
from replace_add_intermediate_layers import changeActivation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optuna


#train_data_dir='/ghome/mcv/datasets/MIT_split/train'
train_data_dir = "../W3/MIT_split/train/"
#val_data_dir='/ghome/mcv/datasets/MIT_split/test'
test_data_dir = "../W3/MIT_split/test"
#test_data_dir='/ghome/mcv/datasets/MIT_split/test'
validation_samples=2288
img_width = 299
img_height= 299
MODEL_FNAME = "model_new.h5"
BEST_MODEL = "best_model.h5"
initial_model = "./RemovedIntermediateLayers/36/initialRemoved.h5"

# Hyperparameter values
batch_sizes = [8, 16, 32, 64, 128]
number_of_epochs = [10, 50, 100]
optimizers = [SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam]
learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
momentums = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
activations = [softmax, softplus, softsign, relu, tanh, sigmoid, hard_sigmoid, linear]
data_augmentations = {"width_shift_range": [0.1, 0], "height_shift_range": [0.1, 0], 
                      "horizontal_flip": [True, False], "vertical_flip": [True, False],
                      "rotation_range": [10,0], "brightness_range": [[0.8,1.2],[1.,1.]],
                      "zoom_range": [0.1, 0], "shear_range": [10, 0]}

bestAcc = 0.

def train_network(trial):
    
    # create the base pre-trained model
    model = load_model(initial_model)
    # Choose hyperparameters
    batch_size = trial.suggest_categorical("batch_size", batch_sizes)
    number_of_epoch = trial.suggest_categorical("epochs", number_of_epochs)
    optimizer = trial.suggest_categorical("optimizer", optimizers)
    learning_rate = trial.suggest_categorical("learning_rate", learning_rates)
    if optimizer == SGD or optimizer == RMSprop:
        momentum = trial.suggest_categorical("momentum", momentums)
        opt = optimizer(learning_rate = learning_rate, momentum = momentum)
    else:
        opt = optimizer(learning_rate = learning_rate)
    activation = trial.suggest_categorical("activation", learning_rates)
    model = changeActivation(model, activation)
    
    width_shift_range = trial.suggest_categorical("width_shift_range", data_augmentations["width_shift_range"])
    height_shift_range = trial.suggest_categorical("height_shift_range", data_augmentations["height_shift_range"])
    horizontal_flip = trial.suggest_categorical("horizontal_flip", data_augmentations["horizontal_flip"])
    vertical_flip = trial.suggest_categorical("vertical_flip", data_augmentations["vertical_flip"])
    rotation_range = trial.suggest_categorical("rotation_range", data_augmentations["rotation_range"])
    brightness_range = trial.suggest_categorical("brightness_range", data_augmentations["brightness_range"])
    zoom_range = trial.suggest_categorical("zoom_range", data_augmentations["zoom_range"])
    shear_range = trial.suggest_categorical("shear_range", data_augmentations["shear_range"])
        
    
    model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
    model.summary()

    datagenTrain = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
    	preprocessing_function=preprocess_input,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        brightness_range= brightness_range,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rescale=None)
    
    datagenTest = ImageDataGenerator(featurewise_center=False,
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
    
    train_generator = datagenTrain.flow_from_directory(train_data_dir,
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
            steps_per_epoch=(int(400//batch_size)+1),
            epochs=number_of_epoch,
            validation_data=test_generator,
            validation_steps= (int(validation_samples//batch_size)+1), callbacks=[checkpointer])
    
    model = load_model(MODEL_FNAME)
    
    result = model.evaluate(test_generator)
    
    return result["val_accuracy"]

# Create Study object
study = optuna.create_study(direction="maximize")
# Optimize the study
study.optimize(train_network, n_trials=50) # Use more 
# Print the result
best_params = study.best_params
best_score = study.best_value
print(f"Best score: {best_score}\n")
print(f"Optimized parameters: {best_params}\n")

# plots
fig = optuna.visualization.plot_slice(study, params=['init_neurons', 'layer_num', 'reduction'])
fig.write_image("plot_slice.png")
fig = optuna.visualization.plot_contour(study, params=['init_neurons', 'layer_num', 'reduction'])
fig.write_image("plot_contour.png")
# Save results
file  = open("results.txt", "w")
file.write(f"Best score: {best_score}\n")
file.write(f"Optimized parameters: {best_params}\n")
file.close()