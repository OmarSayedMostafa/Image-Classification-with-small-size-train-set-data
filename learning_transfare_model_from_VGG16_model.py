import numpy as np  
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from keras.models import Sequential  
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import applications  
from keras.utils.np_utils import to_categorical  
import matplotlib.pyplot as plt  
import math  
import cv2  
from keras.callbacks import TensorBoard
from keras.regularizers import l1, l2
import os

from settings import *
from utilities import save_prediction_to_csv


# log training and graph to view in tensorboard
tenorboard = TensorBoard(log_dir=LEARNING_TRANSFER_TENSORBOARD_LOG_DIR+'/logs')

#data generator for data set augmentation 
datagen = ImageDataGenerator(rescale=1. / 255,rotation_range=10.0, horizontal_flip=True)  

# load data set from directory
'''
data-set:
    class-dir1
    class-dir2
    .
    .
    .
    class-dir(n)
'''

train_gen = datagen.flow_from_directory(  
    TRAIN_DATA_DIR,  
    target_size=(IMAGE_SIZE, IMAGE_SIZE),  
    batch_size=BATCH_SIZE,  
    class_mode=None,  
    shuffle=False) # shuffle= False because we are gonna forward all data set to the pre trained network once and save it to the disk 
                   # so we can save some time in training process , so we need to load the data set in order so we can mapp it with it's labels later.
    

#load validation set "same directory structure as data set dir structre"
valid_gen = datagen.flow_from_directory(  
    VALID_DATA_DIR,  
    target_size=(IMAGE_SIZE, IMAGE_SIZE),  
    batch_size=BATCH_SIZE,  
    class_mode=None,  
    shuffle=False)  #same goes with validation set.




def initialise_top_model(_input_shape, Dense_layers_dims =[512, 256, 128], number_of_classes = len(CATEGORIES), activation = 'relu', keep_prob= LERNING_TRANSFARE_KEEP_PROB):
    ''' this is a function that initialize top fully connected layers that will be added to pre-tarined model to fine tunning it to classify the desired classes.

    args:
        _input_shape: the shape of the output features that extracted by the pre-trained model and the shape of input to the top fully connected layers.
        Dense_layers_dims: a list contains the dimension of the fully connected / Dense layers.
        number_of_classes: number of classes to classify, defualt is length of categories/ classes list that already defined in settings.py
        activation: the activation function for each layer of fc layers.
        keep_prob: the keep probablity hyper parameter for applying Dropout regulization technique.

    Returns:
        returns the model
    '''
    model = Sequential()
    model.add(Flatten(input_shape=_input_shape))  
    #, activity_regularizer=l2(0.01)
    for dim in Dense_layers_dims:
        model.add(Dense(dim))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(Dropout(keep_prob))
    
    model.add(Dense(number_of_classes))
    model.add(Activation('softmax'))

   
    return model





def pre_trained_bottlebeck_features_prediction(train_generator= train_gen, valid_generator= valid_gen, save_flag=True):
    ''' this function predict the bottle neck features of pre trainned model and save it to the disk to use it again later 

    args:
        train_generator : the trained generator that contains training data.
        validation data : the validation generator that contains validation data.

    Returns:
        bottleneck_features_train : the predicted features of train set from pre trainned model
        bottleneck_features_validation : the predicted features of validation set from pre trainned model
    '''
    #fix number of steps 
    remaining_size_of_train_examples = int(math.ceil(len(train_generator.filenames) / train_generator.batch_size))

    model = applications.VGG16(include_top=False, weights='imagenet')  
    
    bottleneck_features_train = model.predict_generator(train_generator, remaining_size_of_train_examples)  
    
    if save_flag:
        np.save(BOTTLE_NECK_FEATURES_SAVE_LOC+'bottleneck_features_train.npy', bottleneck_features_train)

    #-----------------------------------------------------------------------------------
    #same goes with validation data
    remaining_size_of_validation_examples = int(math.ceil(len(valid_generator.filenames) / valid_generator.batch_size))  
   
    bottleneck_features_validation = model.predict_generator(valid_generator, remaining_size_of_validation_examples)  
   
    if save_flag:
        np.save(BOTTLE_NECK_FEATURES_SAVE_LOC+'bottleneck_features_validation.npy', bottleneck_features_validation)


    return bottleneck_features_train, bottleneck_features_validation   




def train_top_model(train_generator= train_gen, valid_generator= valid_gen,epochs=EPOCHS, batch_size=BATCH_SIZE, top_model_weights_path= LEARNING_TRANSFARE_MODEL+"weights/52.h5", load_bottle_neck_features_flag=True):
    ''' this function is responsible for training the top fc layers.

    args:
        train_generator : the trained generator that contains training data.
        validation data : the validation generator that contains validation data.
        epochs : number of training epochs.
        batch_size : the of single batch.
        top_model_weights_path : the path of pretrained weights if there was a pretraining

    '''
    # load the bottleneck features saved earlier
    train_data, validation_data = None, None
    if load_bottle_neck_features_flag:
        try:   
            train_data = np.load('bottleneck_features_train.npy')
            validation_data = np.load('bottleneck_features_validation.npy')
        except:
            print("cannot load "+BOTTLE_NECK_FEATURES_SAVE_LOC+"bottleneck_features_train.npy or "+BOTTLE_NECK_FEATURES_SAVE_LOC+'bottleneck_features_validation.npy')
            print("will call pre_trained_bottlebeck_features_prediction to get bottleneck_features")
            train_data , validation_data = pre_trained_bottlebeck_features_prediction(train_generator= train_gen, valid_generator= valid_gen)
    else:
         train_data , validation_data = pre_trained_bottlebeck_features_prediction(train_generator= train_gen, valid_generator= valid_gen)
    
    # get the class lebels for the training data, in the original order 
    train_labels = train_generator.classes  
    # convert the training labels to categorical vectors  
    train_labels = to_categorical(train_labels, num_classes=len(train_generator.class_indices))


    # get the class lebels for the training data, in the original order 
    validation_labels = valid_generator.classes  
    # convert the training labels to categorical vectors  
    validation_labels = to_categorical(validation_labels, num_classes=len(valid_generator.class_indices))
    
    input_to_model_shape= train_data.shape[1:]
    model = initialise_top_model(input_to_model_shape)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    try:
        model.load_weights(top_model_weights_path)
    except:
        print("cannot load pretrained weights")

    model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(validation_data, validation_labels), callbacks=[tenorboard])  
   
   
    (eval_loss, eval_accuracy) = model.evaluate(validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))  
    print("[INFO] Loss: {}".format(eval_loss))  

    model.save_weights(top_model_weights_path+"model_weights.h5")  




def test(image_path, top_model_weights_path=LEARNING_TRANSFARE_MODEL+"weights"):
    
    image = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))  
    image = img_to_array(image)  
   
    # normalize! 
    image = image / 255.0  
   
    image = np.expand_dims(image, axis=0)
    vgg16_model = applications.VGG16(include_top=False, weights='imagenet')  
    
    # get the bottleneck prediction from the pre-trained VGG16 model  
    bottleneck_prediction = vgg16_model.predict(image)
    
    top_model = initialise_top_model(bottleneck_prediction.shape[1:])
    
    top_model.load_weights(top_model_weights_path)  
   
    # use the bottleneck prediction on the top model to get the final classification  
    class_predicted = top_model.predict_classes(bottleneck_prediction)  
   
    inID = class_predicted[0]
    
    class_dictionary = train_gen.class_indices  
    inv_map = {v: k for k, v in class_dictionary.items()}  
    label = inv_map[inID]    

    return label   



def batch_test(test_dir = TEST_DATA_DIR, top_model_weights_path=LEARNING_TRANSFARE_MODEL+"weights"):

    vgg16_model = applications.VGG16(include_top=False, weights='imagenet')  

    top_model = initialise_top_model([4,4,512])
    top_model.load_weights(top_model_weights_path)  

    prediction_df = {"class":[], "image_name":[]}


    for img_name in os.listdir(test_dir):
        image = load_img(test_dir+img_name, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        original_img = image  
        image = img_to_array(image)  
        # normalize! 
        image = image / 255.0  

        image = np.expand_dims(image, axis=0)
       
        # get the bottleneck prediction from the pre-trained VGG16 model  
        bottleneck_prediction = vgg16_model.predict(image)
        # use the bottleneck prediction on the top model to get the final classification  
        class_predicted = top_model.predict_classes(bottleneck_prediction)  
       
        inID = class_predicted[0]
        class_dictionary = train_gen.class_indices  
        inv_map = {v: k for k, v in class_dictionary.items()}  
        label = inv_map[inID]

        plt.imshow(original_img)
        plt.title(label)
        plt.show()
        
        prediction_df["image_name"].append(img_name)
        prediction_df["class"].append(label)
         
    #save_prediction_to_csv(prediction_df, LEARNING_TRANSFARE_TOP_MODEL_WEIGHTS_PATH)
    





train_top_model()
#test('test-data/3605.jpg',top_model_weights_path='LEARNING_TRANSFARE_TOP_MODEL_WEIGHTS/bottleneck_fc_model.h5')
#batch_test(test_dir='test-data/',top_model_weights_path='LEARNING_TRANSFARE_TOP_MODEL_WEIGHTS/bottleneck_fc_model.h5')
