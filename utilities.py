import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
import pickle
import random
from settings import IMAGE_SIZE,CATEGORIES, TRAIN_DATA_DIR
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import pandas as pd

def image_normalization(img_tensor):
    """
        argument
            - img: input image data in numpy array
        return
            - image normalized in [0,1] range
    """
    img_tensor_normalized = img_tensor / 255.0
    return img_tensor_normalized


def one_hot_matrix(labels, categories_depth):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    CategoriesC -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """
    # Create a tf.constant equal to Categories (depth), name it 'Categories'. 
    categories_depth = tf.constant(categories_depth, name = "depth_of_one_hot_matrix")
    # Use tf.one_hot, be careful with the axis 
    # one_hot_matrix = tf.transpose(tf.one_hot(indices=labels, depth=categories, axis=1))
    one_hot_matrix = tf.one_hot(indices=labels, depth= categories_depth)
    # Create the session 
    with tf.Session() as sess:
        one_hot = sess.run(one_hot_matrix)
    
    return one_hot


def serialize_XY_data(serialize_save_location_path, X, Y):
        pickle_out = open(serialize_save_location_path + "X.pickle","wb")
        pickle.dump(X, pickle_out) 
        pickle_out.close()

        pickle_out = open(serialize_save_location_path + "Y.pickle","wb")
        pickle.dump(Y, pickle_out)
        pickle_out.close()


def deserialisze_XY_data(serialize_save_location_path):
    pickle_in = open(serialize_save_location_path + "X.pickle","rb")
    X = pickle.load(pickle_in)

    pickle_in = open(serialize_save_location_path + "Y.pickle","rb")
    Y = pickle.load(pickle_in)

    return X, Y


def load_data(data_path, preprocessing_need_flag = False, categories_list = [], serialize_save_location_path = ""):
    
    if(preprocessing_need_flag):
        X, Y = load_data_set_from_disk(data_path, categories_list,serialize_save_location_path=serialize_save_location_path, serialize_flage= True)


    else:
         X, Y = deserialisze_XY_data(serialize_save_location_path)
        

    # convert Y to one hot matrix
    Y = one_hot_matrix(Y, len(categories_list))

    return X, Y



def load_data_set_from_disk(data_set_path, categories_list, serialize_save_location_path = "", serialize_flage = False):
        
    # load tarin data set with it's labels as numbers from 0 -> 8 (9 categories) for shuffling later
    train_set_data_with_labels = []
        
    # i prefer using X , Y as featurs and labels variable names
    X = []
    Y = []
        
    faild_to_load_counter = 0
        
    for category in categories_list: 
                
        path = os.path.join(data_set_path, category)
                
        for img in os.listdir(path):
            try:
                img_arr = cv2.cvtColor(cv2.imread(os.path.join(path, img)), cv2.COLOR_BGR2RGB) # opencv load RGB image as BGR so it needs to be converted back to RGB
                class_num = categories_list.index(category) #mapping each category to its index number
                image = cv2.resize(img_arr, (IMAGE_SIZE, IMAGE_SIZE)) # IMAGE_SIZE is already difined in settings file 
                train_set_data_with_labels.append([image, class_num])
            except Exception as e:
                print(e)
                faild_to_load_counter= faild_to_load_counter+1
                print("Category "+ category + " Faild to load "+ img)
                print("number of faild images to load = ", faild_to_load_counter)


    # SHUFFLING THE DATA WGICH WHY I COMBINED BOTH OF FEATURES AND LABELS TOGETHER 
    random.shuffle(train_set_data_with_labels)

    #NOW SEPRATING THE FEATURES AND LABLES AGAIN FOR MODLE FEEDING PURPOSES!
    for feature, label in train_set_data_with_labels:
            X.append(feature)
            Y.append(label)
    
    X = np.array(X)
    # normalize the image data from 0 to one by dividing each pixl by 255.0
    X = image_normalization(X)

    Y = np.array(Y)


    if(serialize_flage):
        print("serializing data")
        serialize_XY_data(serialize_save_location_path, X, Y)
    


    return X, Y


def load_single_img(img_path):
    image = None
    try:
        img_arr = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) # opencv load RGB image as BGR so it needs to be converted back to RGB
        image = cv2.resize(img_arr, (IMAGE_SIZE, IMAGE_SIZE)) # IMAGE_SIZE is already difined in settings file 
    except Exception as e:
        print(e)
        print("failed to load img from " + img_path)

    return image


def augmentation(categories_list=CATEGORIES, data_set_path=TRAIN_DATA_DIR):
    datagen = ImageDataGenerator(
        rotation_range=50,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='constant',
        cval=255)

    for category in categories_list:
        if(category not in [ "brie cheese"]):
            continue
                
        
        path = os.path.join(data_set_path, category)
                
        for image in os.listdir(path):
            try:
                img = load_img(os.path.join(path, image))
                x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
                x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

                # the .flow() command below generates batches of randomly transformed images
                # and saves the results to the `preview/` directory
                i = 0
                for batch in datagen.flow(x, batch_size=1, save_to_dir=path, save_prefix='image', save_format='jpg'):
                    i += 1
                    if i > 5:
                        break  # otherwise the generator would loop indefinitely
                    
                    print(category)
            except Exception as e:
                print(e)
                print("Category "+ category + " Faild to load "+ image)


def save_prediction_to_csv(prediction_dict, save_location_path):
    prediction_dataframe = pd.DataFrame(prediction_dict)
    print(prediction_dataframe.head)
    prediction_dataframe.to_csv(save_location_path+"prediction.csv")
