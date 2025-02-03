# for category in CATEGORIES:
#     path = os.path.join(DATADIR, category)
#     for img in os.listdir(path):
#         img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)           #get rid of grayscale eventually because color is important
#         plt.imshow(img_array, cmap='gray')                                              # figure out what these plt functions do
#         plt.show()
#         # print(img_array.shape) 

#         break
#     break

# print(img_array.shape) # Somehow i can print the last img_array even though its out of scope

# This is for resizing but all of our photos are 224x224 so not really a problem perhaps if u want to make it smaller to make processing faster

# new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
# plt.imshow(new_array, cmap='gray')
# plt.show()



import numpy as np              # to move through arrays
import matplotlib.pyplot as plt # to show image
import os                       # go through directories
import cv2                      # for image operations
import random
import pickle

DATADIR = "/Users/samsolano/Documents/WorkFolder/SeniorProject/Skin_Cancer_Archive/train"
CATEGORIES = ["benign", "malignant"]
IMG_SIZE = 50


# training_data = []

# def create_training_data():
#     for category in CATEGORIES:
#         path = os.path.join(DATADIR, category) # construct path to benign or malignant
#         class_num = CATEGORIES.index(category)
#         for img in os.listdir(path):
#             try:
#                 img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)           # get rid of grayscale eventually because color is important 
#                 new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))                         # figure out what these plt functions do
#                 training_data.append([new_array, class_num])
#             except Exception as e:
#                 continue


# create_training_data()

# random.shuffle(training_data)           # currently its all dog pictures then cat pictures, so shuffle so network doesnt learn to just guess dog then cat

# x, y = [], []       # x is feature, y is label

# for features,label in training_data:
#     x.append(features)
#     y.append(label)


# x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)     # Figure out wtf these functions are and why x has to be numpy array!!!!!    -1 is for all features, 1 is for gray scale, would be 3 for full color

# pickle_out = open("x.pickle", "wb")
# pickle.dump(x, pickle_out)
# pickle_out.close()

# pickle_out = open("y.pickle", "wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()

# print("Hello world!")















import tensorflow as tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, Input
import pickle

x = pickle.load(open("x.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

x = x/255.0
y = np.array(y)

model = Sequential([
    Input(shape=x.shape[1:]),
    Conv2D(64, (3,3)),  
    Activation("relu"),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3)),
    Activation("relu"),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(64),
    Activation("relu"),
    Dense(1),
    Activation('sigmoid')
])

model.compile(loss="binary_crossentropy",                #couldve been categorical_crossentropy
              optimizer="adam",
              metrics=['accuracy'])                    


model.fit(x, y, batch_size=32, epochs=10, validation_split=0.2)

















# Todo: 
# figure out how to get this working with color and in original spec of 224x224
# have to adjust network to account for unequal amount of benign vs malignant
# manually split up data, and just fit model then run it on our separate testing data
