#------------------------------------------------------------Code to Normalize Training Data------------------------------------------------------------#

import numpy as np              # to move through arrays
import matplotlib.pyplot as plt # to show image
import os                       # go through directories
import cv2                      # for image operations
import random
import pickle

DATADIR = "/Users/samsolano/Documents/WorkFolder/SeniorProject/Skin_Cancer_Archive/train"
CATEGORIES = ["benign", "malignant"]


# training_data = []

# def create_training_data():
#     for category in CATEGORIES:
#         path = os.path.join(DATADIR, category) # construct path to benign or malignant
#         class_num = CATEGORIES.index(category)
#         for img in os.listdir(path):
#             try:
#                 img_array = cv2.imread(os.path.join(path, img))           # get rid of grayscale eventually because color is important 
#                 # new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))                         # figure out what these plt functions do
#                 img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
#                 training_data.append([img_array, class_num])
#             except Exception as e:
#                 continue


# create_training_data()


# # plt.imshow(img_array[0], cmap='gray')                                              # figure out how these plt functions work specifically
# # plt.show()

# random.shuffle(training_data)           # currently its all dog pictures then cat pictures, so shuffle so network doesnt learn to just guess dog then cat

# x, y = [], []       # x is feature, y is label

# for features,label in training_data:
#     x.append(features)
#     y.append(label)


# x = np.array(x)    # Figure out wtf these functions are and why x has to be numpy array!!!!!    -1 is for all features/pics, 1 is for gray scale, would be 3 for full color

# pickle_out = open("pickles/x.pickle", "wb")
# pickle.dump(x, pickle_out)
# pickle_out.close()

# pickle_out = open("pickles/y.pickle", "wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()

# print("Hello world!")


#----------------------------------------------------------Code to Normailze Testing Data------------------------------------------------------------#

# DATADIRTEST = "/Users/samsolano/Documents/WorkFolder/SeniorProject/Skin_Cancer_Archive/test"

# training_data = []

# def create_training_data():
#     for category in CATEGORIES:
#         path = os.path.join(DATADIRTEST, category) # construct path to benign or malignant
#         class_num = CATEGORIES.index(category)
#         for img in os.listdir(path):
#             try:
#                 img_array = cv2.imread(os.path.join(path, img))           # get rid of grayscale eventually because color is important 
#                 # new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))                         # figure out what these plt functions do
#                 img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
#                 training_data.append([img_array, class_num])
#             except Exception as e:
#                 continue


# create_training_data()

# random.shuffle(training_data)           # currently its all dog pictures then cat pictures, so shuffle so network doesnt learn to just guess dog then cat

# x, y = [], []       # x is feature, y is label

# for feature,label in training_data:
#     x.append(feature)
#     y.append(label)


# x = np.array(x)    # Figure out wtf these functions are and why x has to be numpy array!!!!!    -1 is for all features/pics, 1 is for gray scale, would be 3 for full color


# pickle_out = open("pickles/x_test.pickle", "wb")
# pickle.dump(x, pickle_out)
# pickle_out.close()

# pickle_out = open("pickles/y_test.pickle", "wb")
# pickle.dump(y, pickle_out)
# pickle_out.close()


#----------------------------------------------------------Code to Train and Save Model------------------------------------------------------------#
from sklearn.model_selection import KFold
import tensorflow as tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, Input
import pickle

# x = pickle.load(open("pickles/x.pickle","rb"))
# y = pickle.load(open("pickles/y.pickle","rb"))

# x = x/255.0
# y = np.array(y)


# # # Set up K-Fold Cross-Validation
# # kf = KFold(n_splits=5, shuffle=True, random_state=42)
# # fold_accuracies = []

# # # Perform 5-Fold Cross-Validation
# # for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
# #     print(f"\nTraining on fold {fold + 1}...")

# #     x_train, x_val = x[train_idx], x[val_idx]
# #     y_train, y_val = y[train_idx], y[val_idx]



# model = Sequential([
#     Input(shape=x.shape[1:]), #can mess with shape too
#     Conv2D(64, (3,3)),  
#     Activation("relu"),
#     MaxPooling2D(pool_size=(2,2)),

#     Conv2D(64, (3,3)),
#     Activation("relu"),
#     MaxPooling2D(pool_size=(2,2)),

#     Flatten(),  # do i need this
#     Dense(64),
#     Activation("relu"),
#     Dense(1),
#     Activation('sigmoid')
# ])

# model.compile(loss="binary_crossentropy",                #couldve been categorical_crossentropy
#             optimizer="adam",
#             metrics=['accuracy'])                    


# fitting_history = model.fit(x, y, batch_size=32, epochs=10, verbose=1) # validation_split=0.2

# # history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val), verbose=1)

# #     # Store accuracy of this fold
# #     val_acc = history.history['val_accuracy'][-1]  # Take last validation accuracy
# #     fold_accuracies.append(val_acc)

# #     print(f"Fold {fold + 1} Validation Accuracy: {val_acc:.4f}")

# # print("\nFinal Cross-Validation Accuracy:", np.mean(fold_accuracies))

# model.save('models/model-2-20_2.keras')


# pickle_out = open("pickles/history.pickle", "wb")
# pickle.dump(fitting_history, pickle_out)
# pickle_out.close()


#----------------------------------------------------------Code to Plot Model Fitting ------------------------------------------------------------#

# fitting_history = pickle.load(open("pickles/history.pickle","rb"))

# print(fitting_history.history)


#  #------------Training Loss vs Validation Loss Plotting------------#

# # Extract loss values
# train_loss = fitting_history.history['loss']
# val_loss = fitting_history.history['val_loss']
# epochs = range(1, len(train_loss) + 1)

# print(range(1, len(train_loss) + 1))

# # Plot loss
# plt.figure(figsize=(8,6))
# plt.plot(epochs, train_loss, 'bo-', label='Training Loss')  # 'bo-' = blue dots + line
# plt.plot(epochs, val_loss, 'r^-', label='Validation Loss')  # 'r^-' = red triangles + line
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training vs Validation Loss')
# plt.legend()
# plt.grid()
# plt.show()


# #---------Accuracy vs Validation Accuracy Plotting------------#

# train_acc = fitting_history.history['accuracy']
# val_acc = fitting_history.history['val_accuracy']

# plt.figure(figsize=(8,6))
# plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
# plt.plot(epochs, val_acc, 'r^-', label='Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.title('Training vs Validation Accuracy')
# plt.legend()
# plt.grid()
# plt.show()
 


#----------------------------------------------------------Code to Evaluate Model------------------------------------------------------------#

from tensorflow.keras.models import load_model

model = load_model('models/model-2-20_2.keras')

x = pickle.load(open("pickles/x_test.pickle","rb"))
y = pickle.load(open("pickles/y_test.pickle","rb"))

x = x/255.0
y = np.array(y)

evaluating_history = model.evaluate(x, y)
print(evaluating_history)




# plot confusion matrix
# figure out f score, precision, accuracy
# try with kernel size 5
