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
from tensorflow.keras.optimizers import Adam
from keras import callbacks
import pickle

x = pickle.load(open("pickles/x.pickle","rb"))
y = pickle.load(open("pickles/y.pickle","rb"))

x = x/255.0
y = np.array(y)

histories = {}

# Hyperparameter search space
dense_layers = [0,1,2]
conv_layers = [1,2,3]

layer_sizes = [32, 64, 128]
kernel_sizes = [(3,3), (5,5)]
dense_sizes = [32, 64, 128]
batch_sizes = [16, 32]


epochs = [10, 20]
# learning_rate = [0.001, 0.0001]

earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min",
                                        patience=2,
                                        restore_best_weights=True)

index = 1

# Iterate over hyperparameters
for dense_layer in dense_layers:
    for conv_layer in conv_layers:
        for layer in layer_sizes:
            for kernel_size in kernel_sizes:
                for dense_size in dense_sizes:
                    for batch_size in batch_sizes:
                        for epoch in epochs:

                            log_string = f"\nTest {index}: DenseLayers={dense_layer}, ConvLayers={conv_layer}, LayerSize={layer}, Kernel={kernel_size}, Dense={dense_size}, Batch={batch_size}, Epochs={epoch}\n\n"
                            print(log_string)



                            model = Sequential()
                            model.add(Input(shape=x.shape[1:]))
                            model.add(Conv2D(layer, kernel_size)) # input_shape=x.shape[1:]
                            model.add(Activation("relu"))
                            model.add(MaxPooling2D(pool_size=(2,2)))

                            for i in range (conv_layer -1):
                                model.add(Conv2D(layer, kernel_size))
                                model.add(Activation("relu"))
                                model.add(MaxPooling2D(pool_size=(2,2)))

                            model.add(Flatten())
                            for i in range(dense_layer):
                                model.add(Dense(dense_size))
                                model.add(Activation("relu"))

                            model.add(Dense(1))
                            model.add(Activation('sigmoid'))

                            
                            


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


                            # optimizer = Adam(learning_rate=lr)
                            model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
                            
                            histories[log_string] = (model.fit(x, y, validation_split=0.2, epochs=epoch, batch_size=batch_size, verbose=1, callbacks=[earlystopping])).history # callbacks=[EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)]
                            index += 1
                            print(histories)



# model.compile(loss="binary_crossentropy",                #couldve been categorical_crossentropy
#             optimizer="adam",
#             metrics=['accuracy'])                    
# fitting_history = model.fit(x, y, batch_size=32, epochs=10, verbose=1) # validation_split=0.2



# model.save('models/model-2-20_2.keras')


pickle_out = open("pickles/history.pickle", "wb")
pickle.dump(histories, pickle_out)
pickle_out.close()


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

# model = load_model('models/model-2-20_2.keras')

# x_test = pickle.load(open("pickles/x_test.pickle","rb"))
# y_test = pickle.load(open("pickles/y_test.pickle","rb"))

# x_test = x_test/255.0
# y_test = np.array(y_test)

# evaluating_history = model.evaluate(x_test, y_test)
# print("Test Loss:", evaluating_history[0])
# print("Test Accuracy:", evaluating_history[1])


#----------------------------------------------------------Code to Plot Confusion Matrix------------------------------------------------------------#


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score


#in test data, 360 benign pics, and 300 malignant


# Predict the test data
# y_pred = model.predict(x_test)
# threshold = 0.5
# y_pred_classes = (np.array(y_pred) > threshold).astype(int)
# y_true_classes = y_test  # Convert one-hot encoded labels to class labels

# # Compute the confusion matrix
# cm = confusion_matrix(y_true_classes, y_pred_classes)

# # Plot the confusion matrix
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')


# # Calculate precision, recall, F1-score, and accuracy
# precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
# recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
# f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
# accuracy = accuracy_score(y_true_classes, y_pred_classes)

# print("Precision:", precision)
# print("Recall:", recall)
# print("F1-Score:", f1)
# print("Accuracy:", accuracy)
# plt.show()



# plot confusion matrix
# figure out f score, precision, accuracy
# try with kernel size 5

# save fig
