#------------------------------------------------------------Code to Normalize Training Data------------------------------------------------------------#

import numpy as np              # to move through arrays
import matplotlib.pyplot as plt # to show image
import os                       # go through directories
import cv2                      # for image operations
import random
import pickle

DATADIR = "C:\\Users\\samue\\OneDrive\\Desktop\\Projects\\seniorProject_skinCancer\\pics\\train"
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

# DATADIRTEST = "C:\\Users\\samue\\OneDrive\\Desktop\\Projects\\seniorProject_skinCancer\\pics\\test"

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

# print("test data done")


#----------------------------------------------------------Code to Train and Save Model------------------------------------------------------------#
import tensorflow as tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, Input
# from tensorflow.keras.optimizers import Adam
from keras import callbacks
import pickle

x = pickle.load(open("pickles/x.pickle","rb"))
y = pickle.load(open("pickles/y.pickle","rb"))

x = x/255.0
y = np.array(y)

histories = {}



earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min",
                                        patience=2,
                                        restore_best_weights=True)


dense_layers = [0]
conv_layers = [3]
kernel_sizes = [(3,3)]


layer_sizes = [128]
# dense_sizes = [32, 64, 128]
dense_sizes = [32]
batch_sizes = [16,32]


epochs = [5]
# learning_rate = [0.001, 0.0001]



index = 1


for dense_layer in dense_layers:
    for conv_layer in conv_layers:
        for layer in layer_sizes:
            for kernel_size in kernel_sizes:
                for dense_size in dense_sizes:
                    for batch_size in batch_sizes:
                        for epoch in epochs:

                            log_string = f"\nTest {index}: Kernel={kernel_size}, Epochs={epoch}\n\n"
                            print(log_string)



                            model = Sequential()
                            model.add(Input(shape=x.shape[1:]))
                            model.add(Conv2D(layer, kernel_size)) 
                            model.add(Activation("relu"))
                            model.add(MaxPooling2D(pool_size=(2,2)))

                            for i in range (conv_layer -1):
                                model.add(Conv2D(layer, kernel_size))
                                model.add(Activation("relu"))
                                model.add(MaxPooling2D(pool_size=(2,2)))

                            model.add(Flatten())
                            # for i in range(dense_layer):
                            #     model.add(Dense(dense_size))
                            #     model.add(Activation("relu"))

                            model.add(Dense(1))
                            model.add(Activation('sigmoid'))



                            model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])

                            histories[log_string] = (model.fit(x, y, validation_split=0.2, epochs=epoch, batch_size=batch_size, verbose=1, callbacks=[earlystopping])).history # callbacks=[EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)]
                            index += 1
                            print(histories)




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



































{'\nTest 1: Kernel=(3, 3), Epochs=5\n\n': 
{'accuracy': 
[0.6216216087341309, 0.7335230112075806, 0.7714556455612183, 0.7885253429412842, 0.7856804132461548], 
'loss': [0.648237943649292, 0.517811119556427, 0.45699718594551086, 0.4235649108886719, 0.42360222339630127],
'val_accuracy': [0.7973484992980957, 0.7007575631141663, 0.7916666865348816, 0.7613636255264282, 0.810606062412262],
'val_loss': [0.45806097984313965, 0.5502316951751709, 0.45434701442718506, 0.4345826208591461, 0.3767198920249939]},

'\nTest 2: Kernel=(3, 3), Epochs=10\n\n': 
{'accuracy': [0.654338538646698, 0.7401612401008606, 0.7738264799118042, 0.7752489447593689, 0.794689416885376],
'loss': [0.6276240348815918, 0.49663904309272766, 0.45246437191963196, 0.43385064601898193, 0.4126872420310974],
'val_accuracy': [0.5795454382896423, 0.7708333134651184, 0.7954545617103577, 0.7613636255264282, 0.7821969985961914],
'val_loss': [0.578090250492096, 0.4456653594970703, 0.4018096327781677, 0.4314216077327728, 0.4712519645690918]},

'\nTest 3: Kernel=(3, 3), Epochs=20\n\n': 
{'accuracy': [0.6405879855155945, 0.7093409299850464, 0.7799904942512512, 0.7738264799118042, 0.7842579483985901], 
'loss': [0.6425449848175049, 0.532450795173645, 0.4539773166179657, 0.45736250281333923, 0.4350755512714386], 
'val_accuracy': [0.7821969985961914, 0.810606062412262, 0.8181818127632141, 0.8162878751754761, 0.8049242496490479], 
'val_loss': [0.49479153752326965, 0.42410361766815186, 0.38424524664878845, 0.39012160897254944, 0.42150962352752686]}, 

'\nTest 4: Kernel=(5, 5), Epochs=5\n\n': 
{'accuracy': [0.6273115277290344, 0.7311521768569946, 0.7283072471618652, 0.7463252544403076, 0.7605500221252441], 
'loss': [0.6759525537490845, 0.5288146138191223, 0.5096479058265686, 0.5058103799819946, 0.4691113829612732], 
'val_accuracy': [0.625, 0.7821969985961914, 0.7916666865348816, 0.7803030014038086, 0.7878788113594055], 
'val_loss': [0.643792986869812, 0.4471815824508667, 0.435253769159317, 0.46460187435150146, 0.4361821115016937]}, 

'\nTest 5: Kernel=(5, 5), Epochs=10\n\n': 
{'accuracy': [0.6239924430847168, 0.7264106273651123, 0.7230914831161499, 0.7572309374809265, 0.7520151734352112, 0.7667140960693359, 0.7714556455612183, 0.7818871736526489, 0.7714556455612183], 
'loss': [0.6488339304924011, 0.553972065448761, 0.5362265110015869, 0.4959684908390045, 0.486129492521286, 0.4704740345478058, 0.45157304406166077, 0.448873370885849, 0.45142653584480286], 
'val_accuracy': [0.6780303120613098, 0.7329545617103577, 0.7935606241226196, 0.7878788113594055, 0.7897727489471436, 0.7708333134651184, 0.7840909361839294, 0.7992424368858337, 0.7746211886405945], 
'val_loss': [0.5566837787628174, 0.5023999214172363, 0.48948732018470764, 0.4581473767757416, 0.4561724066734314, 0.45640432834625244, 0.437534362077713, 0.4397437274456024, 0.4536105692386627]}, 

'\nTest 6: Kernel=(5, 5), Epochs=20\n\n':
{'accuracy': [0.5675675868988037, 0.7022285461425781, 0.7330488562583923, 0.7377904057502747, 0.77098149061203], 
'loss': [0.7033906579017639, 0.5650303959846497, 0.5074737668037415, 0.5305160284042358, 0.47886544466018677], 
'val_accuracy': [0.7159090638160706, 0.7746211886405945, 0.7878788113594055, 0.7803030014038086, 0.7954545617103577], 
'val_loss': [0.5909780859947205, 0.5191885828971863, 0.4605080187320709, 0.4910352826118469, 0.4651860296726227]}}
