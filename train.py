# -*- coding: utf-8 -*-
"""
Pokemon classification
Author:wen227
Github:https://github.com/wen227/Pokemon_Classification
Reference:
1.https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/
2.https://www.kaggle.com/trolukovich/predicting-pokemon-with-cnn-and-keras/notebook
  These two webs above present examples of VGG neural network model.
3.https://arxiv.org/abs/1409.1556 VGGNet network
4.https://keras.io/zh/applications/
5.https://cloud.tencent.com/developer/article/1038802
"""
# Importing all necessary libraries
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

from Model import CNN
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'  # set Graphviz path

# A little bit of data exploration
path = r'dataset\PokemonTrainData'  # Path to directory which contains classes
classes = os.listdir(path)  # List of all classes

# Set
# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 120
INIT_LR = 1e-3
BATCH = 32
IMAGE_DIMS = (96, 96, 3)  # width, height, depth

# initialize the data and labels
X = []  # List for images
Y = []  # List for labels
# Load dataset
for c in classes:
    dir_path = os.path.join(path, c)
    label = classes.index(c)  # Our label is an index of class in 'classes' list

    # Reading, resizing and adding image and label to lists
    for i in os.listdir(dir_path):
        image = cv.imread(os.path.join(dir_path, i))
        try:
            resized = cv.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))  # Resizing images
            X.append(resized)
            Y.append(label)

        # If we can't read image - we skip it
        except:
            print(os.path.join(dir_path, i), '[ERROR] can\'t read the file')
            continue

print('DONE')
print(len(X))
# Scale the raw pixel intensities to the range [0, 1]
X = np.array(X, dtype="float") / 255.0

# Convert labels to categorical format
y = to_categorical(Y, num_classes=len(classes))  # one-hot?

# Splitting data to train and test dataset
# I'll use these dataset only for training, for final predictions I'll use random pictures from internet
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, shuffle=True,
                                                    random_state=35)
# Defining ImageDataGenerator Iinstance
datagen = ImageDataGenerator(rotation_range=45,  # Degree range for random rotations
                             zoom_range=0.2,  # Range for random zoom
                             horizontal_flip=True,  # Randomly flip inputs horizontally
                             width_shift_range=0.15,  # Range for horizontal shift
                             height_shift_range=0.15,  # Range for vertical shift
                             shear_range=0.2)  # Shear Intensity
datagen.fit(X_train)

# initialize the model
model = CNN.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=len(classes))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.summary()  #
checkpoint = ModelCheckpoint('result\best_model.hdf5', verbose=1, monitor='val_accuracy', save_best_only=True)  #
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# train the network
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH),
                              validation_data=(X_test, y_test),
                              steps_per_epoch=len(X_train) // BATCH,
                              epochs=EPOCHS, verbose=1, callbacks=[checkpoint])

# Plot learning curves
plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.grid()
plt.title(f'accuracy')
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.savefig("result\acc.png")  #
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.grid()
plt.title(f'loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.savefig("result\loss.png")  #
plt.show()

# Loading weights from best model
model.load_weights('result\best_model.hdf5')  #
# Saving all model
model.save('result\model.hdf5')  #
# Plot the model
plot_model(model=model, to_file="result\model.png",  #
           show_layer_names=True, show_shapes=True)
