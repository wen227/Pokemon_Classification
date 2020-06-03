# -*- coding: utf-8 -*-
"""
Pokemon classification
Author:wen227
Github:https://github.com/wen227/Pokemon_Classification
Reference:
1.https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/
2.https://www.kaggle.com/trolukovich/predicting-pokemon-with-cnn-and-keras/notebook
  These two webs present examples of VGG neural network model.
3.https://arxiv.org/abs/1409.1556 VGGNet network
"""

# import the necessary packages
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2 as cv
import os
from sklearn.metrics import confusion_matrix
import imutils


def cm_result(cm):
    # Calculate the accuracy of a confusion_matrix,where parameter 'cm' means confusion_matrix.
    a = cm.shape
    corrPred = 0
    falsePred = 0

    for row in range(a[0]):
        for c in range(a[1]):
            if row == c:
                corrPred += cm[row, c]
            else:
                falsePred += cm[row, c]
    Accuracy = corrPred / (cm.sum())
    return Accuracy


# Path
path_model = r'result\best_model.hdf5'
path_test = r'dataset\PokemonTestData'
# Load the trained convolutional neural network
model = load_model(path_model)
# # Set
IMAGE_DIMS = (96, 96, 3)  # width, height, depth

# initialize the data and labels
X = []  # List for images
Y = []  # List for labels
classes = os.listdir(path_test)
# Load dataset
for c in classes:
    dir_path = os.path.join(path_test, c)
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
X_test = np.array(X, dtype="float") / 255.0
# Convert labels to categorical format
y_test = to_categorical(Y, num_classes=len(classes))

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

# finding accuracy from the confusion matrix.
cm_kFold = confusion_matrix(y_test, y_pred)
print(cm_kFold)
Accuracy = cm_result(cm_kFold)
print('Accuracy of the Pokemon Clasification is: ', Accuracy)


