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


# Path
path_model = r'result\best_model.hdf5'
path_test = r'example'
# Load the trained convolutional neural network
model = load_model(path_model)

for file in os.listdir(path_test):
    try:
        image = cv.imread(os.path.join(path_test, file))
        # image = cv.imread(path_image)
        output = image.copy()

        # pre-process the image for classification
        image = cv.resize(image, (96, 96))
        image = image.astype("float") / 255.0
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # classify the input image
        proba = model.predict(image)[0]
        idx = np.argmax(proba)
        classes = ['Bulbasaur', 'Charmander', 'Mewtwo', 'Pikachu', 'Squirtle']
        class_id = np.argmax(proba)
        # build the label and draw the label on the image
        label = "{}: {:.2f}% ".format(classes[class_id], proba[idx] * 100)
        output = imutils.resize(output, width=400)
        cv.putText(output, label, (25, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.imwrite(os.path.join(r'result', file), output)

        # show the output image
        cv.imshow("Output", output)
        cv.waitKey(0)
    # If we can't read image - we skip it
    except:
        print(os.path.join(path_test, file), '[ERROR] can\'t read the file')
        continue
