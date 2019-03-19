# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:22:40 2019
@author: Bogdan : highly inspired from pyimagesearch.com
"""

import os
from typing import Set

import cv2
import numpy as np
import tensorflow as tf
import imutils
from imutils import paths
from PIL import Image

###############################################
### Setting up some path s
###############################################
data_dir = os.path.join(os.getcwd(), 'data')
test_dir = os.path.join(os.getcwd(), 'data_test')

detector_dir = os.path.join(os.getcwd(), 'detector')
detector_protoPath = os.path.join(detector_dir, 'deploy.prototxt')
detector_modelPath = os.path.join(detector_dir, 'res10_300x300_ssd_iter_140000.caffemodel')

embedder_dir = os.path.join(os.getcwd(), 'embedder')
embedder_modelPath = os.path.join(embedder_dir, 'openface_nn4.small2.v1.t7')

###############################################
### Initialize some global variables
###############################################
# load the face detector model, called face_detector, create with SSD framework using ResNet-10 like architecture
print("Loading the face detector (Caffe model)")
detector = cv2.dnn.readNetFromCaffe(detector_protoPath, detector_modelPath)

# load the face embedding model, called OpenFace, for face detection, which gives a 128-d rep of the data
print("Loading the face embedder (Torch model)")
embedder = cv2.dnn.readNetFromTorch(embedder_modelPath)

imagePaths = list(imutils.paths.list_images(data_dir))
knownEmbeddings, knownNames = [], []

###############################################
### Set the hyperparameters
###############################################
min_confidence_loc_face = 0.9  # if confidence of finding face is lower, we don't consider the face
min_confidence_rec_person = 0.2 # if confidence of identifying the person is lower, we don't consider the prediciton (for test)

###############################################
#### Encodding the training data in 128-D
###############################################

# inputing an image through the detector, outputting a resized image with its detections
def image_to_face_det(imagePath):
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)

    #Image.fromarray(image).show()

    # construct a blob from the image # TODO: remove this normalization and see the difference
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # apply OpenCV's deep learning-based face detector to localize faces
    detector.setInput(imageBlob)
    detections = detector.forward()  # shape (?,?,nbr_detections,prop) with prop=(?,proba,corner1,corner2,corner3,corner4)

    return image, detections

# inputing the image and detections through the embeddor, outputting the encodding
def detections_to_encodding(image, detections, det_nbr=None):

    # we're keeping the bounding box with the largest probability (so better if only 1 face)
    if det_nbr is None:
        det_nbr = np.argmax(detections[0, 0, :, 2])

    confidence = detections[0, 0, det_nbr, 2]

    # make sure confidence is largest than a threshold
    if confidence > min_confidence_loc_face:
        # compute the (x, y)-coordinates of the bounding box for the face
        (height, width) = image.shape[:2]
        (startX, startY, endX, endY) = (
                detections[0, 0, det_nbr, 3:7] * np.array([width, height, width, height])).astype("int")

        # extract the face ROI and grab the ROI dimensions
        face = image[startY:endY, startX:endX]
        (fHeight, fWidth) = face.shape[:2]

        # ensure the face width and height are sufficiently large
        if fWidth < 20 or fHeight < 20:
            print(' --- The face found is too small, we dismiss it.')
            return np.array([None])

        # Image.fromarray(face, 'RGB').show()

        # construct a blob for the face ROI, then pass it through the face embedding model to obtain its 128-d rep #TODO: what's the point of swap ?
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True)
        embedder.setInput(faceBlob)
        vec = embedder.forward()

        return vec

    return np.array([None])

# Looping over all images to encode them
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    name = imagePath.split(os.path.sep)[-2].lower()
    print(f'Processing image {i + 1}/{len(imagePaths)} -- {name} : {imagePath.split(os.path.sep)[-1]}')

    # loading the image and detecting faces
    image, detections = image_to_face_det(imagePath)

    # creating the encoding vector
    vec = detections_to_encodding(image, detections).flatten() if len(detections) > 0 else np.array([None])

    # add the name of the person + corresponding face embedding to their respective lists
    if vec.all() is not None:
        knownNames.append(name)
        knownEmbeddings.append(vec)

print('Encodded all the data')

# Collapse all prediction to an average one
# unique_names = {name:idx for idx,name in enumerate(list(set(knownNames)))}
# unique_embeddings = [np.zeros(shape=(len(knownEmbeddings[0])))] * len(unique_names)
#
# for i, names in enumerate(knownNames):
#     unique_embeddings[unique_names[names]] += knownEmbeddings[i]
#
# knownEmbeddings = unique_embeddings
# knownNames = list(unique_names.keys())


#TODO: maybe average all answers for the same person ? instead of keeping all of them, might reduce false positives


###############################################
#### Training a Classifier on the encodded pictures
###############################################

# from sklearn.preprocessing import LabelEncoder
# from sklearn.svm import SVC
#
# # encode the labels
# le = LabelEncoder()
# labels = le.fit_transform(knownNames)
#
# # train the model with the 128-d embeddings of the face and produce the face recognition
# print("Training the classifier with the encoded data")
# recognizer = SVC(gamma='auto', probability=True, kernel='rbf', degree=2)
# recognizer.fit(knownEmbeddings, labels)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras import regularizers
from keras.utils import np_utils
from keras import metrics


X_input = np.vstack(knownEmbeddings)

unique_names = list(set(knownNames))
nbr_classes = len(unique_names)
name_to_class_dict = dict((name,cls) for cls, name in enumerate(unique_names))
class_to_name_dict = dict((cls,name) for cls, name in enumerate(unique_names))
Y = np_utils.to_categorical([name_to_class_dict[name] for name in knownNames], nbr_classes)


model = Sequential()
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.00)))
model.add(Dropout(rate=0.5))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.00)))
model.add(Dropout(rate=0.25))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.00)))
model.add(Dropout(rate=0.25))
model.add(Dense(nbr_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', metrics.categorical_accuracy])

model.fit(X_input, Y, epochs=50, batch_size=1)




###############################################
#### Recognize new faces with the trained model on new images
###############################################

names_to_find = ['christa', 'bogdan']


test_imagePaths = list(imutils.paths.list_images(test_dir))

# Looping over all images
for (k, imagePath) in enumerate(test_imagePaths):
    print(f'Predicting image {k + 1}/{len(test_imagePaths)} {imagePath.split(os.path.sep)[-1]}')

    # loading the image and detecting faces
    image, detections = image_to_face_det(imagePath)

    found = {name: False for name in names_to_find}
    # loop over the detections and predict all the ones above the threshold
    for i in range(0, detections.shape[2]):

        vec = detections_to_encodding(image, detections, i)

        if vec.any() is None:
            continue

        # perform classification to recognize the face
        #preds = recognizer.predict_proba(vec)[0]


        preds = model.predict(vec.reshape(1,128))[0]
        j = np.argmax(preds)
        proba = preds[j]

        if proba < min_confidence_rec_person:
            continue

        name = class_to_name_dict[j]
        print(f'{name}, {proba:1.2f}')
        found[name] = True if name in found else False

        #if all(fnd == True for fnd in found.values()):
        # draw the bounding box of the face along with the associated probability
        (height, width) = image.shape[:2]
        (startX, startY, endX, endY) = (detections[0, 0, i, 3:7] * np.array([width, height, width, height])).astype("int")
        text = "{}:{:.1f}%".format(name, proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX - 2, y - 2), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 255), 2)

        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)
