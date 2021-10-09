from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from simple import *
from shallownet import ShallowNet
from tensorflow.keras.optimizers import Adam
from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

size = 256
sp = SimplePreprocessor(size,size)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float")/255.0

# (trainX, newX, trainY, newY) = train_test_split(data, labels, test_size=0.4, random_state=42)
# (valX, testX, valY, testY) = train_test_split(newX, newY, test_size=0.5, random_state=42)
(trainX, valX, trainY, valY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
valY = LabelBinarizer().fit_transform(valY)
# testY = LabelBinarizer().fit_transform(testY)

print("[INFO] compiling model...")
opt = Adam(lr=0.001)
model = ShallowNet.build(width=size, height=size, depth=3, classes=5)
model.compile(loss="categorical_crossentropy", optimizer=opt,
metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(valX, valY), batch_size=32, epochs=10, verbose=1)

model.save("tyre.h5")

# print("[INFO] evaluating network...")
# predictions = model.predict(testX, batch_size=32)
# print(classification_report(testY.argmax(axis=1),
#             predictions.argmax(axis=1),
#             target_names=["blue","green", "ref", "while", "yellow"]))