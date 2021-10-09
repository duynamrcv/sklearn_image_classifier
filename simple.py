import numpy as np
import cv2
import os

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height 
        # and interpolation method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize the image to a fixed size, 
        # ignoring the aspect ratio
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)

class SimpleDatasetLoader:
    def __init__(self, preprocessors=[]):
        # store the image preprocessors
        self.preprocessors = preprocessors

    def load(self, imagePaths, verbose=-1):
        # init
        data = []; labels = []

        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            label = imagePath.split(os.path.sep)[-2]

            if len(self.preprocessors) != 0:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            data.append(image)
            labels.append(label)

            if verbose > 0  and i > 0 and (i+1)% verbose == 0:
                print(print("[INFO] processed {}/{}".format(i + 1, len(imagePaths))))
        return (np.array(data), np.array(labels))
