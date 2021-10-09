from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from simple import SimpleDatasetLoader, SimplePreprocessor

from imutils import paths
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
    help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
    help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

print("[INFO] loading images ...")
imagePaths = list(paths.list_images(args["dataset"]))

size = 256
sp = SimplePreprocessor(size,size)
sdl = SimpleDatasetLoader(preprocessors=[sp])

(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], size**2*3))

print("[INFO] features matrix: {:.1f}MB".format(data.nbytes/(size**2*3*1000.0)))

le = LabelEncoder()
labels = le.fit_transform(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.15, random_state=42)

print("[INFO] evaluating SVM classifier")
model = svm.SVC()
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))
