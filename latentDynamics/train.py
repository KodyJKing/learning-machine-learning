from glob import glob
import math
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow import keras

from EpisodeLoader import EpisodeLoader
from model import latentDynamicsModel

layers = tf.keras.layers
models = tf.keras.models

checkpointPath = './checkpoints/latentDynamics'

windowLength = 5
batchSize = 30
observationShape = (210, 160, 3)
numberOfEpisodes = 20
trainingSplit = 0.75

def pltimg(img):
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)

def showSample(x, y, p):
    print( np.shape(x), np.shape(y) )

    cols = windowLength + 2
    rows = len(p)

    plt.figure(figsize=(2 * cols, 3 * rows))
    for row in range(rows):
        for i in range(windowLength):
            plt.subplot(rows, cols, i + 1 + row * cols)
            pltimg(x[row,i])
        
        plt.subplot(rows, cols, windowLength + 1 + row * cols)
        pltimg(y[row])

        plt.subplot(rows, cols, windowLength + 2 + row * cols)
        pltimg(p[row])

    plt.show()

def splitEpisodes(directory, numEpisodes, percent):
    files = glob(directory + r"\*.npz")
    numEpisodes = min(numEpisodes, len(files))
    index = math.floor( numEpisodes * percent )
    trainingFiles = files[0:index]
    validationFiles = files[index:numEpisodes]
    return trainingFiles, validationFiles

trainingFiles, validationFiles = splitEpisodes(
    r"data\breakout_random_actor_episodes_2", 
    numberOfEpisodes, trainingSplit
)

loaderkw = {
    "batchSize": batchSize,
    "windowLength": windowLength + 1,
    "observationShape": observationShape
}

model = latentDynamicsModel(windowLength, observationShape, verbose=False)
model.compile(loss="mean_squared_error", optimizer="adam")

try:
    print("\nChecking for model at:", checkpointPath)
    model.load_weights(checkpointPath)
    print("\nLoaded previous model.\n")
except:
    print("No saved weights. Training...")
    trainingData = EpisodeLoader( trainingFiles, **loaderkw )
    model.fit(
        trainingData,
        epochs=10,
        batch_size=batchSize,
        # use_multiprocessing=True,
        # workers=2
    )
    model.save_weights(checkpointPath)
    print("Saved model weights to:", checkpointPath)

validationData = EpisodeLoader( validationFiles, **loaderkw )

nrows = 3
X, Y = validationData[0]
X, Y = validationData[0]
P = model.predict(X)
showSample(X[0:nrows], Y[0:nrows], P[0:nrows])