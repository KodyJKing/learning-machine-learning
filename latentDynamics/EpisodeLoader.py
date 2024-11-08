from time import time
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras

layers = tf.keras.layers
models = tf.keras.models

class EpisodeLoader(tf.keras.utils.Sequence):
    def __init__(self, files, batchSize, windowLength, observationShape, actionShape):
        self.batchSize = batchSize
        self.windowLength = windowLength
        self.observationShape = observationShape
        self.actionShape = actionShape
        self.files = files
        self.episodes = self.loadEpisodes()
        self.windowList = self.buildWindowList()
        self.on_epoch_end()
        print("Total batches:", self.__len__())

    def loadEpisodes(self):
        print("Loading episodes...")
        start = time()

        result = []
        for path in self.files:
            with np.load(path) as file:
                observations, actions = file.values()
                result.append( (observations, actions) )
        end = time()
        print("Loaded", len(result), "episodes in", end - start, "seconds.")
        return result

    def buildWindowList(self):
        start = time()
        windowList = []
        for episodeIndex, episode in enumerate(self.episodes):
            actionCount = len(episode[1])
            for frameIndex in range(actionCount - self.windowLength + 1):
                windowList.append( (episodeIndex, frameIndex) )
        end = time()
        print("Generated", len(windowList), "windows in", end - start, "seconds.")
        return windowList
    
    def on_epoch_end(self):
        random.shuffle(self.windowList)

    def __len__(self):
        return len(self.windowList) // self.batchSize
    
    def __getitem__(self, index):
        # start = time()

        X_Observations = np.empty( (self.batchSize, self.windowLength - 1, *self.observationShape), dtype="float32" )
        X_Actions = np.empty( (self.batchSize, self.windowLength - 1, *self.actionShape ), dtype="uint8" )
        Y = np.empty( (self.batchSize, *self.observationShape), dtype="float32" )

        for i in range(self.batchSize):
            windowIndex = index * self.batchSize + i
            episodeIndex, frameIndex = self.windowList[windowIndex]
            endFrameIndex = frameIndex+self.windowLength-1
            observations, actions = self.episodes[episodeIndex]
            X_Observations[i,] = observations[ frameIndex:endFrameIndex ]
            X_Actions[i,] = actions[ frameIndex:endFrameIndex ]
            Y[i] = observations[ endFrameIndex ]

        # end = time()
        # size = (X.size * X.itemsize + Y.size * Y.itemsize) // (1000 * 1000)
        # print( "Generated", size, "MB batch in", end - start, "seconds." )

        return ( [X_Observations / 255, X_Actions], Y / 255)

