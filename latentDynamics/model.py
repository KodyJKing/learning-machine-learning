import numpy as np
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
# from tensorflow.python.keras import layers
# from tensorflow.python.keras import models
layers = tf.keras.layers
models = tf.keras.models

def getPadding(observationShape):
    bottleneckShape = np.ceil( np.array( np.ceil( np.array(np.ceil( np.array(observationShape) / 2 )) / 2 ) ) / 2 )
    paddedShape = 8 * bottleneckShape
    padding = (paddedShape - np.array(observationShape)) // 2
    padding = ( int(padding[0]), int(padding[1]) )
    bottleneckShape = (int(bottleneckShape[0]), int(bottleneckShape[1]), observationShape[2])
    return padding, bottleneckShape

def frameEncoder(shape, padding, verbose=False):
    model = models.Sequential([
        layers.ZeroPadding2D(padding, input_shape=shape),
        layers.Conv2D( 16, (3, 3), activation="relu", padding="same" ),
        layers.MaxPooling2D(),
        layers.Conv2D( 16, (3, 3), activation="relu", padding="same" ),
        layers.MaxPooling2D(),
        layers.Conv2D( 16, (3, 3), activation="relu", padding="same" ),
        layers.MaxPooling2D(),
        layers.Flatten(),
    ], name="Frame-Encoder")

    if verbose:
        print()
        model.summary()
    
    return model

def frameDecoder(units, bottleneckShape, cropping, verbose=False):
    denseUnits = bottleneckShape[0] * bottleneckShape[1] * bottleneckShape[2]
    model = models.Sequential([
        layers.Dense(denseUnits, input_shape=(units,)),
        layers.Reshape(bottleneckShape),
        layers.Conv2DTranspose(16, (3, 3), strides=2, activation="relu", padding="same"),
        layers.Conv2DTranspose(16, (3, 3), strides=2, activation="relu", padding="same"),
        layers.Conv2DTranspose(16, (3, 3), strides=2, activation="relu", padding="same"),
        layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same"),
        layers.Cropping2D(cropping),
    ], name="Frame-Decoder")
    
    if verbose:
        print()
        model.summary()
    
    return model

def latentDynamicsModel(windowLength: int, observationShape: Tuple["int"], verbose=False):
    input_shape = (windowLength, *observationShape)

    stateUnits = 300

    padding, bottleneckShape = getPadding(observationShape)
    encoder = frameEncoder(observationShape, padding, verbose=verbose)
    decoder = frameDecoder(stateUnits, bottleneckShape, padding, verbose=verbose)

    model = models.Sequential([
        layers.TimeDistributed(encoder, input_shape=input_shape),
        # layers.SimpleRNN(stateUnits, activation="relu", return_sequences=True),
        # layers.TimeDistributed(decoder)
        layers.SimpleRNN(stateUnits, activation="relu"),
        decoder,
    ], name="Latent-Dynamics")
    
    if verbose:
        print()
        model.summary()

    return model

# model = latentDynamicsModel(
#     windowLength = 5,
#     observationShape = (210, 160, 3)
# )