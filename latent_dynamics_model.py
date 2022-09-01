import tensorflow as tf
from tensorflow import keras

# Input: image obsevation (210, 160, 3)
# Carry over: world state
# Output: image prediction (210, 160, 3)

class DynamicsRNNCell(keras.layers.Layer):
    pass