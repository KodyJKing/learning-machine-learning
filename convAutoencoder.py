import tensorflow as tf
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import modelString, stringHash

layers = tf.keras.layers

settings = {
    "version": "0.0.3",
    "compile_args": {
        # "loss": "binary_crossentropy",
        "loss": "mean_squared_error",
        "optimizer": "adam",
    },
    "fit_args": {
        "epochs": 50,
        "batch_size": 128
    }
}
baseCheckpointPath = './checkpoints/convAutoencoder'

# dataset = tf.keras.datasets.mnist
dataset = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
    layers.Input(shape=(28, 28, 1)),

    # Encoder
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"), layers.MaxPooling2D((2, 2), padding="same"),
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"), layers.MaxPooling2D((2, 2), padding="same"),

    layers.Flatten(input_shape=(7, 7, 32)),
    layers.Dense(10, activation="relu"),

    # Decoder

    layers.Dense(7*7*32, activation="relu"),
    layers.Reshape((7, 7, 32), input_shape=(7*7*32,)),

    layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"),
    layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"),
    layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same"),
])
model.compile(metrics=['cosine_similarity'], **settings["compile_args"])
modelStr = modelString(model)
settingsAndModel = json.dumps(settings) + "\n" + modelStr
print("\n")
print(settingsAndModel)
print("\n")
settingsHash = stringHash(settingsAndModel)
checkpointPath = baseCheckpointPath + "_" + str(settingsHash)

try:
    print("\nChecking for model at:", checkpointPath)
    model.load_weights(checkpointPath)
    print("\nLoaded previous model.\n")
except:
    print("\nMissing checkpoint. Training...\n")
    model.fit(train_images, train_images, **settings["fit_args"])
    model.save_weights(checkpointPath)

# predictions = model.predict(test_images)

def displayPredictions():
    n = 10
    indices = np.random.randint(len(test_images), size=n)
    # print(indices)

    imagesIn = test_images[indices, :]
    # imagesOut = predictions[indices, :]
    imagesOut = model.predict(imagesIn)

    rows = 2
    cols = n

    plt.figure(figsize=(n*1.5, 2*1.5))
    for i in range(n):
        plt.subplot(rows, cols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imagesIn[i], cmap=plt.cm.binary)

        plt.subplot(rows, cols, i + 1 + n)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imagesOut[i], cmap=plt.cm.binary)
    plt.show()

displayPredictions()