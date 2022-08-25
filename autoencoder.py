import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
# plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(28 * 28, activation='sigmoid'),
    tf.keras.layers.Reshape((28, 28), input_shape=(28 * 28,)),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['cosine_similarity'])

checkpointPath = './checkpoints/autoencoder_epoch_20_bincross'
# checkpointPath = './checkpoints/autoencoder_epoch_10'
def train():
    model.fit(train_images, train_images, epochs=20)
    model.save_weights(checkpointPath)

try:
    model.load_weights(checkpointPath)
    print("\nLoaded previous model.\n")
except:
    print("\nMissing checkpoint. Training...\n")
    train()

predictions = model.predict(test_images)

def displayPredictions():
    n = 10
    indices = np.random.randint(len(test_images), size=n)
    print(indices)

    imagesIn = test_images[indices, :]
    imagesOut = predictions[indices, :]

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