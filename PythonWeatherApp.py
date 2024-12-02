Q1. Train a Deep Neural network model (CNN) using animal dataset and perform the following operations:

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
classes = ['airplane', 'automobile', 'bird', 'cat', 
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# Check the shapes of the dataset
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test labels shape: {y_test.shape}")
plt.figure(figsize=(5, 3))
plt.imshow(X_train[0])
plt.title(f"Class: {classes[y_train[0][0]]}")
plt.axis("off")
plt.show()
X_train = X_train / 255.0
X_test = X_test / 255.0
cnn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(classes), activation='softmax')
])
cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
cnn.summary()
print("Training CNN model...")
cnn.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
print("Evaluating CNN model...")
cnn_eval = cnn.evaluate(X_test, y_test)
print(f"CNN Accuracy: {cnn_eval[1]*100:.2f}%")
print("Making predictions...")
y_pred_cnn = cnn.predict(X_test)
y_pred_classes_cnn = [np.argmax(element) for element in y_pred_cnn]
# Plotting a sample prediction
def plot_sample_with_prediction(image, true_label, pred_label):
    plt.figure(figsize=(5, 3))
    plt.imshow(image)
    plt.title(f"True: {classes[true_label]}, Pred: {classes[pred_label]}")
    plt.axis("off")
    plt.show()
plot_sample_with_prediction(X_test[0], y_test[0][0], y_pred_classes_cnn[0])

Q2.Plot activation functions along with their derivatives
import numpy as np
import matplotlib.pyplot as plt

# Define activation functions and their derivatives
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def prelu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def prelu_derivative(x, alpha=0.1):
    return np.where(x > 0, 1, alpha)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.1):
    return np.where(x > 0, 1, alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, elu(x, alpha) + alpha)

def selu(x, lambda_=1.0507, alpha=1.67326):
    return lambda_ * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def selu_derivative(x, lambda_=1.0507, alpha=1.67326):
    return lambda_ * np.where(x > 0, 1, selu(x, lambda_, alpha) + alpha)

# Range of x values for plotting
x = np.linspace(-3, 3, 500)

# Plot each activation function and its derivative
activation_functions = {
    "Tanh": (tanh, tanh_derivative),
    "Sigmoid": (sigmoid, sigmoid_derivative),
    "ReLU": (relu, relu_derivative),
    "PReLU": (prelu, prelu_derivative),
    "Leaky ReLU": (leaky_relu, leaky_relu_derivative),
    "ELU": (elu, elu_derivative),
    "SELU": (selu, selu_derivative)
}

plt.figure(figsize=(14, 16))
for i, (name, (func, derivative)) in enumerate(activation_functions.items(), 1):
    plt.subplot(4, 2, i)
    plt.plot(x, func(x), label=f"{name}", color='blue')
    plt.plot(x, derivative(x), label=f"{name} Derivative", color='orange', linestyle='--')
    plt.title(f"{name} and its Derivative")
    plt.legend()
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


Q3. Build an LSTM model to demonstrate sequence learning using amazon reviews dataset. Perform the following operations:

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

In [20]:
# Load the IMDB dataset
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

In [22]:
# Pad the sequences
x_train = pad_sequences(x_train, maxlen=500) 

x_test = pad_sequences(x_test, maxlen=500)

In [23]:
# Build the LSTM model
model = Sequential([
    Embedding(10000, 64),
    LSTM(64),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

In [24]:
# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

In [32]:
history = model.fit(x_train, y_train, epochs=5,batch_size=32, validation_data=(x_test, y_test))
loss, accuracy = model.evaluate(x_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)
import matplotlib.pyplot as plt
# Plot training history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


Q4. Autoencoder architecture training using handwritten character image dataset- visualize compressed, reconstructed images, calculate loss for encoder and decoder.
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Reshape, Conv2DTranspose, UpSampling2D
(x_train, _), (x_test, _) = mnist.load_data()

In [42]:
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train  = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))

In [54]:
# Define the encoder model
encoder = Sequential([
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(8, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    Flatten(),
    Dense(32, activation='relu')
])

In [55]:
decoder = Sequential([
    Dense(7 * 7 * 8, activation='relu'),
    Reshape((7, 7, 8)),
    Conv2DTranspose(8, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2DTranspose(16, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(1, (3, 3), activation='sigmoid', padding='same')
])

In [56]:
autoencoder = Sequential([encoder, decoder])

In [57]:
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

In [58]:
autoencoder.fit(x_train, x_train, epochs=10, batch_size=128, validation_data=(x_test, x_test))
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
# import matplotlib.pyplot as plt
n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
# Calculate reconstruction loss
loss = autoencoder.evaluate(x_test, x_test)
print(f"Reconstruction loss: {loss}")
Q5. Build an inceptionv3 model to classify various kinds of bottles (plastic bottle, wine bottle, glass botter, water bottles etc), load the bottle images dataset, train inception v3 model, calculate loss, calculate training and validation accuracies & confusion matrix, also perform prediction on test image.
In [17]:
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

In [18]:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

In [19]:
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0 

In [20]:
x_train = tf.image.resize(x_train, (75, 75))
x_test = tf.image.resize(x_test, (75, 75))

In [21]:
#inception v3
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(75, 75, 3))
base_model.trainable = False

In [22]:
x = base_model.output
x = GlobalAveragePooling2D()(x)

In [23]:
predictions = Dense(10, activation='softmax')(x)

In [24]:
model = Model(inputs=base_model.input, outputs=predictions)

In [25]:
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

In [26]:
# Train the model
history = model.fit(x_train, y_train,epochs=10,batch_size=32, validation_data=(x_test, y_test))
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
# Plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='upper right')
plt.show()
# Plot training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc='lower right')
plt.show()
# Calculate confusion matrix
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
confusion_mtx = confusion_matrix(y_test, y_pred_classes)
print(confusion_mtx)
# Predict on a test image
index = np.random.randint(0, len(x_test))
test_image = x_test[index]
predicted_class = model.predict(tf.reshape(test_image, (1, 75, 75, 3)))
predicted_class = np.argmax(predicted_class)
print("Predicted class:", predicted_class)
print("Actual class:", y_test[index])

Predicted class: 3
Actual class: [7]

In [46]:
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.scatter(y_test[index], predicted_class)
plt.xlabel('y_test[index]')
plt.ylabel('predicted_class')
plt.title('Actual vs Predicted Values')
plt.grid(True)
plt.show()





