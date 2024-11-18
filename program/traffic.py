import cv2
import numpy
import os
import sys
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

EPOCHS = 10
IMAGE_WIDTH = 30
IMAGE_HEIGHT = 30
N = 43
TEST_SIZE = 0.40


def load_data(directory):
    '''
        Load image data from directory. Assume 'directory' has one directory named after each category, nubered 0 through N - 1. Inside each category directory will be some number of image files. Return tuple '(images, labels)'. 'images' should be a list of all the images in the directory, where each image is formatted as numpy ndarray with dimensions IMAGE_WIDTH x IMAGE_HEIGHT x 3. 'labels' should be a list of integer labels, representing the categories for each of the corresponding 'images'.
    '''
    images = []
    labels = []
    for i in range(N):
        folder = os.path.join(directory, str(i))
        if os.path.exists(folder):
            if not os.listdir(folder):
                print(f"Diretório vazio: {folder}")
                continue
            for file in os.listdir(folder):
                filepath = os.path.join(folder, file)
                img = cv2.imread(filepath)
                if img is not None:
                    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                    images.append(img)
                    labels.append(i)
                else:
                    print(f"Erro ao carregar imagem: {filepath}")
    return numpy.array(images), numpy.array(labels)


def get_model():
    '''
        Returns a compiled convolutional neural network model. Assume that the 'input_shape' of the first layer is '(IMAGE_WIDTH, IMAGE_HEIGHT, 3)'. The output layer should have 'N' units, one for each category.
    '''
    model = Sequential([
        Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3)),  # Ajuste para (30, 30, 3)
        Conv2D(32, (3, 3), activation='relu'),
        Flatten(),
        Dense(N, activation='softmax')  # Supondo 43 classes
    ])
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    print(model.summary()) 
    return model

if __name__ == '__main__':

    # check command-line arguments
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        sys.exit('Usage: python traffic.py directory [model.h5]')

    # get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        numpy.array(images), 
        to_categorical(labels, num_classes=N), 
        test_size = TEST_SIZE
    )

    # get a compiled neural network
    model = get_model()

    # fit model on training data, saving the history object
    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))

    # evaluate neural network performance
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Testes perdidos: {loss:.4f}, Testes acertados: {accuracy:.4f}")

    # Plot the training loss and accuracy
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # save model to file
    if len(sys.argv) == 3:
        # model.save(sys.argv[2])
        model.save("model.keras")
        print(f"Modelo salvo para {sys.argv[2]}")
