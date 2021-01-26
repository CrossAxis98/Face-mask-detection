from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import StrMethodFormatter
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os


def load_images():
    directories = ["with_mask", "with_improperly_wear_mask", "without_mask"]
    images = []
    labels = []

    for directory in directories:
        directory_list = os.listdir(directory)
        for file in directory_list:
            image_path = os.path.join(directory, file)
            image = load_img(image_path, target_size=(100, 100))
            image = img_to_array(image)
            image = preprocess_input(image)
            images.append(image)
            labels.append(directory)

    return images, labels


def convert_data(images, labels):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    images = np.array(images, dtype="float32")
    labels = np.array(labels)

    return images, labels

def create_model():
    model=Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(3,activation='softmax'))

    return model

def create_datagen():
    datagen = ImageDataGenerator(
    	rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
    	zoom_range=0.2,
    	shear_range=0.2,
    	horizontal_flip=True,
        fill_mode="nearest")

    return datagen

def generate_plots(history):
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    validation_loss = history.history['val_loss']

    number_of_epochs = range(len(accuracy))

    plt.plot(number_of_epochs, accuracy, 'bo', label = 'Dokładność trenowania')
    plt.plot(number_of_epochs, validation_accuracy, 'b', label = 'Dokładność walidacji')
    plt.title('Dokładność trenowania i walidacji')
    plt.legend()
    plt.figure()
    plt.plot(number_of_epochs, loss, 'bo', label='Strata trenowania')
    plt.plot(number_of_epochs, validation_loss, 'b', label='Strata walidacji')
    plt.title('Strata trenowania i walidacji')
    plt.legend()
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.show()


def main():
    images, labels = load_images()
    images, labels = convert_data(images, labels)
    datagen = create_datagen()

    (X_train, X_test, Y_train, Y_test) = train_test_split(images, labels, test_size=0.30, stratify=labels, random_state=40)

    model = create_model()
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.0001), metrics=["accuracy"])
    history = model.fit(datagen.flow(X_train, Y_train, batch_size=32), steps_per_epoch=len(X_train) // 32, epochs=20, validation_data=(X_test, Y_test), validation_steps=len(X_test) // 32)
    model.save("face_mask.model", save_format="h5")

    generate_plots(history)

if __name__ == "__main__":
    main()
