import pandas as pd
import csv

def load_json(filename):
  with open(filename, 'r') as f:
    return json.load(f)


def load_dataset(filename):
    images = []
    labels = []
    with open(filename) as f:
        cr = csv.reader(f)
        skip=next(cr)  #skip the first row of keys "a,b,c,d"
        for l in cr:
            labels.append(l.pop())
            images.append(l)


    return np.array(images), np.array(labels)







import cv2
import numpy as np
import random

# Using keras to build and train the model.
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, Dropout, ELU, Conv2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


def get_image(pixels):
    """
    Image pixels are already in memory, just return
    """

    return pixels

def process_get_data(pixels, label):
	"""
	Randomly choose Center/Left/Right image -> Read the image and steering values -> pre-process image
	-> flip the image -> return both the original and the flipped image.
	"""
	"""
	Randomly choose Center/Left/Right image -> Read the image and steering values -> pre-process image
	-> flip the image -> return both the original and the flipped image.
	"""


	img = get_image(pixels)

	img = image_preprocessing(img)

	return img, label

def data_generator(pixels, labels,  batch_size=64):
	"""
	data generator, which is used to obtain the traning an validation data in batches
	while training the model
	"""
    # Create empty arrays to contain batch of features and labels#
	X_batch = np.zeros((batch_size, 16, 16, 1), dtype=np.float32)
	y_batch = np.zeros((batch_size,74), dtype=np.float32)

	N = len(pixels)
	no_batches_per_epoch = (N // batch_size)
	total_count = 0
	while True:
		for j in range(batch_size):

		# choose random index in features.
			X_batch[j, :, : ,0], y_batch[j] = process_get_data(pixels[total_count + j], labels[total_count + j] )

		total_count = total_count + batch_size
		if total_count >= N - batch_size - 1:
            # reset the index, this allows iterating though the dataset again.
			total_count = 0

		yield X_batch, y_batch


def image_preprocessing(image):
	"""
	Crop the image, resize and normalize.
	"""
	image = np.reshape(image, (-1, 16))
	image = image.astype(np.float32)
	#image = image/255.0 - 0.5
	return image


def get_model():
    """
    Obtain the convolutional neural network model
    The model contains 3 convolutional layer and 2 fully connected layer.
    ELU is used as activation function.
    """
    model = Sequential()
	# Normalize the image using lambda function in the model.
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(16, 16, 1)))
	# Convolution Layer 1.
    model.add(Conv2D(32, (5, 5), input_shape=(16, 16, 1), activation='elu'))

	#  Convolution Layer 2.
    model.add(Conv2D(16, (3, 3), activation='elu'))

    model.add(Dropout(.25))
    model.add(MaxPooling2D((2, 2)))

	# Convolution Layer 3.
    model.add(Conv2D(8, (3, 3), activation='elu'))

    model.add(Dropout(.25))

	# Flatten the output

    model.add(Flatten())

	# layer 4
    model.add(Dense(1024))
    model.add(Dropout(.3))
    model.add(ELU())

	# layer 5
    model.add(Dense(512))
    model.add(Dropout(.2))
    model.add(ELU())

    # layer 6
    model.add(Dense(256))
    model.add(ELU())

    # layer 7
    model.add(Dense(128))
    model.add(ELU())


	# Finally a single output.
    model.add(Dense(74, activation='sigmoid'))

    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model


# Load the dataset and get dummies for the labels.
X_train, y_train = load_dataset("/data/pixel_classification/REF.csv")

encoder =  LabelEncoder()
labels = encoder.fit_transform(y_train)

Y_train = pd.get_dummies(labels).values

# Load the json dataset.
X_test, y_test = load_dataset("/data/pixel_classification/ref_1.csv")

encoder =  LabelEncoder()
labels = encoder.fit_transform(y_test)

Y_test = pd.get_dummies(labels).values

#X_train, X_test,  y_train, y_test = train_test_split(X_test, Y_test, test_size = 0.33, random_state = 42)
print("train shape: ", X_train.shape)
print("test shape: ", y_test.shape)





BATCH_SIZE = 32

# obtain generators for training and validation data.
training_data_generator = data_generator(X_train, y_train, batch_size=BATCH_SIZE)
validation_data_generator = data_generator(X_test, y_test, batch_size=BATCH_SIZE)

# fetch the model.
model = get_model()

 # extracts around 22000 samples in each epoch from the generator.
samples_per_epoch = (len(X_train) // BATCH_SIZE) * BATCH_SIZE


model.fit_generator(training_data_generator, validation_data=validation_data_generator,samples_per_epoch=samples_per_epoch, nb_epoch=3, nb_val_samples=len(X_test))

print("Saving model.")

model.save('/output/model.h5')  # always save your weights after training or during training
