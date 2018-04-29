import json
import pydicom as dicom
import pandas as pd
import cv2

def load_json(filename):
  with open(filename, 'r') as f:
    return json.load(f)


def load_dataset(filename):
  dataset_info = load_json(filename)  #deserialized data into JSON dictionary again.  key
#is the ID/filepath.  value is another dictionary that contains the values of trajectory,
#view, and sequence
  #print(dataset_info.items())
  i = 0
  files = []
  field_of_views = []
  for filepath, tags in dataset_info.items():

    # tags is the dictionary of the three category values
    files.append("./CardiacClassification/" + filepath)
    field_of_views.append(tags['field_of_view'] / 100)


  return files, field_of_views



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


def get_image(path):
    """
    Read the image from its path and convert it to RGB and return.
    """
    img = cv2.imread(path)
    return img

def process_get_data(img_path, field_of_view):
	"""
    Read the image and process it
	"""


	img = get_image(img_path)

	img = image_preprocessing(img)

	return img, field_of_view

def data_generator(mri_paths, field_of_views,  batch_size=64):
	"""
	data generator, which is used to obtain the traning an validation data in batches
	while training the model
	"""
    # Create empty arrays to contain batch of features and labels#
	X_batch = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
	y_batch = np.zeros((batch_size,1), dtype=np.float32)

	N = len(mri_paths)
	no_batches_per_epoch = (N // batch_size)
	total_count = 0
	while True:
		for j in range(batch_size):

		# choose random index in features.
			X_batch[j], y_batch[j] = process_get_data(mri_paths[total_count + j], field_of_views[total_count + j] )

		total_count = total_count + batch_size
		if total_count >= N - batch_size - 1:
            # reset the index, this allows iterating though the dataset again.
			total_count = 0

		yield X_batch, y_batch


def image_preprocessing(image):
	"""
	Crop the image, resize and normalize.
	"""
	image = cv2.resize(image, (64, 64))
	image = image.astype(np.float32)
	#image = image/255.0 - 0.5
	return image


def get_model():
    """
    Obtain the convolutional neural network model
    The model contains 3 convolutional layer and 2 fully connected layer.
    ELU is used as activation function.
    """
    """
	Obtain the convolutional neural network model
	The model contains 3 convolutional layer and 2 fully connected layer.
	ELU is used as activation function.
	"""
    model = Sequential()
    # Normalize the image using lambda function in the model.
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64, 64, 3)))

    # Convolution Layer 1.
    model.add(Conv2D(32, (5, 5), input_shape=(64, 64, 3), activation='elu'))
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


    # Finally a single output, since this is a regression problem
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

    return model



# Load the json dataset.
img_paths, field_of_views   = load_dataset("CardiacClassification/fov_test.json")

print(field_of_views[10:20])

img_paths, field_of_views = sklearn.utils.shuffle(img_paths, field_of_views)


# separate training and validation data and create different generators for them.

X_train, X_test, y_train, y_test = train_test_split(img_paths, field_of_views, test_size=0.2, random_state=42)

print("training len: ", len(X_train))
print("validation len: ", len(X_test))

BATCH_SIZE = 32

# obtain generators for training and validation data.
training_data_generator = data_generator(X_train, y_train, batch_size=BATCH_SIZE)
validation_data_generator = data_generator(X_test, y_test, batch_size=BATCH_SIZE)

# fetch the model.
model = get_model()

 # extracts around 22000 samples in each epoch from the generator.
samples_per_epoch = (len(X_train) // BATCH_SIZE) * BATCH_SIZE


model.fit_generator(training_data_generator, samples_per_epoch=samples_per_epoch, nb_epoch=10)

X_test_images = []
values = []
for i in range(10):
    X_test_image, value = process_get_data(X_test[i], y_test[i])
    X_test_images.append(X_test_image)
    values.append(value)

for i in range(10):

    x = X_test_images[i]
    print(x[None, :, :, :].shape)
    results = model.predict(x[None, :, :, :],batch_size=1)[0] * 100
    print("Filename: ",X_test[i])
    print("results: ", results[0] )

    print("Acutal value: ", values[i] * 100)
# print("Saving model.")
#
# model.save('model.h5')  # always save your weights after training or during training
