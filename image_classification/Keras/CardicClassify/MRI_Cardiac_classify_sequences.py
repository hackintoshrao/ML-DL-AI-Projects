import json
import pydicom as dicom
import pandas as pd

def load_json(filename):
  with open(filename, 'r') as f:
    return json.load(f)


def load_dataset(filename):
  dataset_info = load_json(filename)  #deserialized data into JSON dictionary again.  key
#is the ID/filepath.  value is another dictionary that contains the values of trajectory,
#view, and sequence
  #print(dataset_info.items())
  i = 0
  dataset = []
  for filepath, tags in dataset_info.items():  #.items() gives us a list of tuples.  each tuple
        #is a single key-value pair.  string is the key, dictionary is the value

    # tags is the dictionary of the three category values
    dataset.append({ "filepath" : filepath, "sequence" : tags['sequence'] })


    #dicom_data = dicom.read_file(filepath)
    #entry['image'] = dicom_data.pixel_array
    #entry['spatial_resolution'] = float(dicom_data.PixelSpacing[0])
    #entry['field_of_view'] = np.multiply(entry['spatial_resolution'], dicom_data.pixel_array.shape).max()
    # we add 3 columns from the above 3 lines to the entry array which originally only contained the 3
    #categories wer'e trying to sort
    #dataset.append(entry)

  le = preprocessing.LabelEncoder()
  df = pd.DataFrame(dataset)
  return df#dataset is a list of all the entry elements for each file.  each is a



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


def get_image(path):
    """
    Read the image from its path and convert it to RGB and return.
    """
    dicom_data = dicom.read_file("./CardiacClassification/" + path)
    img =  dicom_data.pixel_array
    return img

def process_get_data(mri_path, label):
	"""
	Randomly choose Center/Left/Right image -> Read the image and steering values -> pre-process image
	-> flip the image -> return both the original and the flipped image.
	"""
	"""
	Randomly choose Center/Left/Right image -> Read the image and steering values -> pre-process image
	-> flip the image -> return both the original and the flipped image.
	"""


	img = get_image(mri_path)

	img = image_preprocessing(img)

	return img, label

def data_generator(mri_paths, labels,  batch_size=64):
	"""
	data generator, which is used to obtain the traning an validation data in batches
	while training the model
	"""
    # Create empty arrays to contain batch of features and labels#
	X_batch = np.zeros((batch_size, 64, 64, 1), dtype=np.float32)
	y_batch = np.zeros((batch_size,4), dtype=np.float32)

	N = len(mri_paths)
	no_batches_per_epoch = (N // batch_size)
	total_count = 0
	while True:
		for j in range(batch_size):

		# choose random index in features.
			X_batch[j, :, : ,0], y_batch[j] = process_get_data(mri_paths[total_count + j], labels[total_count + j] )

		total_count = total_count + batch_size
		if total_count >= N - batchsize - 1:
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
    model = Sequential()
	# Normalize the image using lambda function in the model.
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(64, 64, 1)))
	# Convolution Layer 1.
    model.add(Conv2D(32, (5, 5), input_shape=(64, 64, 1), activation='elu'))

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
    model.add(Dense(4, activation='sigmoid'))

    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model


# Load the json dataset.
dataset  = load_dataset('./CardiacClassification/cardiac.json')

img_paths = dataset['filepath'].values
labels = dataset['sequence'].values

encoder =  LabelEncoder()
labels = encoder.fit_transform(labels)

labels = pd.get_dummies(labels).values



img_paths, labels = sklearn.utils.shuffle(img_paths, labels)
 # 80% of the data is used for training and 20% for validation.
training_split = 0.8

training_set_num = int(len(labels) * training_split)

# separate training and validation data and create different generators for them.

X_train, X_test, y_train, y_test = train_test_split(img_paths, labels, test_size=0.2, random_state=42)

print("training len: ", len(X_train))
print("validation len: ", len(X_test))

BATCH_SIZE = 32

# obtain generators for training and validation data.
training_data_generator = data_generator(X_train, y_train, batch_size=BATCH_SIZE)
validation_data_generator = data_generator(X_test, y_test, batch_size=BATCH_SIZE)

# fetch the model.
model = get_model()

 # extracts around 22000 samples in each epoch from the generator.
samples_per_epoch = (4000 // BATCH_SIZE) * BATCH_SIZE


model.fit_generator(training_data_generator, validation_data=validation_data_generator,samples_per_epoch=samples_per_epoch, nb_epoch=3, nb_val_samples=600)

print("Saving model.")

model.save('model_udacity_6.h5')  # always save your weights after training or during training
