import csv
import cv2
import numpy as np
import sklearn
import random

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the headers
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

print("***** len(train_samples): "+str(len(train_samples)))

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                left_name = 'data/IMG/'+batch_sample[1].split('/')[-1]
                right_name = 'data/IMG/'+batch_sample[2].split('/')[-1]
                center_image = cv2.imread(center_name)
                left_image = cv2.imread(left_name)
                right_image = cv2.imread(right_name)
                center_angle = float(batch_sample[3])
                left_angle = float(batch_sample[3]) + 0.25
                right_angle = float(batch_sample[3]) - 0.25

                # Convert BGR to HSV
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2HSV)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2HSV)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2HSV)

                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)
                # Augment Images with flipped ones
                images.append(cv2.flip(center_image,1))
                images.append(cv2.flip(left_image,1))
                images.append(cv2.flip(right_image,1))
                angles.append(center_angle*-1.0)
                angles.append(left_angle*-1.0)
                angles.append(right_angle*-1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D 

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((65,25),(0,0))))
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

import matplotlib as mpl
mpl.use('Agg') # only for AWS instance without $DISPLAY
import matplotlib.pyplot as plt

batch_size=32

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=(len(train_samples)*6//batch_size)*batch_size,
                                     validation_data=validation_generator,
                                     nb_val_samples=(len(validation_samples)*6//batch_size)*batch_size,
                                     nb_epoch=4,
                                     verbose=1)

model.save('model.h5')

# plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')

plt.savefig("metrics.png")
#plt.show()
