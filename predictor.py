import os
import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers
from keras.models import load_model

from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt

height, width, channels = 224, 224, 3
model = load_model('model.h5')

#Take two images as input and resize the images
os.chdir('extra_images/')
raw_image = Image.open('test_barca_1.jpg')
compressed_image = raw_image.resize((width, height), Image.ANTIALIAS)
compressed_image.save('test_barca_resized_1.jpg', quality = 95) 

raw_image = Image.open('test_chelsea_1.jpg')
compressed_image = raw_image.resize((width, height), Image.ANTIALIAS)
compressed_image.save('test_chelsea_resized_1.jpg', quality = 95) 

#Input the resized images to the model
temp = []
temp.append(io.imread('test_barca_resized_1.jpg'))
temp.append(io.imread('test_chelsea_resized_1.jpg'))
images = np.array(temp)
predictions = model.predict(images)
predictions_list = list(predictions)
modified_predictions = list(np.argmax(predictions, axis = 1))
for each in modified_predictions:
    if each == 0:
        print('Barca')
    elif each == 1:
        print('Real Madrid')
    elif each == 2:
        print('Manchester United')
    elif each == 3:
        print('Dortmund')
    elif each == 4:
        print('Inter Milan')
    else:
        print('Chelsea')


