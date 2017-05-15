import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os, sys
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Conv2D, Dense, Flatten
from keras_tqdm import TQDMNotebookCallback
from keras.optimizers import SGD, Adam, Nadam, RMSprop

### Start Here with pre-loaded images
X = np.load('mdata_X.npy')
y = np.load('mdata_y.npy')

### base VGG-16 model
model_base = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))

### Add additional custom-designed top-layers
x = model_base.layers[-1].output
x = Flatten()(x)
x = Dense(512, init='orthogonal', activation='relu')(x)
x = Dense(4, init='orthogonal', activation='softmax')(x)

### Train, Validate, and Test the Model
model2 = Model(model_base.inputs, x)
model2.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(lr=0.0001))
model2.fit(X, y, batch_size=64, epochs=10, validation_split=0.2, 
	verbose=0,callbacks=[TQDMNotebookCallback(leave_inner=True)])
y_pred = model2.predict(X)