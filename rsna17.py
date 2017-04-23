
# coding: utf-8

# In[1]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os, sys
import scipy.ndimage
import pickle
from multiprocessing import Pool
from skimage import measure, morphology
import random
import pylab
import skimage.transform

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import imshow, figure, hist, plot, scatter, colorbar
get_ipython().magic('matplotlib inline')

np.random.seed(314159)
random.seed(314159)



# In[2]:

#Continue here if preprocessing data from scratch
DATA_PATH = '/media/sohn/Storage/data_tcia_mammo/CBIS-DDSM/'


# In[6]:

#Getting names of files in DATA_PATH
def get_file_names(path):
    x=[]
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            x.append(os.path.join(root, name))
    return x


# In[7]:

#Read in the labels & file names
file_names = get_file_names(DATA_PATH)
df = pd.read_csv(DATA_PATH + '../mass_case_description_train_set.csv')
df2 = pd.read_csv(DATA_PATH + '../calc_case_description_train_set.csv')


# In[8]:

#Convert file names to a dataframe for Mass-Training Set
df["file_name"] = [ None for x in range(len(df)) ]

for n in range(len(df)):
    key = '/Mass-Training_' + df["patient_id"][n] +'_'+ df["side"][n] +'_'+ df["view"][n] + '/'
    for fn in file_names:
        if key in fn:
            df.loc[n, "file_name"] = fn
            break


# In[9]:

#Convert file names to a dataframe for Calcium-Training Set
df2["file_name"] = [ None for x in range(len(df2)) ]
for n in range(len(df2)):
    key = 'Calc-Training_' + df2["patient_id"][n] +'_'+ df2["side"][n] +'_'+ df2["view"][n] + '/'
    for fn in file_names:
        if key in fn:
            df2.loc[n, "file_name"] = fn
            break


# In[10]:

#Experiment: Check if all filenames are accounted for. 
for n in range(len(df)):
    if  df.loc[n, "file_name"] is None:
        print(n)


# In[11]:

#Drop labels/files that are not found in both label & folder name: several in Calc-Training
k=[]
for n in range(len(df2)):
    if  df2.loc[n, "file_name"] is None:
        k.append(n)
df2=df2.drop(k)
print(k)


# In[12]:

#Create full file names list
file_names = list(df["file_name"]) + list(df2["file_name"])
breast_density = list(df["breast_density"]) + list(df2["breast_density"])
len(file_names)
len(breast_density)


# In[13]:

#Experiment
file_names[1227]


# In[14]:

#Exclude mass 732, 956, 1227 (ID 1057 R MLO, ID 1371 R CC, ID 1757 R CC) due to corrupt dicom
file_names = file_names[:732] + file_names[733:956] + file_names[957:1227] + file_names[1228:]
breast_density = breast_density[:732] + breast_density[733:956] + breast_density[957:1227] + breast_density[1228:]
# TODO same for density


# In[15]:

#Experiment
d = dicom.read_file(file_names[955]) #732 gives error
image = d.pixel_array
image_small = skimage.transform.resize(image, (224,224), preserve_range=True)
imshow(image, cmap='gray')
plt.show()


# In[36]:

#WARNING Take Long Time: Create an image file of all images from file_names
images = []
for fn in file_names:
    d = dicom.read_file(fn)
    image = d.pixel_array
    image_small = skimage.transform.resize(image, (224,224), preserve_range=True) #specify image resize
    images.append(image_small)
len(images)


# In[39]:

#Form X and Y numpy arrays from loaded DICOM images
X = np.stack(images)[...,None]
X = np.repeat(X, 3, axis=-1) #because ImageNet is RGB
X = (X - np.mean(X)) / np.std(X)

y = np.asarray(breast_density)-1
y = y[ :X.shape[0] ]


# In[40]:

#Save loaded images to numpy files
np.save('mdata_X.npy', X)
np.save('mdata_y.npy', y)


# In[116]:

#Experiment
X[1,1:3,1:5]


# In[121]:

#Experiment
y[1:10]


# In[3]:

### Start Here if using previously saved DICOM images
X = np.load('mdata_X.npy')
y = np.load('mdata_y.npy')


# In[17]:

#Experiment
len(X)


# In[18]:

#Experiment
len(y)


# In[4]:

#Import Keras Libraries
from keras.applications import VGG16, Xception, ResNet50
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Conv2D, Dense, Flatten
from keras_tqdm import TQDMNotebookCallback
from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.preprocessing.image import ImageDataGenerator


# In[5]:

#Import Model from Keras

model_base = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))
#model_base = Xception(include_top=False, weights='imagenet', input_shape=(224,224,3))
#model_base = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))

#model_base.summary()


# In[6]:

#Add additional custom-designed top-layers
x = model_base.layers[-1].output
#x = Flatten()(x)
x = GlobalAveragePooling2D()(x)
x = Dense(512, kernel_initializer='orthogonal', activation='relu')(x)
x = Dense(4, kernel_initializer='orthogonal', activation='softmax')(x)

model2 = Model(model_base.inputs, x)
#print(model2.summary())


# In[72]:

print(model2.summary())


# In[7]:

#Specify Model Hyperparameters

#model2.compile(loss='squared_hinge', metrics=['accuracy'], optimizer=optimizer)
optimizer = Adam(lr=0.0001)
model2.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)


# In[27]:

#Train the Model
model2.fit(
    X, 
    y, 
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    verbose=0, 
    callbacks=[TQDMNotebookCallback(leave_inner=True)])


# In[8]:

#X_train, y_train = X,y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# kf = KFold(n_splits=5)
# train, test = next( kf.split(X) )
# X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=15.0,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(X_train)

#model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
#                    steps_per_epoch=len(X_train), epochs=epochs)

dgf = datagen.flow(X_train, y_train, batch_size=32)

model2.fit_generator(dgf,
    epochs=10,
    steps_per_epoch=1000,
    verbose=0, 
    validation_data=(X_test, y_test),
    callbacks=[TQDMNotebookCallback(leave_inner=True)])


# In[75]:

model2.fit_generator(dgf,
    epochs=3,
    steps_per_epoch=100,
    verbose=0, 
    validation_data=(X_test, y_test),
    callbacks=[TQDMNotebookCallback(leave_inner=True)])


# In[9]:

#Save the Model
model2.save('br_vgg_aug.h5')


# In[ ]:

#Load Existing Model - Start from Here if available
model2 = model.load('br_vgg_aug.h5')


# In[129]:

shift = 0.2
train_datagen = ImageDataGenerator(horizontal_flip=True, width_shift_range=shift, height_shift_range=shift)


# In[130]:

test_datagen = ImageDataGenerator()


# In[ ]:

train_generator = train_datagen.flow(X,y)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')


# In[10]:

#Make Predictions based on trained model
y_pred = model2.predict(X_test)


# In[83]:

y_pred[4]


# In[67]:

np.round(y_pred[4].dot(np.arange((4))))


# In[11]:

#Assign variables for error analysis.
#y_pred_assigned = [ np.argmax(y_pred[n]) for n in range(len(y_pred)) ]
y_pred_assigned = [ np.round(y_pred[n].dot(np.arange((4)))) for n in range(len(y_pred)) ]
y_pred_conf = [ np.max(y_pred[n]) for n in range(len(y_pred)) ]


# In[15]:

#Experiment: Take a look at y vs ypred assignment
list(zip(y_test, y_pred_assigned, y_pred_conf))[:-50]


# In[12]:

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          do_cell_labels=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm / np.amax(cm), interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    if do_cell_labels:
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[13]:

#Confusion Matrix for Later 20% of Data
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_assigned)
cm


# In[14]:

plot_confusion_matrix(cm, ['A', 'B', 'C', 'D'], normalize=False, do_cell_labels=True)


# In[69]:

cm = confusion_matrix(y_test, y_pred_assigned)
plot_confusion_matrix(cm, ['A', 'B', 'C', 'D'], normalize=False, do_cell_labels=True)


# In[58]:

#Experiment
imshow(X[1,:,:,0], cmap='gray')


# In[90]:




# In[96]:

# Most confident, Correct, Density B
idxs = np.argsort(y_pred[:,1])[::-1]
for n in range(4):
    idx = idxs[n]
    if y_test[idx] == 1:
        print("Radiologist:", y_test[idx], "; Algorithm: ", y_pred_assigned[idx], "(", y_pred[idx,1], ")")
        imshow(X_test[idx,:,:,0], cmap='gray')
        plt.show()


# In[107]:

# Most confident, Incorrect, Predicted Density D
idxs = np.argsort(y_pred[:,3])[::-1]
for n in range(2000):
    idx = idxs[n]
    if y_test[idx] == 0:
        print("Radiologist:", y_test[idx], "; Algorithm: ", y_pred_assigned[idx], "(", y_pred[idx,3], ")")
        imshow(X_test[idx,:,:,0], cmap='gray')
        plt.show()


# In[88]:

# Random 10
idxs = np.random.permutation(len(y_test))
for n in range(10):
    idx = idxs[n]
    print("Radiologist:", y_test[idx], "; Algorithm: ", yassignedt[idx], "(", yconft[idx], ")")
    imshow(Xt[idx,:,:,0], cmap='gray')
    plt.show()


# In[142]:

# Most confident, Correct, Density C
idxs = np.argsort(ypredt[:,2])[::-1]
for n in range(10):
    idx = idxs[n]
    if yt[idx] == 2:
        print("Radiologist:", yt[idx], "; Algorithm: ", yassignedt[idx], "(", yconft[idx], ")")
        imshow(Xt[idx,:,:,0], cmap='gray')
        plt.show()


# In[151]:

# Most confident, Incorrect, Density C
idxs = np.argsort(ypredt[:,2])[::-1]
for n in range(50):
    idx = idxs[n]
    if yt[idx] != 2:
        print("Radiologist:", yt[idx], "; Algorithm: ", yassignedt[idx], "(", yconft[idx], ")")
        imshow(Xt[idx,:,:,0], cmap='gray')
        plt.show()


# In[149]:

# Most confident, Incorrect, Density D
idxs = np.argsort(ypredt[:,3])[::-1]
for n in range(50):
    idx = idxs[n]
    if yt[idx] != int(yassignedt[idx]):
        print("Radiologist:", yt[idx], "; Algorithm: ", yassignedt[idx], "(", yconft[idx], ")")
        imshow(Xt[idx,:,:,0], cmap='gray')
        plt.show()


# In[150]:




# In[70]:

for n in range(100,120):
    idx = n
    print(idx, file_names[n])
    imshow(Xt[idx,:,:,0], cmap='gray')
    plt.show()


# In[34]:

print(keras.__version__)


# In[33]:

import keras


# In[ ]:



