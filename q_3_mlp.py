import sys
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense,Activation
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split 
from PIL import Image
import os

def define_model_mlp():
  hidden_units = 200
  image_size = 200
  input_size = image_size * image_size*3
  n=244

  model=Sequential()
  model.add(Dense(hidden_units, input_shape=(200*200*3,)))
  for i in range(n-2):
    model.add(Dense(hidden_units))
  model.add(Dense(1))
  # this is the output for one-hot vector
  model.add(Activation('softmax'))
  # compile model
  opt = SGD(lr=0.001, momentum=0.9)
  model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

  return model

def plotmodelhistory(history): 
    fig, axs = plt.subplots(1,2,figsize=(15,5)) 
    # summarize history for accuracy
    axs[0].plot(history.history['accuracy']) 
    #axs[0].plot(history.history['val_accuracy']) 
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy') 
    axs[0].set_xlabel('Epoch')
    
    axs[0].legend(['train', 'validate'], loc='upper left')
    # summarize history for loss
    axs[1].plot(history.history['loss']) 
    #axs[1].plot(history.history['val_loss']) 
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss') 
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'validate'], loc='upper left')
    plt.show()
 
# plot diagnostic learning curves
def summarize_diagnostics(history):
  # plot loss
  pyplot.subplot(211)
  pyplot.title('Cross Entropy Loss')
  pyplot.plot(history.history['loss'], color='blue', label='train')
  #pyplot.plot(history.history['val_loss'], color='orange', label='test')
  # plot accuracy
  pyplot.subplot(212)
  pyplot.title('Classification Accuracy')
  pyplot.plot(history.history['accuracy'], color='blue', label='train')
  #pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
  # save plot to file
  filename = sys.argv[0].split('/')[-1]
  pyplot.savefig(filename + '_plot.png')
  pyplot.close()

PATH_TRAIN_dirt = 'dataset_aero_vs_dirt/train/dirt' 
PATH_TRAIN_aero = 'dataset_aero_vs_dirt/train/aero' 
PATH_TEST_dirt = 'dataset_aero_vs_dirt/test/dirt' 
PATH_TEST_aero = 'dataset_aero_vs_dirt/test/aero' 

dirt = 1 
aero = 0

IMG_WIDTH = 200 
IMG_HEIGHT = 200 
NR_PIX = IMG_HEIGHT * IMG_WIDTH
NR_CHANELS = 3
NR_FEATURES = NR_PIX * NR_CHANELS

def load_data_asanarray(path):
    data = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            specific_path = os.path.join(dirname, filename)
            image = Image.open(specific_path).resize((200,200))
            img_array = np.asarray(image)
            data.append(img_array)
    return np.asarray(data).reshape(len(data),IMG_WIDTH, IMG_HEIGHT, NR_CHANELS)

data_train_dirt = load_data_asanarray(PATH_TRAIN_dirt)
data_train_aero = load_data_asanarray(PATH_TRAIN_aero)
print(f'data_train_dirt size is {len(data_train_dirt)} data_train_aero size is {len(data_train_aero)}')

y_label_dirt = np.ones((len(data_train_dirt),1))
y_label_aero = np.zeros((len(data_train_aero),1))

x_train = np.concatenate((data_train_dirt, data_train_aero), axis=0)
y_train = np.concatenate((y_label_dirt, y_label_aero), axis=0)

data_test_dirt = load_data_asanarray(PATH_TEST_dirt)
data_test_aero = load_data_asanarray(PATH_TEST_aero)
print(f'data_train_dirt size is {len(data_train_dirt)} data_train_aero size is {len(data_train_aero)}')

y_label_dirt = np.ones((len(data_test_dirt),1))
y_label_aero= np.zeros((len(data_test_aero),1))
x_test = np.concatenate((data_test_dirt, data_test_aero), axis=0)
y_test = np.concatenate((y_label_dirt, y_label_aero), axis=0)
print(f'x_test shape is {x_test.shape} and y_shape is {y_test.shape}')

X = np.concatenate((x_train, x_test), axis=0)
Y = np.concatenate((y_train, y_test), axis=0) 

x_train,x_test,y_train, y_test = train_test_split(X,Y, test_size=0.15, random_state=45)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_train = x_train.reshape(len(x_train), NR_FEATURES).T
x_test = x_test.reshape(len(x_test), NR_FEATURES).T

x_train = x_train.T
x_test = x_test.T
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
x_train, x_test = x_train / 255, x_test/ 255 


model=define_model_mlp()

history2= model.fit(x_train,y_train, steps_per_epoch=len(x_train), epochs=50, verbose=0)

# evaluate model
_, acc = model.evaluate(x_test,y_test, steps=len(x_test), verbose=0)
print(acc * 100.0)
# learning curves
summarize_diagnostics(history2)
plotmodelhistory(history2)
 

