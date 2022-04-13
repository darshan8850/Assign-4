# load dogs vs cats dataset, reshape and save to a new file

from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
# define location of dataset
folder = 'data/aero vs dirt/'
photos, labels = list(), list()
# enumerate files in the directory
for file in listdir(folder):
  # determine class
  output = 0.0
  if file.startswith('aero'):
    output = 1.0
  # load image
  photo = load_img(folder + file, target_size=(200, 200))
  # convert to numpy array
  photo = img_to_array(photo)
  # store
  photos.append(photo)
  labels.append(output)
# convert to a numpy arrays
photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)
# save the reshaped photos
save('aero_vs_dirt_photos.npy', photos)
save('aero_vs_dirt_labels.npy', labels)