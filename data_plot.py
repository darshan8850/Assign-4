# plot dog photos from the dogs vs cats dataset
import matplotlib
from matplotlib import pyplot
from matplotlib.image import imread
# define location of dataset
folder = 'data/aeroplane/'
# plot first few images
for i in range(9):
  # define subplot
  pyplot.subplot(330 + 1 + i)
  # define filename
  filename = folder + 'aero.' + str(i+1) + '.jpg'
  # load image pixels
  image = imread(filename)
  # plot raw pixel data
  pyplot.imshow(image)
# show the figure
pyplot.show()

# define location of dataset
folder = 'data/Dirt bike/'
# plot first few images
for i in range(9):
  # define subplot
  pyplot.subplot(330 + 1 + i)
  # define filename
  filename = folder + 'dirt.' + str(i+1) + '.jpg'
  # load image pixels
  image = imread(filename)
  # plot raw pixel data
  pyplot.imshow(image)
# show the figure
pyplot.show()

