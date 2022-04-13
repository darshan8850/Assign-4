from os import listdir, makedirs
from random import seed
from random import random
from shutil import copyfile
from sklearn.model_selection import train_test_split

# create directories
dataset_home = 'dataset_aero_vs_dirt/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
  # create label subdirectories
  labeldirs = ['aero/', 'dirt/']
  for labldir in labeldirs:
    newdir = dataset_home + subdir + labldir
    makedirs(newdir, exist_ok=True)



# seed random number generator
#seed(1)

i=1
# copy training dataset images into subdirectories
src_directory = 'data/aero vs dirt/'

for file in listdir(src_directory):
  src = src_directory + '/' + file
  dst_dir = 'train/'
  if i <=10:
    dst_dir = 'test/'
    i+=1
  if file.startswith('aero'):
    dst = dataset_home + dst_dir + 'aero/'  + file
    copyfile(src, dst)


j=1
for file in listdir(src_directory):
  src = src_directory + '/' + file
  dst_dir = 'train/'
  if j <=10:
    dst_dir = 'test/'
    j+=1

  if file.startswith('dirt'):
      print(file)
      dst = dataset_home + dst_dir + 'dirt/'  + file
      copyfile(src, dst)     
