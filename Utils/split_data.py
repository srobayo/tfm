"""
data-|
     |--> covid
     |--> normal

Once the images have been stored in the covid and normal directories,
partitions are created for the training, validation and test phase.

"""

import os
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split
import shutil
from shutil import copy2
import warnings

warnings.filterwarnings('ignore')


def make_dir(path_dir):
    if os.path.isdir(path_dir):
        shutil.rmtree(path_dir)
        os.makedirs(path_dir, exist_ok=True)
        print('The directory exists, has been deleted and created again')
    else:
        os.makedirs(path_dir, exist_ok=True)
        print('The directories has been created')


main_dir = 'data'

# set the path to the covid images dir
covid_img_dir = os.path.join(main_dir, "covid")

# set the path to the normal images dir
normal_img_dir = os.path.join(main_dir, "normal")

# print the filenames
covid_file_names = tf.io.gfile.glob(str(main_dir + '/covid/*'))
print('************** COVID *******************')
print(covid_file_names[:10])

normal_file_names = tf.io.gfile.glob(str(main_dir + '/normal/*'))
print('************** NORMAL *******************')
print(normal_file_names[:10])

print('****** The total no of images present in each dir*******')
print("total images present in covid folder :", len(covid_file_names))
print("total images present in normal folder :", len(normal_file_names))

print('****************** SHUFFLE DATA ***********************')
random.seed(54000)
tf.random.set_seed(6000)
random.shuffle(normal_file_names)
filenames = normal_file_names[0:6300]  # 5300
filenames.extend(covid_file_names)

# TRAIN_FILENAMES, VALID_FILENAMES = train_test_split(filenames, test_size=0.25, random_state=5)
# TRAIN_FILENAMES, TEST_FILENAMES = train_test_split(TRAIN_FILENAMES, test_size=0.2, random_state=5)

# 90% -> 60/20/20
# TRAIN_FILENAMES, TEST_FILENAMES = train_test_split(filenames, test_size=0.2, random_state=100)
# TRAIN_FILENAMES, VALID_FILENAMES = train_test_split(TRAIN_FILENAMES, test_size=0.25, random_state=42)

# 91% -> 70/15/15
# TRAIN_FILENAMES, TEST_FILENAMES = train_test_split(filenames, test_size=0.15, random_state=100)
# TRAIN_FILENAMES, VALID_FILENAMES = train_test_split(TRAIN_FILENAMES, test_size=0.177, random_state=42)

# 93-95% -> 80/10/10
TRAIN_FILENAMES, TEST_FILENAMES = train_test_split(filenames, test_size=0.1, random_state=100, shuffle=True)
TRAIN_FILENAMES, VALID_FILENAMES = train_test_split(TRAIN_FILENAMES, test_size=0.111, random_state=100, shuffle=True)

print('TRAIN_FILENAMES: ', len(TRAIN_FILENAMES))
print('VALID_FILENAMES: ', len(VALID_FILENAMES))
print('TEST_FILENAMES: ', len(TEST_FILENAMES))

########## CREATE DIRECTORIES TO STORE IMAGES ############

rootc = 'C:/Users/Sixto/Downloads/TFMData/RAWData/data/toGenerator'
contain_directories = ['train', 'test', 'valid']
inner_directories = ['covid', 'normal']

# create main directory container
make_dir(rootc)

for cdir in contain_directories:
    for innerf in inner_directories:
        path = os.path.join(rootc, cdir, innerf)
        make_dir(path)

print('****************** START STORING INTO TRAIN_FOLDER ******************')

src = 'C:/Users/Sixto/Downloads/TFMData/RAWData/data/'

for file in TRAIN_FILENAMES:
    label = file.split('\\')[-2]
    copy2(os.path.join(src + label, str(file).lower()), rootc + '/train/' + label)

print('****************** START STORING INTO VALID_FOLDER ***************')
for file in VALID_FILENAMES:
    label = file.split('\\')[-2]
    copy2(os.path.join(src + label, str(file).lower()), rootc + '/valid/' + label)

print('****************** START STORING INTO TEST_FOLDER ******************')
for file in TEST_FILENAMES:
    label = file.split('\\')[-2]
    copy2(os.path.join(src + label, str(file).lower()), rootc + '/test/' + label)

print('****************** ENDS STORING INTO FOLDERS ******************')
