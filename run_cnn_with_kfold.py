import glob
from sklearn.model_selection import KFold

import os
import shutil

import keras
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Activation, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator

import pickle

if not os.path.exists('dataset/test_images'):
  os.mkdir('dataset/test_images/')

class_list = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
class_list_length  = len(class_list)

for i in class_list:
  if not os.path.exists('dataset/test_images/' + i):
    os.mkdir('dataset/test_images/' + i)


file_list = []
for i in class_list:
  file_list.append([i for i in glob.glob('dataset/images/' + i + '/*.jpg')])
  
  
split_number = 4
kfold_object = KFold(n_splits=split_number)

index_list =  []
for i in range(class_list_length):
  kfold_object.get_n_splits(file_list[i])
  index_list.append([[training_index, test_index] for training_index, test_index in kfold_object.split(file_list[i])])
  
  
for count in range(split_number):
  print("round: ", count)
  for i in range(class_list_length):
    for j in index_list[i][count][1]:
      # print(file_list[i][j], 'dataset/test_images/' + class_list[i] + '/' + file_list[i][j].split('/')[-1])
      shutil.move(file_list[i][j], 'dataset/test_images/' + class_list[i] + '/' + file_list[i][j].split('/')[-1])


  train_dataset = ImageDataGenerator(rescale = 1/255).flow_from_directory(directory='dataset/images/', target_size=(50,50), shuffle=True)
  test_dataset = ImageDataGenerator(rescale = 1/255).flow_from_directory(directory='dataset/test_images/', target_size=(50,50), shuffle=True)                 

  values = list(train_dataset.class_indices.values())
  keys = list(train_dataset.class_indices.keys())
  print([[values[i], keys[i]] for i in range(len(values))])

  machine = Sequential()
  machine.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(50,50,3)))
  machine.add(Activation('relu'))
  machine.add(Conv2D(filters=32, kernel_size=(3,3)))
  machine.add(Dropout(0.25))
  machine.add(Activation('relu'))
  machine.add(MaxPooling2D(pool_size=(2,2)))

  machine.add(Conv2D(filters=64, kernel_size=(3,3)))
  machine.add(Activation('relu'))
  machine.add(Conv2D(filters=64, kernel_size=(3,3)))
  machine.add(Dropout(0.25))
  machine.add(Activation('relu'))
  machine.add(MaxPooling2D(pool_size=(2,2)))

  machine.add(Flatten())
  machine.add(Dense(units=64, activation='relu'))
  machine.add(Dense(units=64, activation='relu'))
  machine.add(Dropout(0.25))
  machine.add(Dense(6, activation='softmax'))

  # machine.summary()

  machine.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
  # machine.fit(train_dataset, batch_size=128, epochs=3) 
  machine.fit(train_dataset, batch_size=128, epochs=30, validation_data=test_dataset) 


  for i in range(class_list_length):
    for j in class_list:
      for file_name in os.listdir('dataset/test_images/'+ j):
          shutil.move(os.path.join('dataset/test_images/' + j, file_name), 'dataset/images/' + j)





train_dataset = ImageDataGenerator(rescale = 1/255).flow_from_directory(directory='dataset/images/',
                                         target_size=(50,50), shuffle=True)
                                         
values = list(train_dataset.class_indices.values())
keys = list(train_dataset.class_indices.keys())
print([[values[i], keys[i]] for i in range(len(values))])


# Initializing the model
machine = Sequential()
machine.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(50,50,3)))
machine.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
machine.add(MaxPooling2D(pool_size=(2,2)))
machine.add(Dropout(0.25))

machine.add(Conv2D(filters=64, activation='relu', kernel_size=(3,3)))
machine.add(Conv2D(filters=64, activation='relu', kernel_size=(3,3)))
machine.add(MaxPooling2D(pool_size=(2,2)))
machine.add(Dropout(0.25))
          
machine.add(Flatten())
machine.add(Dense(units=64, activation='relu'))
machine.add(Dense(units=64, activation='relu'))
machine.add(Dropout(0.25))
machine.add(Dense(6, activation='softmax'))

# machine.summary()

machine.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
machine.fit(train_dataset, batch_size=128, epochs=30) 

pickle.dump(machine, open('cnn_image_machine.pickle', 'wb'))




