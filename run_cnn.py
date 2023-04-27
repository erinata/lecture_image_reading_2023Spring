import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from keras import layers
import pickle

train_dataset = ImageDataGenerator(rescale = 1/255).flow_from_directory(directory='dataset/images/',
                                         target_size=(50,50), shuffle=True)
                                         
values = list(train_dataset.class_indices.values())
keys = list(train_dataset.class_indices.keys())
print([[values[i], keys[i]] for i in range(len(values))])


# Initializing the model
machine = Sequential()
machine.add(layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(50,50,3)))
machine.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
machine.add(layers.MaxPooling2D(pool_size=(2,2)))
machine.add(layers.Dropout(0.25))

machine.add(layers.Conv2D(filters=64, activation='relu', kernel_size=(3,3)))
machine.add(layers.Conv2D(filters=64, activation='relu', kernel_size=(3,3)))
machine.add(layers.MaxPooling2D(pool_size=(2,2)))
machine.add(layers.Dropout(0.25))
          
machine.add(layers.Flatten())
machine.add(layers.Dense(units=64, activation='relu'))
machine.add(layers.Dense(units=64, activation='relu'))
machine.add(layers.Dropout(0.25))
machine.add(layers.Dense(6, activation='softmax'))

machine.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
machine.fit(train_dataset, batch_size=128, epochs=30) 

pickle.dump(machine, open('cnn_image_machine.pickle', 'wb'))
