
import pickle
import pandas
import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy
import glob


machine = pickle.load(open('cnn_image_machine.pickle', 'rb'))


new_data = ImageDataGenerator(rescale = 1/255).flow_from_directory('dataset/new_images/',
                                             target_size=(50,50), batch_size=1, shuffle=False)

new_data.reset()

new_data_length = len([i for i in glob.glob('dataset/new_images/images/*.jpg')])

prediction = numpy.argmax(machine.predict_generator(new_data,steps = new_data_length), axis=1)

results = [[new_data.filenames[i], prediction[i]]for i in range(new_data_length)]
results = pandas.DataFrame(results, columns=['image', 'prediction'])

print(results)


# results.to_csv('new_images_prediction.csv', index=False)

