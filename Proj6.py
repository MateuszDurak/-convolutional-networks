import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical
import numpy as np
import ssl
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
ssl._create_default_https_context = ssl._create_unverified_context # To ignore SSL certificate error

# 0 airplane
# 1 automobile
# 2 bird
# 3 cat
# 4 deer
# 5 dog
# 6 frog
# 7 horse
# 8 ship
# 9 truck
# Przydział do klas (Zwierze, Pojazd)
def data_selection(y_data):
  counter = 0
  iterations = len(y_data)

  for i in range(iterations):
    if  y_data[i] == 2 or y_data[i] == 3 or y_data[i] == 4  or y_data[i] == 5  or y_data[i] == 6 or y_data[i] == 7: # zwierze
      y_data[i] = 1 # 1 - zwierze
      counter+=1
    elif y_data[i] == 0 or y_data[i] == 1 or y_data[i] == 8  or y_data[i] == 9: # Pojazd
      y_data[i] = 0 # 0 - pojazd
      counter+=1


# Wczytanie danych i podzielenie na treningowe i testowe
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
# normalizacja danych
X_train = X_train / 255.0
X_test = X_test / 255.0

classes = ['Pojazd', 'Zwierze']

# Wybieranie danych Y
data_selection(y_train)
data_selection(y_test)


plt.figure(figsize = (10,10))

images_quantity = 10
for i in range(images_quantity):
  plt.subplot(1, 10, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(X_train[i], cmap=plt.cm.binary)
  plt.xlabel(classes[y_train[i][0]])

plt.show()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# pojedyncza warstwa
model_one = Sequential()
model_one.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model_one.add(MaxPooling2D((2, 2)))
model_one.add(Flatten())
model_one.add(Dense(2, activation = 'softmax'))

model_one.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model_one.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_train, y_train))
model_one.save('miw_s21415_f_models_model{}_fit.h5'.format(1))

# dwie warstwy
model_two = Sequential()
model_two.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model_two.add(MaxPooling2D((2, 2)))
model_two.add(Conv2D(64, (3, 3), activation='tanh'))
model_two.add(MaxPooling2D((2, 2)))
model_two.add(Flatten())
model_two.add(Dense(2, activation = 'softmax'))

model_two.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model_two.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_train, y_train))
model_two.save('miw_s21415_f_models_model{}_fit.h5'.format(2))

# trzy warstwy

model_three = Sequential()
model_three.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model_three.add(MaxPooling2D((2, 2)))
model_three.add(Conv2D(64, (3, 3), activation='tanh'))
model_three.add(MaxPooling2D((2, 2)))
model_three.add(Conv2D(128, (3, 3), activation='sigmoid'))
model_three.add(MaxPooling2D((2, 2)))
model_three.add(Flatten())
model_three.add(Dense(2, activation = 'softmax'))

model_three.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model_three.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_train, y_train))
model_three.save('miw_s21415_f_models_model{}_fit.h5'.format(3))

# jedna warstwa
model_one = keras.models.load_model('miw_s21415_f_models_model{}_fit.h5'.format(1))
loss, acc = model_one.evaluate(X_train, y_train, verbose=0)
print('Jedna warstwa:')
print('accuracy: {}'.format(acc))
print('loss: {}'.format(loss))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.title('jedna warstwa')
plt.show()

# dwie
model_two = keras.models.load_model('miw_s21415_f_models_model{}_fit.h5'.format(2))
loss, acc = model_two.evaluate(X_train, y_train, verbose=0)
print('dwie warstwy:')
print('accuracy: {}'.format(acc))
print('loss: {}'.format(loss))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.title('dwie warstwy')
plt.show()

# trzy
model_three = keras.models.load_model('miw_s21415_f_models_model{}_fit.h5'.format(3))
loss, acc = model_three.evaluate(X_train, y_train, verbose=0)
print('trzy warstwy:')
print('accuracy: {}'.format(acc))
print('loss: {}'.format(loss))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.title('trzy warstwy')
plt.show()