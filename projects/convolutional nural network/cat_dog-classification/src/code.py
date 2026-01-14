import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



data_path = 'PetImages'
dog_path = os.path.join(data_path, 'Dog')
cat_path = os.path.join(data_path, 'Cat')


def clean(path):
    images = os.listdir(path)

    for image in images:
        if os.path.getsize(os.path.join(path, image)) == 0 or image.split('.')[-1] != 'jpg':
            images.remove(image)
            print('remove')
    return images


dogs = clean(dog_path)
cats = clean(cat_path)


def train_validation_dirs( dir_name, data, SPLIT_SIZE = 0.7, data_path = data_path, data_length = 7000):
    training_length = round( data_length * SPLIT_SIZE)
    
    train_set = data[0:training_length]
    val_set = data[training_length: data_length]
    
    os.makedirs(os.path.join(data_path,'training', dir_name))
    os.makedirs(os.path.join(data_path,'validation', dir_name))
    
    for image in train_set:
        shutil.copyfile(os.path.join(data_path, dir_name, image), os.path.join(data_path,'training', dir_name, image) )
    
    for image in val_set:
        shutil.copyfile(os.path.join(data_path, dir_name, image), os.path.join(data_path,'validation', dir_name, image) )


train_validation_dirs("Dog", dogs)
train_validation_dirs("Cat", cats)


train_generator = ImageDataGenerator(rescale=1/255, horizontal_flip=True, zoom_range=0.4, fill_mode='nearest', rotation_range=45)
val_generator = ImageDataGenerator(rescale=1/255, horizontal_flip=True, zoom_range=0.4, fill_mode='nearest', rotation_range=45)


train_generator = train_generator.flow_from_directory(directory=os.path.join(data_path, 'training'), shuffle=True, class_mode='binary', target_size=(150, 150), batch_size=100)
val_generator = val_generator.flow_from_directory(directory=os.path.join(data_path, 'validation'), shuffle=True, class_mode='binary', target_size=(150, 150), batch_size=100)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(train_generator, validation_data=val_generator, epochs=10, verbose=1)



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()



def predict(src):
    test = Image.open(os.path.join(data_path, src))
    test = test.resize((150, 150))

    test = np.asarray(test)
    test = test / 255

    prediction  = model.predict(tf.expand_dims(test, 0))

    if prediction < 0.5:
        print('cat', prediction)
    else:
        print('dog', prediction)
    