# Classification_of_cat-dog
# Classification_of_cat-dog
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

nb_train_samples = 2000
nb_validation_samples = 800

# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all test images
train_generator = train_data_gen.flow_from_directory(r'path/train',target_size=(150,150),
        batch_size=64,
        class_mode='binary')

# Preprocess all train images
validation_generator = validation_data_gen.flow_from_directory(
        r"path/validation",
        target_size=(150,150),
        batch_size=64,
        class_mode='binary')

# create model structure
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(150,150, 3)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Use 'learning_rate' instead of 'lr'
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Train the neural network/model using Model.fit
model_info = model.fit(train_generator,
                       steps_per_epoch=nb_train_samples // 64,  # Adjust based on batch_size
                       epochs=35,
                       validation_data=validation_generator,
                       validation_steps=nb_validation_samples // 64)  # Adjust based on batch_size



# save model structure in jason file
model_json = model.to_json()

with open("path/emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
model.save('path/emotion_model.h5')



