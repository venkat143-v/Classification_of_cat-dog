# Classification_of_cat-dog
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.optimizers import Adam

# Load the saved model structure from the JSON file
with open("path/emotion_model.json", "r") as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# Load the saved model weights from the .h5 file
loaded_model.load_weights('path/emotion_model.h5')

# Compile the loaded model (necessary for predictions)
loaded_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Function to preprocess an image for prediction
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Test image path
test_image_path = r"dog.39.jpg" # Replace with the path of the test image

# Preprocess the test image for prediction
test_image = preprocess_image(test_image_path)

# Make predictions on the test image
prediction = loaded_model.predict(test_image)

# Classify the prediction (assuming binary classification)
class_label = "dog" if prediction[0][0] >= 0.5 else "cat"

# Print the prediction result
print(class_label)


