import numpy as np
import cv2
from collections import deque
import tensorflow as tf
from tensorflow import keras
# from keras.models import load_model

model = tf.keras.models.load_model("/Users/saniyamulla/Documents/Project/Air Canvas/model_hand.h5")

# Preprocess the character image for model input
character_image = cv2.imread('cropped_img.jpg')
character_image = cv2.cvtColor(character_image, cv2.COLOR_BGR2GRAY)
character_image = cv2.resize(character_image, (28, 28))
character_image = character_image.reshape(1, 28, 28, 1)
character_image = character_image.astype('float32')
character_image /= 255.0

# Get the prediction from the model
# prediction = model.predict_step(character_image)[0]
prediction = model.predict(character_image)[0]
print(np.argmax(prediction))
# print(model.class_indices)

    # Map the predicted class label to the corresponding character
#predicted_character = chr(ord('A') + prediction)  