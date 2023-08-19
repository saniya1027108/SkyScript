import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}


model = tf.keras.models.load_model("model_hand.h1")

img = cv2.imread(r'english-handwritten-characters-dataset/Img/img013-004.png')
img_copy = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (400,440))

img_copy = cv2.GaussianBlur(img_copy, (7,7), 0)
img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
_, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

img_final = cv2.resize(img_thresh, (64, 64))
#img_final =np.reshape(img_final, (1,64,64,1))

img_final = np.expand_dims(img_final, axis=-1)  # Add a new axis to represent the single channel
img_final = np.repeat(img_final, 3, axis=-1)    
img_final = np.reshape(img_final, (1, 64, 64, 3))

img_pred = word_dict[np.argmax(model.predict(img_final))]

cv2.putText(img, "Character Recognition", (20,25), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color = (0,0,230))
cv2.putText(img, "Prediction: " + img_pred, (20,410), cv2.FONT_HERSHEY_DUPLEX, 1.3, color = (255,0,30))
cv2.imshow('Dataflair handwritten character recognition _ _ _ ', img)


while (1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()

