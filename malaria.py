import numpy as np 
import pandas as pd
import os
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import keras
from keras.utils import np_utils
import warnings
from sklearn.externals import joblib
warnings.filterwarnings("ignore")
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
data = []
labels= []
data_1=os.listdir("../input/cell_images/cell_images/Parasitized/")
for i in data_1:
    try:
        image = cv2.imread("../input/cell_images/cell_images/Parasitized/"+i)
        image_from_array= Image.fromarray(image , "RGB")
        size_image =image_from_array.resize((50,50))
        data.append(np.array(size_image))
        labels.append(0)
    except AttributeError:
        print("")
Uninfected = os.listdir("../input/cell_images/cell_images/Uninfected/")
for b in Uninfected:
    try :
        image = cv2.imread("../input/cell_images/cell_images/Uninfected/"+b)
        array_image=Image.fromarray(image,"RGB")
        size_image=array_image.resize((50,50))
        resize45= size_image.rotate(15)
        resize75 = size_image.rotate(25)
        data.append(np.array(size_image))
        labels.append(1)
    except AttributeError:
        print("")
Cells =np.array(data)
labels =np.array(labels)
s=np.arange(Cells.shape[0])
np.random.shuffle(s)
len_data = len(Cells)
Cells=Cells[s]
labels =labels[s]
labels =keras.utils.to_categorical(labels)
model =Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dense(2,activation="softmax"))
model.summary()#
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
Cells=Cells/255
model.fit(Cells,labels,batch_size=50,epochs=10,verbose=1)
model.save("my_model.h5")#
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)#
tflite_model = converter.convert()#
converter = tf.lite.TFLiteConverter.from_saved_model("../input/my_model.h5")#
tflite_model = converter.convert()#
joblib.load("model")#
joblib.dump(model,"model")#
model.save("model111.h5")#
from keras.models import load_model
model11=load_model("model111.h5")
model11.predict(Cells[73].reshape(1,50,50,3))#
blur=cv2.blur(Cells[1000].rotate(45),(5,5))#
from sklearn.externals import joblib
from keras.applications.xception import Xception
model1=Xception()#
modl= keras.applications.vgg16.VGG16()#
from keras.applications import VGG16 
vgg_conv = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))
from keras.preprocessing.image import ImageDataGenerator
train_img=ImageDataGenerator(rescale=1./255,shear_range=0.1,zoom_range=0.2,horizontal_flip=True)
train_images=train_img.flow_from_directory("../input/cell_images/cell_images/Parasitized/",target_size=(64,64,3),batch_size=32)
