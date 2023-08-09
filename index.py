import os
import glob
import random
import numpy as np
import pandas as pd
​
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
​
from tqdm import tqdm
​
from PIL import Image
​
from tensorflow.keras.utils import to_categorical
​
import seaborn as sns
import matplotlib.image as img
import matplotlib.pyplot as plt

### Setting up the path and loading csv files
train_csv = pd.read_csv("../input/human-action-recognition-har-dataset/Human Action Recognition/Training_set.csv")
test_csv = pd.read_csv("../input/human-action-recognition-har-dataset/Human Action Recognition/Testing_set.csv")

train_fol = glob.glob("../input/human-action-recognition-har-dataset/Human Action Recognition/train/*") 
test_fol = glob.glob("../input/human-action-recognition-har-dataset/Human Action Recognition/test/*")

train_csv

train_csv.label.value_counts()

import plotly.express as px
l = train_csv.label.value_counts()
fig = px.pie(train_csv, values=l.values, names=l.index, title='Distribution of Human Activity')
fig.show()

filename = train_csv['filename']
​
situation = train_csv['label']

filename

situation

### Creating a function to random take a image and display it with its label

def disp():
    num = random.randint(1,10000)
    imgg = "Image_{}.jpg".format(num)
    train = "../input/human-action-recognition-har-dataset/Human Action Recognition/train/"
    if os.path.exists(train+imgg):
        testImage = img.imread(train+imgg)
        plt.imshow(testImage)
        plt.title("{}".format(train_csv.loc[train_csv['filename'] == "{}".format(imgg), 'label'].item()))
​
    else:
        #print(train+img)
        print("File Path not found \nSkipping the file!!")

disp()

# Processing data

img_data = []
img_label = []
length = len(train_fol)
for i in (range(len(train_fol)-1)):
    t = '../input/human-action-recognition-har-dataset/Human Action Recognition/train/' + filename[i]    
    temp_img = Image.open(t)
    img_data.append(np.asarray(temp_img.resize((160,160))))
    img_label.append(situation[i])

inp_shape = (160,160,3)

iii = img_data
iii = np.asarray(iii)
type(iii)

y_train = to_categorical(np.asarray(train_csv['label'].factorize()[0]))
print(y_train[0])

y_train[25]

pretrained_model= tf.keras.applications.VGG16(include_top=False,
                   input_shape=(160,160,3),
                   pooling='avg',classes=15,
                   weights='imagenet')
​
for layer in pretrained_model.layers:
        layer.trainable=False
​
vgg_model.add(pretrained_model)
vgg_model.add(Flatten())
vgg_model.add(Dense(512, activation='relu'))
vgg_model.add(Dense(15, activation='softmax'))

vgg_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

vgg_model.summary()

history = vgg_model.fit(iii,y_train, epochs=60)

vgg_model.save_weights("deliLab2_HARmodel.h5")

losss = history.history['loss']
plt.plot(losss)

accu = history.history['accuracy']
plt.plot(accu)


from keras.models import Sequential
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
​
# Create a new instance of the VGG16 mode
vgg_model = Sequential()
​
pretrained_model= tf.keras.applications.VGG16(include_top=False,
                   input_shape=(160,160,3),
                   pooling='avg',classes=15,
                   weights='imagenet')
​
for layer in pretrained_model.layers:
        layer.trainable=False
​
vgg_model.add(pretrained_model)
vgg_model.add(Flatten())
vgg_model.add(Dense(512, activation='relu'))
vgg_model.add(Dense(15, activation='softmax'))
​
​
​
​
# Load the saved weights into the model
vgg_model.load_weights('/kaggle/input/models/deliLab2_HARmodel.h5')

# Custom Testing

# Function to read images as array
​
def read_image(fn):
    image = Image.open(fn)
    return np.asarray(image.resize((160,160)))

# Function to predict
​
def test_predict(test_image):
    result = vgg_model.predict(np.asarray([read_image(test_image)]))
​
    itemindex = np.where(result==np.max(result))
    prediction = itemindex[1][0]
#     print("probability: "+str(np.max(result)*100) + "%\nPredicted class : ", prediction)
    return prediction
#     image = img.imread(test_image)
#     plt.imshow(image)
#     plt.title(prediction)

test_predict('../input/human-action-recognition-har-dataset/Human Action Recognition/test/Image_1010.jpg')

test_predict('../input/human-action-recognition-har-dataset/Human Action Recognition/test/Image_198.jpg')

test_predict('../input/human-action-recognition-har-dataset/Human Action Recognition/test/Image_1091.jpg')

import os
import pandas as pd
​
# Define the path to the folder containing the images
folder_path = "/kaggle/input/human-action-recognition-har-dataset/Human Action Recognition/test"
​
# Create an empty dataframe to store the results
predictions_df = pd.DataFrame(columns=["image_name", "prediction"])
​
# Loop through each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".jpg"): # Adjust the file extension to match your image format
        # Make a prediction for the current image file
        prediction = test_predict(os.path.join(folder_path, file_name))
        
        # Add the prediction result to the dataframe
        predictions_df = pd.concat([predictions_df, pd.DataFrame({"image_name": [file_name], "prediction": [prediction]})], ignore_index=True)
​
final=predictions_df
final

# Define a dictionary that maps each numerical value to its corresponding label
label_map = {
    0: "sitting",
    1: "using laptop",
    2: "hugging",
    3: "sleeping",
    4: "drinking",
    5: "clapping",
    6: "dancing",
    7: "cycling",
    8: "calling",
    9: "laughing",
    10: "eating",
    11: "fighting",
    12: "listening_to_music",
    13: "running",
    14: "texting"
}
​
# Replace the numerical values in the 'prediction' column with their corresponding labels
final["prediction"] = final["prediction"].map(label_map)
​
final


output_filename = "predictions.csv"
​
# Save the dataframe as a CSV file
final.to_csv(os.path.join(output_filename), index=False)
