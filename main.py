import keras
from keras.models import load_model
import cv2
import numpy as np
import os
import gradio as gr


# Load the trained model to classify the images
model = load_model(r'model/model.h5')
# load the image
img = cv2.imread(r'images/non_fire.10.png')

# this is the function that will be called when the user want to use the image from the local machine
def predict_input_image(img):
    # Load the image
    image = cv2.imread(img)
    # Preprocess the image
    image = cv2.resize(image, (224, 224))
    class_names = ['fire_images', 'non_fire_images']
    img_4d=image.reshape(-1, 224, 224, 3)
    prediction=model.predict(img_4d)[0]
    if prediction>0.5:
        pred = [1-prediction, prediction]
    else:
         pred = [1-prediction, prediction]
    confidences = {class_names[i]: float(pred[i]) for i in range(2)}
    return confidences

# thats is how we call the function
result = predict_input_image(img)
# print the result
print(result)

# this is the function that will be called when the user uploads an image
def predict_input_image_gr(img):
    class_names = ['fire_images', 'non_fire_images']
    img_4d = img.reshape(-1, 224, 224, 3)
    prediction = model.predict(img_4d)[0]
    if prediction > 0.5:
        pred = [1-prediction, prediction]
    else:
         pred = [1-prediction, prediction]
    confidences = {class_names[i]: float(pred[i]) for i in range(2)}
    return confidences

# this is the interface that will show up by local and public link when the user run the code
# that mean the income is an image and the output is a label
image = gr.inputs.Image(shape=(224, 224))
label = gr.outputs.Label(num_top_classes=1)
gr.Interface(fn=predict_input_image_gr,
inputs=image, outputs=label, interpretation='default').launch(debug='True', share='True')