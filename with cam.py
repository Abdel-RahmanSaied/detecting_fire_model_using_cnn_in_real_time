import numpy as np
import cv2
import os
import keras
from keras.models import load_model

img = r'images/fire.90.png'
model = load_model(r'model/model.h5')

def predict_input_image(img):
    class_names = ['fire_images', 'non_fire_images']
    img_4d = img.reshape(-1, 224, 224, 3)
    prediction = model.predict(img_4d)[0]
    if prediction > 0.5:
        pred = [1-prediction, prediction]
    else:
         pred = [1-prediction, prediction]
    print(predlab)
    confidences = {class_names[i]: float(pred[i]) for i in range(2)}
    print(confidences)
    return confidences


cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print('Camera not found')
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print('Can not receive frame (stream end?). Exiting ...')
        break

    # Preprocess the image
    resized_frame = cv2.resize(frame, (224, 224))
    pred = predict_input_image(resized_frame)
    if pred['fire_images'] > pred['non_fire_images']:
        res = f"Fire {round(pred['fire_images']*100, 2)} %"
    else:
        res = f"No Fire {round(pred['non_fire_images']*100, 2)} %"
    frame = cv2.putText(frame, str(res), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break









# image = gr.inputs.Image(shape=(224, 224))
# label = gr.outputs.Label(num_top_classes=1)
#
#
# gr.Interface(fn=predict_input_image,
# inputs=image, outputs=label,interpretation='default').launch(debug='True', share='True')