import cv2
import numpy as np
from keras.models import load_model

facetracker = load_model('facetracker.h5') # Huge thanks to Nicholas Renotte for his YouTube video which helped me learn how to make an image detection model.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1700)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 850)
while cap.isOpened():
    _ , frame = cap.read()
    frame = frame[50:1080, 210:1920, :]
    old_frame = frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (120,120))
    
    yhat = facetracker.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[1][0]
    
    if yhat[0] > 0.7: # If the prediction is good, then crop.
        valtop = list(np.multiply(sample_coords[:2], [1700,1080]).astype(int))
        valbot = list(np.multiply(sample_coords[2:], [1700,1080]).astype(int))
        valtop = (valtop[0] - 40,valtop[1] - 80) # Adding and subtracting a few values to make the crop box slightly bigger than the actual face
        valbot = (valbot[0] + 20,valbot[1] + 60)
        # Cropping into the detected image
        frame = frame[valtop[1]:valbot[1],valtop[0]:valbot[0]]

    if np.shape(frame)[1] == 0: # Incase the image is empty, that is face couldn't be detected, simply show the old, unprocessed video frame. Without this program will crash
        cv2.imshow('CenterStage',old_frame)
    else:
        cv2.imshow('CenterStage', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): #Press Q key to stop the image 
        break
cap.release()
cv2.destroyAllWindows()