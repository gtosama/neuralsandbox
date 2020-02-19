from realtimeutils import process
import cv2
import numpy as np


from keras.models import load_model
from keras import backend as K

model = load_model("mnist.model")
model.load_weights('mnist.h5')

cap = cv2.VideoCapture(0)
if __name__ == "__main__" :

    while cap.isOpened():
        _ , f = cap.read()
        org = f
        im = process(cv2, np , f , model)
        cv2.imshow("digits", im)
        d = cv2.waitKey(1)
        
        if d==27:
            break
    cap.release()
    cv2.destroyAllWindows()