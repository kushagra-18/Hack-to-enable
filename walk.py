from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import pyttsx3
engine = pyttsx3.init() 

volume = engine.getProperty('volume')   
print (volume)                          
engine.setProperty('volume',1.0)  


classifier = load_model('blind.hdf5')

class_labels = ['Forward','Left','Right']

cap=cv2.VideoCapture(0)


ds_factor=0.6
class VideoCamera(object):
    def __init__(self):
       #capturing video
       self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        #releasing camera
        self.video.release()
        
    def get_frame(self):
        while True:

            ret,frame=cap.read()

            labels=[]

            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            roi_gray = gray

            roi_gray=cv2.resize(roi_gray,(150,150),interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray])!=0:

                roi=roi_gray.astype('float')/255.0

                roi=img_to_array(roi)

                roi=np.expand_dims(roi,axis=0)

                preds=classifier.predict(roi)[0]



                #res = pred.argmax()

                #if res > .55:

                label=class_labels[preds.argmax()]

                print(label)







                label_position=(100,100)

                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            else:
                cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    
   # cv2.imshow('Blind Walk Assistant',frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break
        
#cap.release()
#cv2.destroyAllWindows()
