from keras.models import load_model
from time import sleep
#from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import seaborn as sns
import pandas as pd

def img_to_array(img, data_format='channels_last', dtype='float32'):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x


face_classifier = cv2.CascadeClassifier(r'C:/Users/DIKSHA DESWAL/PycharmProjects/pythonProject_deverse/haarcascade_frontalface_default.xml')
classifier =load_model(r'C:/Users/DIKSHA DESWAL/PycharmProjects/pythonProject_deverse/model 2.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
arr=[]
time=[]
i=0

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    labels = []
    if (_==False):
        break

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            print(label)
            time.append(i+1)
            i=i+1
            list = [0] * 7
            if(label=='Happy') :
                arr.append(1)
                list[0]+=1
            elif(label=='Neutral'):
                arr.append(2)
                list[1] += 1
            elif (label == 'Sad'):
                arr.append(3)
                list[2] += 1
            elif (label == 'Angry'):
                arr.append(4)
                list[3] += 1
            elif (label == 'Fear'):
                arr.append(5)
                list[4] += 1
            elif (label == 'Disgust'):
                arr.append(6)
                list[5] += 1
            elif (label == 'Surprise'):
                arr.append(7)
                list[6] += 1

            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
for i in arr:
    print(i)
    print(" ")

# create plot

import seaborn as sns
sns.set()
sns_plot = sns.scatterplot(time, arr)
fig = sns_plot.get_figure()
fig.savefig("output3.png")


keys = ['Happy', 'Neutral', 'Sad', 'Angry', 'Fear','Disgust','Surprise']

# # define Seaborn color palette to use
palette_color = sns.color_palette('bright')

#plotting data on chart
plt.pie(list, labels=keys, colors=palette_color, autopct='%.0f%%')

 # displaying chart
#plt.show()
