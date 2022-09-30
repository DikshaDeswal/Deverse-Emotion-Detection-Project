from keras.models import load_model
from time import sleep
#from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import seaborn as sns
import pandas as pd
import imageio

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


def EmotionDetectProcess():

    face_classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
    classifier =load_model(r'model 2.h5')

    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
    arr=[]
    time=[]
    emotion=[]
    list = [0] * 7
    i=0

    cap = cv2.VideoCapture(r"static/video/1.mp4")

    while (cap.isOpened()):
        print("loop is running")
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
                #print(label)
                time.append(i+1)
                i=i+1
                if(label=='Happy') :
                    arr.append(1)
                    emotion.append(label)
                    list[0]+=1
                elif(label=='Neutral'):
                    arr.append(2)
                    emotion.append(label)
                    list[1] += 1
                elif (label == 'Sad'):
                    arr.append(3)
                    emotion.append(label)
                    list[2] += 1
                elif (label == 'Angry'):
                    arr.append(4)
                    emotion.append(label)
                    list[3] += 1
                elif (label == 'Fear'):
                    arr.append(5)
                    emotion.append(label)
                    list[4] += 1
                elif (label == 'Disgust'):
                    arr.append(6)
                    emotion.append(label)
                    list[5] += 1
                elif (label == 'Surprise'):
                    arr.append(7)
                    emotion.append(label)
                    list[6] += 1

                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        frame = cv2.imencode('.JPEG', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        sleep(0.016)
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # cv2.imshow('Emotion Detector',frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # cap.release()
    # cv2.destroyAllWindows()



# # create plot
#
#
# keys = ['Happy', 'Neutral', 'Sad', 'Angry', 'Fear','Disgust','Surprise']
# palette_color = sns.color_palette('bright')
# fig=plt.pie(list, labels=keys, colors=palette_color, autopct='%.0f%%')
# plt.savefig("Output2.jpg")
# plt.close()
#
# import seaborn as sns
# sns.set()
# sns_plot1 = sns.scatterplot(time, arr)
# fig1 = sns_plot1.get_figure()
# fig1.savefig("output1.jpg")
#
# palette_color = sns.color_palette('bright')
# sns_plot3=sns.countplot(emotion)
# fig2 = sns_plot3.get_figure()
# fig2.savefig("Output3.jpg")