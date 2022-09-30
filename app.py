from flask import Flask,render_template,url_for,request,redirect,Response
import cv2,imutils
import os
import numpy as np
from werkzeug.utils import secure_filename
import main

#creating an instance of flask application
app=Flask(__name__)

#routes to html pages.
@app.route("/")
def index():
    return render_template('home.html')

@app.route("/myproduct")
def myproduct():
    return render_template('myproduct.html')

@app.route("/aboutus")
def aboutus():
    return render_template('aboutus.html')

@app.route('/feedback',methods=['GET','POST'])
def detect():
    if request.method=='POST':

        if request.files:
            print("found video")
            file = request.files["file"]
            filename = secure_filename(file.filename)
            # save image
            file.save(os.path.join("static", "video","1.mp4"))
            print("\n app.py :- mp4 video saved")

            return render_template('displayVideo.html')

    else:
        return render_template('feedback.html')

@app.route('/displayVideo')
def video_feed():
    return Response(main.EmotionDetectProcess(),mimetype='multipart/x-mixed-replace; boundary=frame')

#run flask app
if __name__ == '__main__':
    app.run(debug=True,threaded=True)