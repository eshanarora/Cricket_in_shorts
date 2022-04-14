import tempfile
import numpy as np
import pandas as pd
import os
import cv2
import shutil
import math
from pathlib import Path
import PIL
from skimage.transform import resize   # for resizing imagesy
import tensorflow as tf
import glob
import streamlit as st
import time

from tensorflow import keras

import base64

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-repeat: no-repeat;
    background-position: center;
    background-size: 1707px 1000px;
    background-blend-mode: darken;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('C:/Users/Eshan Arora/OneDrive/Desktop/Promject/background.png')


header=st.container()
dataset =st.container()
output =st.container()

#<div style="background-color:tomato;padding:1.5px">
with header:
	html_temp = """
	
	<h1 style="color:white;text-align:center;font-size: 80px;">Welcome to Cricket Inshorts </h1>
	</div><br>"""
	st.markdown(html_temp,unsafe_allow_html=True)
	st.title('Upload the Video File')
	st.markdown('<style>h1{color: white;font-size: 30px;}</style>', unsafe_allow_html=True)
	#st.header('Welcome to Cricket Inshorts')
	#html_temp = """
	#<h1 style="color:white;text-align:center;">Welcome to Cricket Inshorts </h1>
	#</div><br>"""
	#st.markdown(html_temp,unsafe_allow_html=True)
	#st.markdown('<style>h1{color: white;text-align:center;font-size: 100px;}</style>', unsafe_allow_html=True)

with dataset:
	#st.header('Upload the Input Video')
	#html_temp = """
	#<h2 style="color:white;text-align:center;"> Upload the Input Video </h2>
	#</div><br>"""
	#st.markdown(html_temp,unsafe_allow_html=True)
	#st.markdown('<style>h2{color: white;text-align:center;font-size: 200px;}</style>', unsafe_allow_html=True)



	count = 0
	f = st.file_uploader("")

	tfile = tempfile.NamedTemporaryFile(delete=False) 
	tfile.write(f.read())


	cap = cv2.VideoCapture(tfile.name)
	#cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
	frameRate = cap.get(5) #frame rate
	x=1
	while(cap.isOpened()):
	    frameId = cap.get(1) #current frame number
	    ret, frame = cap.read()
	    if (ret != True):
	        break
	    if (frameId % math.floor(frameRate) == 0):
	        filename ="frame%d.jpg" % count;count+=1
	        cv2.imwrite(filename, frame)
	cap.release()
	lst = list()
	for i in range(0,count):
	    lst.append('frame'+str(i)+'.jpg')
	df=pd.DataFrame(lst, columns = ['Image_ID'])
	
	#st.dataframe(df)
	os.makedirs(os.path.join('./','CricketTest'))
	model_4 = keras.models.load_model('C:/Users/Eshan Arora/OneDrive/Desktop/Promject/Inception V3 50_epoch.h5')
	for i in df['Image_ID']:    
	    get_image = os.path.join('.//',i)
	    move_image_to_cat = shutil.move(get_image, 'C:/Users/Eshan Arora/OneDrive/Desktop/Promject/CricketTest/' + str(i))
	test_list = tf.io.gfile.listdir('./CricketTest/')
	test_data = []
	for f in test_list:
	    img = tf.keras.preprocessing.image.load_img('./CricketTest/' + f, color_mode = "rgb", target_size = (227,227))
	    img = tf.keras.preprocessing.image.img_to_array(img)
	    img = img/255
	    test_data.append(img)
	pred_new = model_4.predict(tf.convert_to_tensor(test_data))
	prednew = np.argmax(pred_new, axis = 1)
	df['Class'] = prednew
	Batting = df[df['Class']==0]
	Bowling = df[df['Class']==1]
	Boundary = df[df['Class']==2]
	CloseUp = df[df['Class']==3]
	Misc = df[df['Class']==4]
	img_array = []
	for i,row in Batting.iterrows():
	    filename = os.path.join('./CricketTest/',row.Image_ID)
	    img = cv2.imread(filename)
	    height, width, layers = img.shape
	    size = (width,height)
	    img_array.append(img)
	 
	 
	out = cv2.VideoWriter('Batting.avi',cv2.VideoWriter_fourcc(*'DIVX'), 1, size)
	 
	for i in range(len(img_array)):
	    out.write(img_array[i])
	out.release()

#time.sleep(300)	
os.system("ffmpeg -y -i Batting.avi -vcodec libx264 Batting.mp4")

with output:
	st.title("Here's your Cricket in Shorts Video")
	st.markdown('<style>h1{color: white;font-size: 30px;}</style>', unsafe_allow_html=True)

	video_file = open("C:/Users/Eshan Arora/OneDrive/Desktop/Promject/Batting.mp4", 'rb')
	video_bytes = video_file.read()

	st.video(video_bytes)



shutil.rmtree('C:/Users/Eshan Arora/OneDrive/Desktop/Promject/CricketTest')
#os.remove('C:/Users/Eshan Arora/OneDrive/Desktop/Promject/Batting.mp4')