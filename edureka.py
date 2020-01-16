#import OpenCV Module
import cv2
import pandas 
#colored image
img = cv2.imread('/home/sahil/Documents/snial.jpg',1)#read the image in RGB colored format
#img = cv2.imread('/home/sahil/Documents/snail.jpg',0)#read the image in gray scale image or black and white image
print(type(img))#if you wanna see what kind of matrice it is
print(img)
print(img.shape)
#opens the window to display the image
cv2.imshow('snial',img)#img = image object and snail = name of the window
cv2.waitKey(0)#wait until a user press key
cv2.waitKey(2000)#in ms
cv2.destroyAllWindows()#closes the window based on waitforkey parameters
#resized_image = cv2.resize(img,(300,300))
resized_image = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))#inside int are the values of the new image
cv2.imshow("snial",resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#image detection
#create a CascadeClassifier object
face_cascade = cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_default.xml")
#haarcascade_frontal_faace_default.xml is the path to the xml file which contains the face features
#reading the image ass it is
img = cv2.imread('/home/sahil/Documents/snial.jpg',1)
#reading the image as gray scale image
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#converting colored image to gray scale
# search the coordinate of the image
faces=face_cascade.detectMultiScale(gray_img,1.3,5)
#faces = face_cascade.detecMultiScale(gray_img, scaleFactor = 1.05, minNeighbors=5)#detecMultiScale helps us to search
#the rectangle co-oordinates scaleFactor decreases the shape value by 5%, until the face is found , Smaller this value,
#grater the accuracy
print(type(faces))
print(faces)
#how are we going to add the rectangular face is by adding a for loop ( a method to create a face rectangle)
for x,y,w,h in faces:
	img = cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),3)
	#rectangle = method to create the face object
	#img = imgage object
	#(0,255,0) = RGB value of the rectangular box aand 3 = width of the rectangle
cv2.imshow("snial", img)
cv2.waitKey()
cv2.destroyAllWindows()


# Dispaying Video
# we need to create a frame object , which will read the images of the VideoCapture object
# we will recessively show each frame of the video being captured
video = cv2.VideoCapture(0)#this method is used to create VideoCapture objects. It will trigger the camera
#either give path to video capture or use numbers. Numbers specify to use web came to capture video
#zero is to specify to use built in camera
video.realse()#this will release camera in some milliseconds
import cv2, time #import time module
video =  cv2.VideoCapture(0)
check, frame = video.read()
# check : it is a bool data type returns true if python is able to read the VideoCapture object
# frame : it is a numpy array which represents the first image that video captures
print(check)
print(frame)
time.sleep(3)#this will stop the script for three seconds
cv2.imshow("Capturing", frame)
cv2.waitKey(0)
video.release()
cv2.destroyAllWindows()


# How to capture the video instead of first image/frame of the video?
# in order to capture the video , we will be using 'while loop'. while condition will be such that,
# until unless'check' is True, python will display the frames
import cv2, time
video = cv2.VideoCapture(0)
a=1
while True:# this will iterate through the frames and display the window
	a = a+1
	check, frame = video.read()
	print(frame)
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#convert each frame to gray scale image
	cv2.imshow("Capturing", gray)
	key = cv2.waitKey(1)#this will generate a new frame after every 1 milliseconds
	if key == ord('q'):
		break#once you entered 'q' the window will be destroyed
print(a)#this will print the number of frames
video.release()
cv2.destroyAllWindows()
	

#Use Case - Motion Detector
#SOLUTION LOGIC
# start->save the initial image in a frame ->Convert this image to a gaussian blur image(image without object)
# take the frame with the object annd convert it into gaussian blur image -> calculate the difference
# Define the threshold to remove the shadows and other noises ->Define the borders of the object
# Add the rectangular boc around the object ->Calculate the time when object appears and exits the frame
import cv2, time
first_frame = None
video = cv2.VideoCapture(0)#create a Videocapture oject to record video using web cam
while True:
	check, frame = video.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#convert the frame color into gray scale
	gray = cv2.GaussianBlur(gray,(21,21),0)#convert the gray scale frame into GaussianBlur
	if first_frame is None:#this is used to store the first image/frame of the video
		first_frame = gray
		continue
delta_frame = cv2.absdiff(first_frame,gray)#calculate difference betweeb first frame and ither frames
thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]#provides a threshold value, such that 
# it will convert the difference value with less than 30 to black, if grater than 30, those pixels to white
thresh_delta = cv2.dilate(thresh_delta, None, iterations= 0)#ncreases the object area ,Used to accentuate features
(_cnts_) = cv2.findcontours(thres_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#define the contour area
#basically add the borders
for contour in cnts:
	if cv2.contourArea < 1000:#removes noises and shadows.Basically, it will keep only those parts white which
		continue#has area greater than 1000pixels
	#creates a rectangular box around the object in the frame
	(x,y,w,h) = cv2.boundingRect(contour)
	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
	cv2.imshow('frame',frame)
	cv2.imshow('Capturing',gray)
	cv2.imshow('delta',delta_frame)
	cv2.imshow('thres',thresh_delta)
	key = cv2.waitKey(1)#frame will change in 1 millisecond
	if key == ord('q'):#this will break the loop once the user press 'q'
		break
video.release()
cv2.destroyAllWindows()#this will close all the windows


# Now, we can calculate the time for which the objectwas in the fronyt of the camera
# storing time values
first_frame = None
status_list = [None, None]
times = []
# Dataframe to store the time values during which the object detection and movement appears
df=pandas.DataFrame(columns=["Start", "End"])
video = cv2.VideoCapture(0)
while True:
	check,frame = video.read
	status = 0#status at the beginning of the recording is zero as the object is not visible
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21,21),0)
	(_,cnts,_) = cv2.findcontours(thres_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for contour in cnts:
		if cv2.contourArea(contour) < 1000:
			continue
		status = 1#change in status when object is being deetected
		(x,y,w,h) = cv2.boundingRect(contour)
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
	status_list.append(status)#list of status for every frame
	status_list=status_list[-2:]
	#record datetime when change occurs
	if status_list[-1] == 1 and status_list[-2] == 0:
		times.append(datetime.now())
	if status_list[-1] == 0 and status_list[-2] == 1:
		times.append(datetime.now())
	print(status_list)
	print(times)
	for i in range(0, len(times), 2):
		df=df.append({"Start":times[i], "End":times[i+1]}, ignore_index=True)#store times value in a dataframe
	df.to_csv("Times.csv")#write the dataframe to a csv files
video.release()
cv2.destroyAllWindows()

# Plotting the Motion Detection Graph
# Import the DataFrame from the motion_detector.py
from motion_detector import df
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource

df["Start_string"]=df["Start"].dt.strftime("%Y-%m-%d %H:%M:%S")# convert time to a sring format
df["End_string"]=df["End"].dt.strftime("%Y-%m-%d %H:%M:%S")
cds=ColumnDataSource(df)
p=figure(x_axis_type='datetime',height=100, width=500, responsive=True, title="Motion Graph")
p.yaxis.minor_tick_line_color=None
p.ygrid[0].ticker.desired_num_ticks=1
# The DataFrame of time values is plotted on the browser using Bokeh plots
hover=HoverTool(tooltips=[("Start","@Start_string"), ("End","@End_string")])
p.add_tools(hover)
q=p.quad(left="Start",right="End",bottom=0,top=1,color="red",source=cds)
output_file("Graph.html")
show(p)