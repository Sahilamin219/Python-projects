# # y = np.arange(3, dtype=float)
# # >>> y
# # array([0., 1., 2.])
# # >>> np.zeros_like(y)
# # array([0.,  0.,  0.])
# import cv2
# img = cv2.imread('images.jpeg')
# # while True:
# # 	cv2.imshow('images',img)
# # 	key = cv2.waitKey(1)
# # 	print(key)
# # 	if cv2.waitKey(1) & 0xFF == 27:
# # 		break
# # cv2.destroyAllwindows()
# # saving of image
# cv2.imwrite('final_image.jpeg',img)
# # Basic Operation on images
# import numpy as np
# import matplotlib.pyplot as plt
# # %matplotlib inline sets the backend of matplotlib to the
# # 'inline' backend: With this backend, the output of plotting
# # commands is displayed inline within frontends like the
# # Jupyter notebook, directly below the code cell that produced it. 
# # The resulting plots will then also be stored in the notebook document
# # %matplotlib inline
# # creating a black image will act as a template
# image_blank = np.zeros(shape=(512,512,3),dtype=np.int16)
# #The desired data-type for the array, e.g., numpy.int8. Default is numpy.float64.
# # displaying black image
# plt.imshow(image_blank)


#sentex course
import cv2
import numpy
face_cascade = cv2.CascadeClassifier('/home/sahil/Documents/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/sahil/Documents/harcascade_eye.xml')
test = face_cascade.load('haarcascade_frontalface_default.xml')
print(test)
test = face_cascade.load('harcascade_eye.xml')
print(test)
# returns false
cap = cv2.VideoCapture(0)
while True:
	ret,img = cap.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#faces = face_cascade.detectMultiScale(gray,1.3,5)
	faces = face_cascade.detectMultiScale(gray,1.3,5)
	for(x,y,w,h) in faces:
		cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for(ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh), (0,255,0), 2)

	cv2.imshow('img',img)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
	elif k == ord('s'): # wait for 's' key to save and exit
		cv2.imwrite('messigray.png',img)
		cv2.destroyAllWindows()
cap.release()
cap.destroyAllwindows()