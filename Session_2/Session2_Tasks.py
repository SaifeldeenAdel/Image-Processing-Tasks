import cv2
import numpy as np


#-----------
# TASK 1 

def colorToGrey(img):
    img = img.astype('uint16')
    img = (img[:,:,0] + img[:,:,1] + img[:,:,2]) / 3
    img = img.astype('uint8')
    cv2.imshow('image', img)


# UNCOMMENT BELOW TO RUN TASK 1
# image = cv2.imread('./Images/art2.jpeg')
# colorToGrey(image)
# cv2.waitKey(0)

#-----------

#------------
# TASK 2

# Global variables
min_hue = 0
max_hue = 0
min_sat = 0
max_sat = 0
min_val = 0
max_val = 0

# Trackbar callback functions to assign values to variables
def on_min_hue_trackbar(val):
  global min_hue
  min_hue = val

def on_max_hue_trackbar(val):
  global max_hue
  max_hue = val

def on_min_sat_trackbar(val):
  global min_sat
  min_sat = val
  
def on_max_sat_trackbar(val):
  global max_sat
  max_sat = val

def on_min_val_trackbar(val):
  global min_val
  min_val = val
  
def on_max_val_trackbar(val):
  global max_val 
  max_val = val

# Function for creating all trackbars
def makeTrackBars():
  cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
  cv2.createTrackbar('Min Hue', "Trackbars" , 0, 255, on_min_hue_trackbar)
  cv2.createTrackbar('Max Hue', "Trackbars" , 0, 255, on_max_hue_trackbar)
  cv2.createTrackbar('Min Sat', "Trackbars" , 0, 255, on_min_sat_trackbar)
  cv2.createTrackbar('Max Sat', "Trackbars" , 0, 255, on_max_sat_trackbar)
  cv2.createTrackbar('Min Value', "Trackbars" , 0, 255, on_min_val_trackbar)
  cv2.createTrackbar('Max Value', "Trackbars" , 0, 255, on_max_val_trackbar)


# Function for extracting color from image
def extractColor(img, min, max):
  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv_img, min, max)
  extracted_img= cv2.bitwise_and(img,img,mask=mask)
  return extracted_img


def videoColorExtraction():
  cap = cv2.VideoCapture(0)
  while True:
    lowColor = np.array([min_hue, min_sat, min_val]).reshape((1,1,3))
    highColor = np.array([max_hue, max_sat, max_val]).reshape((1,1,3))
    ret, frame = cap.read()
    frame = np.flip(frame, 1)
    cv2.imshow('Original', frame)
    cv2.imshow('Extracted', extractColor(frame, lowColor, highColor))
    if cv2.waitKey(1) == 27:
        break
  cv2.destroyAllWindows()
  cap.release() 


# UNCOMMENT BELOW TO RUN TASK 2
# makeTrackBars()
# videoColorExtraction()


#------------

