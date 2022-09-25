import cv2
import numpy as np

point1 = None
point2 = None
cropped = None

font = cv2.FONT_HERSHEY_SIMPLEX
org = (100, 50)
fontScale = 0.9
color = (0, 255, 0)
thickness = 2

# Function for adding text to image
def putText(img, text):
    img = cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    return img


# Function for extracting the color yellow from the image and checking how many pixels of yellow there are and deciding if its enough
def yellowSquareVisible(img):
  # Lower and upper bounds for the yellow box I detected using trackbars
  low_yellow = np.array([16, 87, 0]).reshape((1,1,3))
  high_yellow = np.array([41, 255, 255]).reshape((1,1,3))
  
  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv_img, low_yellow, high_yellow)
  count = np.count_nonzero(mask)
  # print(f"Yellow {count}")
  if count > 6000:
      return True
  return False


# Function for extracting the color red from the image and checking how many pixels of red there are and deciding if its enough
def redStarVisible(img):
  # Lower and upper bounds for the red star I detected using trackbars
  low_red = np.array([141, 70, 0]).reshape((1,1,3))
  high_red = np.array([200, 255, 241]).reshape((1,1,3))
  
  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(hsv_img, low_red, high_red)
  count = np.count_nonzero(mask)
  # print(f"Red {count}")
  if count > 6000:
      return True
  return False


# Creating the cropped image based on mouse click
def crop(event, x,y,flag,param):
  global point1, point2, cropped
  
  if event == cv2.EVENT_LBUTTONDOWN:
    if point1 is None:
      point1 = (x,y)
    elif point2 is None:
      point2 = (x,y)
      
      minY = min(point1[1], point2[1])
      maxY = max(point1[1], point2[1])
      if minY != maxY:
        cropped= image[minY:maxY, :] 
      point1= None
      point2= None
      
    
cv2.namedWindow('Image') 
cv2.setMouseCallback("Image", crop)


# CHANGE IMAGE HERE
image = cv2.imread('./Test_Images/test3.png')

try:
  while(1):  
    cv2.imshow('Image',image)  
    if cropped is not None:
      croppedHSV = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
      cv2.imshow('Crop', cropped)
      
      # Dividing image into three parts horizontally
      width = int(cropped.shape[1]/3)
      square1 = cropped[:, :width]
      square2 = cropped[:, width: width*2]
      square3 = cropped[:, width*2: ]
      
      
      # Checks each segment (or square) for a yellow square and red star and prints accordingly
      if yellowSquareVisible(square1):
        print("Square 1 has a Yellow Square")
        square1 = putText(square1, "Yellow Square visible")

      if yellowSquareVisible(square2):
        print("Square 2 has a Yellow Square")
        square2 = putText(square2, "Yellow Square visible")

      if yellowSquareVisible(square3):
        print("Square 3 has a Yellow Square")
        square3 = putText(square3, "Yellow Square visible")


      if redStarVisible(square1):
        print("Square 1 has a Red Star")
        square1 = putText(square1, "Red Star visible")

      if redStarVisible(square2):
        print("Square 2 has a Red Star")
        square2 = putText(square2, "Red Star visible")

      if redStarVisible(square3):
        print("Square 3 has a Red Star")
        square3 = putText(square3, "Red Star visible")

      cv2.imshow('Square 1',square1)  
      cv2.imshow('Square 2',square2)  
      cv2.imshow('Sqaure 3',square3) 

      cropped= None
    if cv2.waitKey(1) & 0xFF == 27:  
        break 
except Exception as e:
  print(e)

cv2.destroyAllWindows()
