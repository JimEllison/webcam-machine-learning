## Importing the required module 
import cv2                      # For webcam
import matplotlib.pyplot as plt # For plot

## Webcam - Initialized part

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

## Analysis - Initialized part

# x-coordinates of left sides of bars  
left = [1, 2, 3, 4, 5] 
  
# heights of bars 
height = [10, 24, 36, 40, 5] 
  
# labels for bars 
tick_label = ['one', 'two', 'three', 'four', 'five'] 
  
# plotting a bar chart 
plt.bar(left, height, tick_label = tick_label, 
        width = 0.8, color = ['red', 'green']) 
  
# naming the x-axis 
plt.xlabel('x - axis') 
# naming the y-axis 
plt.ylabel('y - axis') 
# plot title 
plt.title('My bar chart!') 

plt.show()

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()    

    key = cv2.waitKey(20)

    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")