# Utility Commands
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Reading and displaying image with numpy/matplot
im2disp = cv2.imread("ahmad.jpg")
plt.figure()
plt.imshow(im2disp)

# Reading and displaying image with OpenCV
cvim2disp = cv2.imread('ahmad.jpg')
cv2.imshow('Hello World', cvim2disp)
cv2.waitKey() # Image will not show until this is called
cv2.destroyWindow('HelloWorld') # Make shure window closes cleanly

# Are numpy/cv2 the same?
plt.figure()
imshow(cvim2disp)

# Convert BGR format of openCV to RGB
cvimrgb = cv2.cvtColor(cvim2disp,cv2.COLOR_BGR2RGB)
plt.figure()
imshow(cvimrgb)

# Useful utility function
def mycvshow(imagein, title='Image'):
    cv2.imshow(title, imagein)
    cv2.waitKey()
    cv2.destroyWindow(title)

# Filtering example
im2disp = imread('ahmad.jpg')
blurred = cv2.GaussianBlur(im2disp,(19,19),0)

# More general method
kernel = np.ones((5,5), np.float32)/25
blurred2 = cv2.filter2D(im2disp, -1, kernel)

figure()
imshow(blurred2)

# Saving an image
cv2.imwrite('mycvimage.png', cvim2disp)
# or
imsave('myimage.png', im2disp)