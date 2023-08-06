import cv2
import matplotlib.pyplot as plt
import os

image_path = 'img1.jpg'

img = cv2.imread(image_path) # Reading the image as an array
print(img)

print(img.shape)# Dimensions of the image

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray_image.shape)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
) # We can change the classifiers here to detect different parts of the face

face = face_classifier.detectMultiScale(
    gray_image, 
    scaleFactor = 1.1, 
    minNeighbors = 5, # Minimum number of neighbors
    minSize = (40, 40)
)

for (x, y, w, h) in face:
    cv2.rectangle(img, (x ,y), (x + w, y + h), (0, 255, 0), 4)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize = (20, 10))
plt.imshow(img_rgb)
plt.axis('off')

# path = 'img1.jpg'
# cv2.imwrite(os.path.join(path , 'test1.jpg'), img)

cv2.imwrite('test1.jpg', img)
cv2.waitKey(0)