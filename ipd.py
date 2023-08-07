import cv2
import dlib
import matplotlib.pyplot as plt
import math
import os

# image_path = 'img1.jpg'

# img = cv2.imread(image_path) # Reading the image as an array
# print(img)

# print(img.shape)# Dimensions of the image

# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(gray_image.shape)

# face_classifier = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# ) # We can change the classifiers here to detect different parts of the face

# face = face_classifier.detectMultiScale(
#     gray_image, 
#     scaleFactor = 1.1, 
#     minNeighbors = 5, # Minimum number of neighbors
#     minSize = (40, 40)
# )

# for (x, y, w, h) in face:
#     cv2.rectangle(img, (x ,y), (x + w, y + h), (0, 255, 0), 4)

# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# plt.figure(figsize = (20, 10))
# plt.imshow(img_rgb)
# plt.axis('off')

# # path = 'img1.jpg'
# # cv2.imwrite(os.path.join(path , 'test1.jpg'), img)

# cv2.imwrite('test1.jpg', img)
# cv2.waitKey(0)

# For real time

# Declaring Classifier
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Declaring classifier to check if the user is facing the camera
landmark_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') 

# Declaring a classifier to check if the user is smiling
smile_classfier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Declaring a classifier to detect eye 
eye_classfier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Declaring a classifier to detect is mouth is open
mouth_classifier  =cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
# Accessing webcam
web_cam = cv2.VideoCapture(0)


def calculate_angle(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    angle = math.atan2(y2 - y1, x2 - x1)
    return math.degrees(angle)

# Displaying the box and setting the coordinates 
def faceDetect(cam):
    vid_gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
    vid_rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

    faces = face_classifier.detectMultiScale(
        vid_rgb,
        1.1,
        5,
        minSize = (40, 40)
    )

    for (x, y, w, h) in faces:
        face_img = cam[y:y + h,x:x + w]

        landmarks = landmark_detector(
            vid_rgb,
            dlib.rectangle(
                x,
                y,
                x + w,
                y + h
            )
        )

        # Getting coordinates of facial landmarks
        left_eye = (landmarks.part(10).x, landmarks.part(15).y)
        right_eye =(landmarks.part(45).x, landmarks.part(45).y)
        nose_tip = (landmarks.part(30).x, landmarks.part(30).y)

        # Calculating angle between eyes and nose
        angle_left = calculate_angle(left_eye, nose_tip)
        angle_right = calculate_angle(right_eye, nose_tip)

        # Averaging to calculate if the angle on the right 
        # and the left are the same to determine if the user is facing the camera
        avg_angle = (angle_left + angle_right) / 2

        # Displaying angle in the frame
        cv2.putText(
            cam,
            f'Angle: {avg_angle:.2f}',
            (x, y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        if len(faces) > 0:
            cv2.putText(
                cam,
                'Facing camera',
                (x, y - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        elif abs (avg_angle) >= 30:
            cv2.putText(
                cam,
                'Not Facing Camera',
                (x, y - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0 , 0),
                2
            )

        cv2.rectangle(
            cam,
            (x , y),
            (x + w, y + h),
            (0, 255, 0),
            4
        )
        
        # For eyes detection
        eyes = eye_classfier.detectMultiScale (
            vid_gray[y:y + h, x: x + w],
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize=(40, 40)
        )

        # Detecting the eyes of the user
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                cam,
                (x + ex, y + ey),
                (x + ex + ew, y + ey + eh),
                (0, 0, 255),
                2
            )

        # For smile detection
        smile = smile_classfier.detectMultiScale(
            vid_gray[y: y + h, x: x + w],
            scaleFactor = 1.8,
            minNeighbors = 20,
            minSize = (40, 40)
        )
             
        # Determining if the user is smiling or not
        if len(smile) > 0:
            cv2.putText(
                cam,
                'Smiling',
                (x, y - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )
        else:
            cv2.putText(
                cam,
                'Not Smiling',
                (x, y - 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )
        
        # Determining is the mouth is open
        mouth = mouth_classifier.detectMultiScale(
            vid_gray[y: y + h, x: x + w],
            scaleFactor = 1.8,
            minNeighbors = 20,
            minSize = (30, 30)
        )

        for (mx, my, mw, mh) in mouth:
            cv2.rectangle(
                cam,
                (x + mx, y + my),
                (x + mx + mw, y + my + mh),
                (255, 0, 0),
                2
            )
        
        if len(mouth) > 0:
            cv2.putText(
                cam,
                'Mouth is open',
                (x, y - 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )
        else:
             cv2.putText(
                cam,
                'Mouth is not open',
                (x, y - 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )


    return cam, faces



while True:

    result, videoFrame = web_cam.read()

    if result is False:
        break

    faces = faceDetect(
        videoFrame
    )

    cv2.imshow(
        "Face Detection",
        videoFrame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

web_cam.release()
cv2.destroyAllWindows()

