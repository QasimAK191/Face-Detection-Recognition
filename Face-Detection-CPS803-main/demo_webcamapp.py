import face_recognition
import cv2 as cv
import numpy as np
import hog_svm
import random

video_capture = cv.VideoCapture(0)

hog = hog_svm.HOG_SVM()
hog.read_preprocess()
hog.train()

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def blur(img, x, y, w, h):
    if x and y != None:
        updated = img.copy()
        blurred = cv.GaussianBlur(updated, (75, 75), 0)
        blurred = cv.GaussianBlur(blurred, (275, 275), 0)
        # Swap x y for image indexing
        updated[x:w, y:h] = blurred[x:w, y:h]
    return updated

while True:
    ret, frame = video_capture.read()

    small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        # Find all the faces in the current frame of video
        face_locations = hog.model.face_locations(rgb_small_frame)
        face_encodings = hog.model.face_encodings(rgb_small_frame, face_locations)

    process_this_frame = not process_this_frame

    switch = False
    # Display the results
    for (top, right, bottom, left) in face_locations:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        

        # Draw a box around the face
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        
        frame = blur(frame, top, left, bottom, right)


    cv.imshow('Face Detection Demo - Press q to quit', frame)

    # q to quit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv.destroyAllWindows()
