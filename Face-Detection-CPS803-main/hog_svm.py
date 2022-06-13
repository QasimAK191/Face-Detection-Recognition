import face_recognition
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

import util
from model import Model


def blur(img, x, y, w, h):
    updated = img.copy()
    blurred = cv.GaussianBlur(updated, (75, 75), 0)
    # Swap x y for image indexing
    updated[x:w, y:h] = blurred[x:w, y:h]
    return updated


class HOG_SVM(Model):

    def __init__(self):
        super().__init__()
        self.model = None
        self.images = None

    def train(self):
        self.model = face_recognition

    def predict(self):
        sample_result = True    # Show the model result for 2 images
        if sample_result:
            self.images = self.read_sample_images()

        i = 0
        while i < len(self.images):
            # Detect boxes and save to Image
            boxes = self.model.face_locations(self.images[i].original)
            self.images[i].boxes = boxes

            if self.images[i].boxes != ():
                # Add blur to image for each detected box
                blur_img = self.images[i].original.copy()
                for detected_box in self.images[i].boxes:
                    a1, a2, a3, a4 = detected_box
                    self.images[i].boxes = [[a4,a1,(a2-a4),(a3-a1)]]
                    #print("box ",i,": ", detected_box)
                    blur_img = blur(blur_img, a1, a4, a3, a2)

                # Save blurred image on Image
                self.images[i].blurred = blur_img
                #print(i)
                #print(detected_box)
                #show = cv.rectangle(blur_img, (a4, a1),(a2, a3), (0, 255, 0), 3)
                #a,b,c,d = self.images[i].true_boxes[0]
                #show = cv.rectangle(show, (a,b),(a+c,b+d),(255,0,0),3)
                #print(self.images[i].true_boxes[0])
                #print()
                #cv.imshow('lines',show)
                #cv.waitKey(0)
                # Display image
                #cv.imshow("Blurred image", self.images[i].blurred)
                #cv.waitKey(0)
                #if i > 0:
                    #break
            if sample_result:
                # Show original image with true box
                ox = self.images[i].true_boxes[0][0]
                oy = self.images[i].true_boxes[0][1]
                ox1 = self.images[i].true_boxes[0][2]
                oy1 = self.images[i].true_boxes[0][3]
                cv.imshow('Original', self.images[i].original)
                cv.waitKey(0)
                og = cv.rectangle(self.images[i].original, (ox,oy),(ox+ox1,oy+oy1),(0,0,255),3)
                # cv.imshow('Original', og)
                # cv.waitKey(0)
                pr = self.images[i].original
                for detected_box in self.images[i].boxes:
                    # Show image with predicted bounding box
                    (x, y, x1, y1) = detected_box
                    pr = cv.rectangle(self.images[i].original, (x,y),(x+x1,y+y1),(0,255,0),3)
                cv.imshow('Predicted', pr)
                cv.waitKey(0)
                # Show image after blurring
                cv.imshow('Blurred', self.images[i].blurred)
                cv.waitKey(0)

            i += 1
