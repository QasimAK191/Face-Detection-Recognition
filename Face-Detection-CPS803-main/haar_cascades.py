import cv2 as cv
import matplotlib.pyplot as plt

import util
from model import Model


def blur(img, x, y, w, h):
    updated = img.copy()
    blurred = cv.GaussianBlur(updated, (75, 75), 0)
    # Swap x y for image indexing
    updated[y:y+h, x:x+w] = blurred[y:y+h, x:x+w]
    return updated


class HaarCascades(Model):

    def __init__(self):
        super().__init__()
        self.model = None
        self.images = None

    def train(self):
        harr_model_path = 'models/haarcascade_frontalface_default.xml'
        face_cascade = cv.CascadeClassifier(harr_model_path)
        self.model = face_cascade

    def predict(self):
        sample_result = True    # Show the model result for 2 images
        if sample_result:
            self.images = self.read_sample_images()

        i = 0
        while i < len(self.images):
            # Detect boxes and save to Image
            gray_img = cv.cvtColor(self.images[i].original, cv.COLOR_BGR2GRAY)
            boxes = self.model.detectMultiScale(gray_img,
                                                scaleFactor=1.3,
                                                minNeighbors=5)
            self.images[i].boxes = boxes

            if self.images[i].boxes != ():
                #print(i, ": Predicted", self.images[i].bounding_boxes)
                #print(i, ": True Box ", self.images[i].true_boxes)

                # Add blur to image for each detected box
                blur_img = self.images[i].original.copy()
                for detected_box in self.images[i].boxes:
                    x, y, w, h = detected_box
                    blur_img = blur(blur_img, x, y, w, h)

                # Save blurred image on Image
                self.images[i].blurred = blur_img

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
                    (x, y, x1, y1) = detected_box.astype("int")
                    pr = cv.rectangle(self.images[i].original, (x,y),(x+x1,y+y1),(0,255,0),3)
                cv.imshow('Predicted', pr)
                cv.waitKey(0)
                    # Show image after blurring
                cv.imshow('Blurred', self.images[i].blurred)
                cv.waitKey(0)
            i += 1
