import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

import util
from model import Model
from image import Image

modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "models/deploy.prototxt"


def blur(img, x, y, w, h):
	updated = img.copy()
	blurred = cv.GaussianBlur(updated, (75, 75), 0)
	blurred = cv.GaussianBlur(blurred, (75, 75), 0)
	blurred = cv.GaussianBlur(blurred, (75, 75), 0)
	updated[y:h, x:w] = blurred[y:h, x:w]
	return updated


class DNN(Model):
	def __init__(self):
		super().__init__()
		self.model = None
		self.images = None

	def train(self):
		self.model = cv.dnn.readNetFromCaffe(configFile, modelFile)

	def predict(self):
		sample_result = True    # Show the model result for 2 images
		if sample_result:
			self.images = self.read_sample_images()

		for img in self.images:
			im_blob = cv.dnn.blobFromImage(cv.resize(img.original, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
			self.model.setInput(im_blob)
			img.boxes = self.model.forward()

			

	def draw_faces(self):
		sample_result = True    # Show the model result for 2 images
		for im in self.images:
			h, w = im.original.shape[:2]
			confidence = im.boxes[0, 0, 0, 2]
			if confidence > 0.95:
				box = im.boxes[0, 0, 0, 3:7] * np.array([w, h, w, h])
				(x, y, x1, y1) = box.astype("int")
				im.blurred = blur(im.original, x, y, x1, y1)
				if sample_result:
					# Show original image with true box
					ox = im.true_boxes[0][0]
					oy = im.true_boxes[0][1]
					ox1 = im.true_boxes[0][2]
					oy1 = im.true_boxes[0][3]
					cv.imshow('Original', im.original)
					cv.waitKey(0)
					og = cv.rectangle(im.original, (ox,oy),(ox+ox1,oy+oy1),(0,0,255),3)
					pr = im.original
					for detected_box in im.boxes:
						# Show image with predicted bounding box
						h, w = im.original.shape[:2]
						bb = detected_box[0, 0, 3:7] * np.array([w,h,w,h])
						(x, y, x1, y1) = bb.astype("int")
						pr = cv.rectangle(im.original, (x,y),(x1,y1),(0,255,0),3)
					cv.imshow('Predicted', pr)
					cv.waitKey(0)
					#Show image after blurring
					cv.imshow('Blurred', im.blurred)
					cv.waitKey(0)
				# show = cv.rectangle(im.blurred, (x, y),(x1, y1), (0, 255, 0), 3)
				# a,b,c,d = im.true_boxes[0]
				# show = cv.rectangle(im.blurred, (a,b),(a+c,b+d),(255,0,0),3)
				# print(box)
				# print(im.true_boxes)
				# cv.imshow('lines',show)
				# cv.waitKey(0)
