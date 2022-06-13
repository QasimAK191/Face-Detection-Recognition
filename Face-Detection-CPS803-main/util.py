import numpy as np
from shapely.geometry import Polygon

def calcError(images):
	for im in images:
		if ((im.boxes == () or im.boxes == []) and im.true_boxes == []):
			im.error = 1.0
		elif (im.boxes == () or  im.boxes == []):
			im.error = 0
		else:
			if np.shape(im.boxes) == (1,1,200,7):
				h, w = im.original.shape[:2]
				holder = im.boxes[0,0,0, 3:7] * np.array([w,h,w,h])
				box = calculate_box2(holder)
				trueBox = calculate_box(im.true_boxes[0])
				im.error = iou(box,trueBox)
			else:
				box = calculate_box(im.boxes[0])
				trueBox = calculate_box(im.true_boxes[0])
				im.error = iou(box,trueBox)

def calculate_box(arr):
	x,y,w,h = arr
	bottom_left = [x,y]
	top_left = [x,y+h]
	top_right = [x+w,y+h]
	bottom_right = [x+w,y]
	return [top_left, top_right, bottom_right, bottom_left]

def calculate_box2(arr):
	x,y,x1,y1 = arr
	top_left = [x,y]
	bottom_left = [x,y1]
	top_right = [x1,y]
	bottom_right = [x1,y1]
	return [top_left,top_right,bottom_right,bottom_left]

def iou(box1, box2):
	p1 = Polygon(box1)
	p2 = Polygon(box2)
	ret = p1.intersection(p2).area / p1.union(p2).area
	return ret 

def mAP(images):
	threshold = 0.5
	TP = 0
	FP = 0
	#precision = TP / (TP + FP)
	for im in images:
		if (im.error >= 0.5):
			TP += 1
		elif (im.error < 0.5):
			FP += 1

	return (TP / (TP+FP +1e-16))

def fpr(images):
	threshold = 0.5
	TP = 0
	FP = 0
	#precision = TP / (TP + FP)
	for im in images:
		if (im.error < 0.5):
			FP += 1

	return (FP / (len(images) +1e-16))

def coverageAccuracy(images):
	total = 0
	count = 1
	for im in images:
		total += im.error
		count += 1
	return total / count
