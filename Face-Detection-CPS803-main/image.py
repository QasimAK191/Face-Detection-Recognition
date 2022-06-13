import cv2 as cv

class Image:
	def __init__(self, img, true_boxes=None):
		"""
		Instantiates an Image object containing image, blurred image, the
			predicted bounding boxes, the true boxes, and the accuracy error
			from mean average precision
		:param img: the read image
		:param true_boxes: the true bounding boxes of all faces in the image
							of format [ [x,y,width,height], ]
		"""
		self.original = cv.normalize(src=img, dst=None, alpha=0, beta=255,
									 norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
		self.blurred = []
		self.boxes = []
		self.true_boxes = true_boxes if true_boxes is not None else []
		self.error = 0
