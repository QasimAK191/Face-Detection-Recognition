import os
import cv2 as cv

import preprocess
import util
from image import Image


class Model:
    def __init__(self):
        self.model = None
        self.images = None

    def read_preprocess(self):
        images_dict = preprocess.read_image_data()
        images = []
        for filepath in images_dict:
            if os.path.isfile(filepath):
                # Ensure image file exists
                image = cv.imread(filepath)
                img_obj = Image(image, true_boxes=images_dict[filepath])
                images.append(img_obj)
        self.images = images

    def read_sample_images(self):
        images_dict = preprocess.read_image_data()
        ret_images = []
        im_1 = 'images/13--Interview/13_Interview_Interview_2_People_Visible_13_154.jpg'
        im_2 = 'images/9--Press_Conference/9_Press_Conference_Press_Conference_9_935.jpg'
        if im_1 in images_dict and os.path.isfile(im_1):
            image = cv.imread(im_1)
            img_obj = Image(image, true_boxes=images_dict[im_1])
            ret_images.append(img_obj)
        if im_2 in images_dict and os.path.isfile(im_2):
            image = cv.imread(im_2)
            img_obj = Image(image, true_boxes=images_dict[im_2])
            ret_images.append(img_obj)
        return ret_images

    def evaluate(self):
        util.calcError(self.images)
        print("Calculating mean average precision")
        print("map: ", util.mAP(self.images))
        print("Calculating average coverage")
        print("avg accuracy: ", util.coverageAccuracy(self.images))
        print("Calculating false positive rate")
        print("fp: ", util.fpr(self.images))
        print()
        print()
        return util.mAP(self.images), util.coverageAccuracy(self.images), util.fpr(self.images)
