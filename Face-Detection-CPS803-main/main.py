import haar_cascades
import hog_svm
import DNN
import faceDet
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # Haar Cascades using OpenCV
    print("Haar Cascades model")
    hc = haar_cascades.HaarCascades()
    hc.read_preprocess()
    hc.train()
    hc.predict()
    mp1, cov1, fpr1 = hc.evaluate()

    # HOG SVM using facial_recognition Dlib package
    print("HOG SVM model")
    hog = hog_svm.HOG_SVM()
    hog.read_preprocess()
    hog.train()
    hog.predict()
    mp2, cov2, fpr2 = hog.evaluate()

    # DNN (Same models)
    # DNN using opencv with res10 caffe model
    print("DNN model with OpenCV res10")
    myDNN = DNN.DNN()
    myDNN.read_preprocess()
    myDNN.train()
    myDNN.predict()
    myDNN.draw_faces()
    mp3, cov3, fpr3 = myDNN.evaluate()

    # DNN using opencv with face_detection caffe model
    print("DNN with openCV face detection model")
    my_face_Det = faceDet.FaceDetDNN()
    my_face_Det.read_preprocess()
    my_face_Det.train()
    my_face_Det.predict()
    my_face_Det.draw_faces()
    mp4, cov4, fpr4 = my_face_Det.evaluate()

    #plot section
    labels = ['Haar Cascade', 'HOG SVM', 'DNN I', 'DNN II' ]
    mp = [mp1, mp2, mp3, mp4]
    cov_acc = [cov1, cov2, cov3, cov4]
    fpr = [fpr1, fpr2, fpr3, fpr4]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/3, mp, width, label='Mean Average Precision')
    rects2 = ax.bar(x, cov_acc, width, label='Coverage Accuracy')
    rects2 = ax.bar(x + width/3, fpr, width, label='False Positive Rate')

    ax.set_ylabel('Scores')
    ax.set_title('Model Scores')
    ax.set_xticks(x, labels)
    ax.legend()

    plt.savefig('Model Scores.pdf')
