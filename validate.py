# import the necessary packages
from imutils.face_utils.helpers import FACIAL_LANDMARKS_IDXS
from imutils.face_utils.helpers import shape_to_np
from imutils.face_utils import rect_to_bb
import matplotlib.pyplot as plt
import numpy as np
import imutils
import dlib
import cv2
import os
import shutil

p = 1.0 # weight for classification
data_num = 1000 # number of data
alignDir = os.path.abspath(os.path.join('.', 'aligned'))
notAlignDir = os.path.abspath(os.path.join('.', 'not aligned'))
imageDir = os.path.abspath(os.path.join('.', 'image_test'))
if os.path.exists(alignDir):
    shutil.rmtree(alignDir)
os.mkdir(alignDir)
if os.path.exists(notAlignDir) :
    shutil.rmtree(notAlignDir)
os.mkdir(notAlignDir)
leftEyes = [0]*data_num
rightEyes = [0]*data_num
leftEyeStat = []
rightEyeStat = []
alignedLeftEyes = []
notAlignedLeftEyes = []
alignedRightEyes = []
notAlignedRightEyes = []
class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=None, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
 
        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth
            
    def align(self, image, gray, rect, number):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)
 
        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]
        
        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        #print(leftEyeCenter)
        leftEyes[number] = leftEyeCenter.tolist()
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        #print(rightEyeCenter)
        rightEyes[number] = rightEyeCenter.tolist()
''' previous save function(original)
def save(is_aligned, number) :
    if is_aligned is True :
        shutil.copy(os.path.join(imageDir, "1_0" + "%04d" % (i,)  + ".jpg"), alignDir)
    else :
        shutil.copy(os.path.join(imageDir, "1_0" + "%04d" % (i,)  + ".jpg"), notAlignDir)
'''
def save(is_aligned, number) :
    global alignedLeftEyes
    global notAlignedLeftEyes
    global alignedRightEyes
    global notAlignedRightEyes
    if is_aligned is True :
        alignedLeftEyes.append(leftEyes[number-1])
        alignedRightEyes.append(rightEyes[number-1])
    else :
        notAlignedLeftEyes.append(leftEyes[number-1])
        notAlignedRightEyes.append(rightEyes[number-1])
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("predictor.dat")
for i in range(1, data_num + 1) :
    img_name = "image_test/1_0" + "%04d" % (i,)  + ".jpg"
    image = cv2.imread(img_name)
    fa = FaceAligner(predictor, desiredFaceWidth=image.shape[:2][0])
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 2)
    #print(img_name)
    for rect in rects:
       # print(rect)
        (x, y, w, h) = rect_to_bb(rect)
        #print((x,y,w,h))
        #faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        fa.align(image, gray, rect, i - 1)
while 0 in leftEyes :
    leftEyes.remove(0)
while 0 in rightEyes :
    rightEyes.remove(0)    
leftEyes_arr = np.array(leftEyes)
rightEyes_arr = np.array(rightEyes)
leftEyeStat.append(np.average(np.array(leftEyes)[:,0])) # average of x-coord of left eyes
leftEyeStat.append(np.average(np.array(leftEyes)[:,1])) # average of y-coord of left eyes
leftEyeStat.append(np.std(np.array(leftEyes)[:,0])) # std.dev of x-coord of left eyes
leftEyeStat.append(np.std(np.array(leftEyes)[:,1])) # std.dev of y-coord of left eyes
rightEyeStat.append(np.average(np.array(rightEyes)[:,0])) # average of x-coord of right eyes
rightEyeStat.append(np.average(np.array(rightEyes)[:,1])) # average of y-coord of right eyes
rightEyeStat.append(np.std(np.array(rightEyes)[:,0])) # std.dev of x-coord of right eyes
rightEyeStat.append(np.std(np.array(rightEyes)[:,1])) # std.dev of y-coord of right eyes
for i in range(1, len(leftEyes) + 1) :
    flag = False
    if (leftEyeStat[0] - p*leftEyeStat[2]) <= leftEyes[i-1][0] <= (leftEyeStat[0] + p*leftEyeStat[2]) :
        if (leftEyeStat[1] - p*leftEyeStat[3]) <= leftEyes[i-1][1] <= (leftEyeStat[1] + p*leftEyeStat[3]) : 
            if (rightEyeStat[0] - p*rightEyeStat[2]) <= rightEyes[i-1][0] <= (rightEyeStat[0] + p*rightEyeStat[2]) :
                if (rightEyeStat[1] - p*rightEyeStat[3]) <= rightEyes[i-1][1] <= (rightEyeStat[1] + p*rightEyeStat[3]) : 
                    flag = True
    save(flag, i)

colors = "r" * len(notAlignedLeftEyes) + "b" * len(alignedLeftEyes) + "r" * len(notAlignedRightEyes) + "b" * len(alignedRightEyes)
color_left = "r" * len(notAlignedLeftEyes) + "b" * len(alignedLeftEyes)
color_right = "m" * len(notAlignedRightEyes) + "c" * len(alignedRightEyes)
#print(alignedLeftEyes)
#print(notAlignedLeftEyes)

if alignedLeftEyes != [] and notAlignedLeftEyes != [] :
    x_left_arr = np.append(np.array(notAlignedLeftEyes)[:,0], np.array(alignedLeftEyes)[:,0])
    y_left_arr = np.append(np.array(notAlignedLeftEyes)[:,1], np.array(alignedLeftEyes)[:,1])
    x_right_arr = np.append(np.array(notAlignedRightEyes)[:,0], np.array(alignedRightEyes)[:,0])
    y_right_arr = np.append(np.array(notAlignedRightEyes)[:,1], np.array(alignedRightEyes)[:,1])
    '''
    x_arr = np.append(leftEyeStat[0]-np.append(np.array(notAlignedLeftEyes)[:,0], np.array(alignedLeftEyes)[:,0]),
                           rightEyeStat[0]-np.append(np.array(notAlignedRightEyes)[:,0], np.array(alignedRightEyes)[:,0]))
    y_arr = np.append(leftEyeStat[1]-np.append(np.array(notAlignedLeftEyes)[:,1], np.array(alignedLeftEyes)[:,1]),
                           rightEyeStat[1]-np.append(np.array(notAlignedRightEyes)[:,1], np.array(alignedRightEyes)[:,1]))
    '''
figure = plt.figure()
plt.scatter(x_left_arr, y_left_arr, color = color_left)
plt.scatter(x_right_arr, y_right_arr, color = color_right)
plt.axvline(x = leftEyeStat[0] + leftEyeStat[2], linewidth = 1, color = 'r')
plt.axvline(x = leftEyeStat[0] - leftEyeStat[2], linewidth = 1, color = 'r')
plt.axhline(y = leftEyeStat[1] - leftEyeStat[3], linewidth = 1, color = 'r')
plt.axhline(y = leftEyeStat[1] + leftEyeStat[3], linewidth = 1, color = 'r')
plt.axvline(x = rightEyeStat[0] + rightEyeStat[2], linewidth = 1, color = 'b')
plt.axvline(x = rightEyeStat[0] - rightEyeStat[2], linewidth = 1, color = 'b')
plt.axhline(y = rightEyeStat[1] - rightEyeStat[3], linewidth = 1, color = 'b')
plt.axhline(y = rightEyeStat[1] + rightEyeStat[3], linewidth = 1, color = 'b')
'''
plt.scatter(x_arr, y_arr, color = colors)
plt.axvline(x = leftEyeStat[2], linewidth = 1, color = 'r')
plt.axvline(x = - leftEyeStat[2], linewidth = 1, color = 'r')
plt.axhline(y = - leftEyeStat[3], linewidth = 1, color = 'r')
plt.axhline(y = leftEyeStat[3], linewidth = 1, color = 'r')
plt.axvline(x = rightEyeStat[2], linewidth = 1, color = 'b')
plt.axvline(x = - rightEyeStat[2], linewidth = 1, color = 'b')
plt.axhline(y = - rightEyeStat[3], linewidth = 1, color = 'b')
plt.axhline(y = rightEyeStat[3], linewidth = 1, color = 'b')
'''
plt.grid(True)
plt.show()