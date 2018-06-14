# import the necessary packages
from imutils.face_utils.helpers import FACIAL_LANDMARKS_IDXS
from imutils.face_utils.helpers import shape_to_np
from imutils.face_utils import rect_to_bb
import numpy as np
import imutils
import dlib
import cv2
import os
from shutil import copy
import matplotlib.pyplot as plt

p = 1.0 # weight for classification
data_num = 10 # number of data
alignDir = os.path.abspath(os.path.join('.', 'aligned'))
notAlignDir = os.path.abspath(os.path.join('.', 'not aligned'))
imageDir = os.path.abspath(os.path.join('.', 'image'))
if not os.path.exists(alignDir):
    os.mkdir(alignDir)
if not os.path.exists(notAlignDir):
    os.mkdir(notAlignDir)
leftEyes = np.zeros(shape = [data_num, 2], dtype = int)
rightEyes = np.zeros(shape = [data_num, 2], dtype = int)
leftEyeStat = []
rightEyeStat = []
alignedLeftEyes = np.zeros(shape = 0, dtype = int)
notAlignedLeftEyes = np.zeros(shape = 0, dtype = int)
alignedRightEyes = np.zeros(shape = 0, dtype = int)
notAlignedRightEyes = np.zeros(shape = 0, dtype = int)
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
        leftEyes[number] = leftEyeCenter
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        #print(rightEyeCenter)
        rightEyes[number] = rightEyeCenter
        '''
        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180
        
        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
 
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist
        
        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
 
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
 
        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        
        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)
 
        # return the aligned face
        return output
        
        '''
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
        alignedLeftEyes = np.append(alignedLeftEyes, leftEyes[number-1], axis = 0)
        #alignedRightEyes = np.append(alignedRightEyes, rightEyes[number-1])
    else :
        notAlignedLeftEyes = np.append(notAlignedLeftEyes, leftEyes[number-1], axis = 0)
        #notAlignedRightEyes = np.append(notAlignedRightEyes, rightEyes[number-1])
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
        faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
        fa.align(image, gray, rect, i - 1)
leftEyeStat.append(np.average(leftEyes[:,0])) # average of x-coord of left eyes
leftEyeStat.append(np.average(leftEyes[:,1])) # average of y-coord of left eyes
leftEyeStat.append(np.std(leftEyes[:,0])) # std.dev of x-coord of left eyes
leftEyeStat.append(np.std(leftEyes[:,1])) # std.dev of y-coord of left eyes
rightEyeStat.append(np.average(rightEyes[:,0])) # average of x-coord of right eyes
rightEyeStat.append(np.average(rightEyes[:,1])) # average of y-coord of right eyes
rightEyeStat.append(np.std(rightEyes[:,0])) # std.dev of x-coord of right eyes
rightEyeStat.append(np.std(rightEyes[:,1])) # std.dev of y-coord of right eyes
for i in range(1, len(leftEyes) + 1) :
    flag = False
    if (leftEyeStat[0] - p*leftEyeStat[2]) <= leftEyes[i-1][0] <= (leftEyeStat[0] + p*leftEyeStat[2]) :
        if (leftEyeStat[1] - p*leftEyeStat[3]) <= leftEyes[i-1][1] <= (leftEyeStat[1] + p*leftEyeStat[3]) : 
            if (rightEyeStat[0] - p*rightEyeStat[2]) <= rightEyes[i-1][0] <= (rightEyeStat[0] + p*rightEyeStat[2]) :
                if (rightEyeStat[1] - p*rightEyeStat[3]) <= rightEyes[i-1][1] <= (rightEyeStat[1] + p*rightEyeStat[3]) : 
                    flag = True
    save(flag, i)
    colors = "r" * data_num + "b" * data_num
print(notAlignedLeftEyes + alignedLeftEyes)
#print(notAlignedLeftEyes[:,0] + alignedLeftEyes[:,0])
#print(notAlignedLeftEyes[:,1] + alignedLeftEyes[:,1])
#plt.scatter(notAlignedLeftEyes[:,0] + alignedLeftEyes[:,0], notAlignedLeftEyes[:,1] + alignedLeftEyes[:,1], color = colors)